from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from backgrounder.config import PipelineConfig, SegmenterID, Device
from backgrounder.models import BiRefNetSegmenter, BEN2Segmenter, InSPyReNetSegmenter, DepthAnythingV2Small
from backgrounder.models.base import BaseSegmenter
from backgrounder.result import MattingResult
from backgrounder.stages import (
    ensemble_predict,
    generate_trimap,
    unknown_mask,
    depth_aware_refine,
    smooth_alpha_boundary,
    score_alpha,
)
from backgrounder.utils import (
    resolve_device,
    choose_dtype,
    compute_depth_edges,
    estimate_foreground,
    compose_rgba,
    timer,
)


class BackgroundRemovalPipeline:
    """
    Modular background removal pipeline.

    Stage B — Coarse: parallel ensemble of segmenters (BiRefNet_HR + BEN2_Base)
    Stage C — Trimap: BEN2 confidence + inter-model disagreement → unknown band
    Stage D — Refine: depth-aware boundary correction (Depth Anything V2-Small)
    Stage E — Decontam: foreground colour-spill removal
    Stage F — Judge: quality score; retry with wider trimap if score < threshold
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._device = resolve_device(self.config.device.value)
        self._fp16 = self.config.fp16

        self._segmenters: List[BaseSegmenter] = []
        self._depth_model: Optional[DepthAnythingV2Small] = None
        self._ready = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> "BackgroundRemovalPipeline":
        """Eagerly load all models.  Call once at startup to amortise latency."""
        self._build_segmenters()
        for seg in self._segmenters:
            seg.load()
        if self.config.use_depth:
            self._depth_model = DepthAnythingV2Small(
                device=self._device,
                fp16=self._fp16,
                model_id=self.config.depth_model_id,
            )
            self._depth_model.load()
        self._ready = True
        return self

    def process(self, image: Image.Image) -> MattingResult:
        """Run the full pipeline on a single PIL image."""
        if not self._ready:
            self.load()

        meta: dict = {"device": self._device, "timings_ms": {}}
        t_total = time.perf_counter()

        # Stage B — Coarse ensemble
        with timer("stage_B_ensemble", meta["timings_ms"]):
            alpha, uncertainty, outputs = ensemble_predict(
                self._segmenters,
                image,
                parallel=True,
            )

        # BEN2 confidence map (if available)
        ben2_confidence: Optional[np.ndarray] = None
        for out in outputs:
            if out.model_name == "ben2_base" and out.confidence is not None:
                ben2_confidence = out.confidence
                break

        # Stage C — Depth edges (used for trimap widening and judge)
        depth_edges: Optional[np.ndarray] = None
        if self._depth_model is not None:
            with timer("stage_C_depth", meta["timings_ms"]):
                depth_map = self._depth_model.predict(image)
                depth_edges = compute_depth_edges(depth_map)

        # Stage C — Trimap
        with timer("stage_C_trimap", meta["timings_ms"]):
            trimap = generate_trimap(
                alpha,
                dilation=self.config.trimap_dilation,
                confidence=ben2_confidence,
                depth_edges=depth_edges,
            )
            unk = unknown_mask(trimap)

        # Stage D — Depth-aware boundary refinement
        if depth_edges is not None:
            with timer("stage_D_depth_refine", meta["timings_ms"]):
                alpha = depth_aware_refine(alpha, depth_edges, unk)
                alpha = smooth_alpha_boundary(alpha)

        # Stage F — Quality judge (first pass)
        with timer("stage_F_judge", meta["timings_ms"]):
            report = score_alpha(alpha, depth_edges=depth_edges, confidence=ben2_confidence)
            meta["quality"] = str(report)

        # If quality is low, try once more with a wider trimap band.
        if report.score < self.config.quality_threshold and self.config.max_refine_attempts > 0:
            wider_trimap = generate_trimap(
                alpha,
                dilation=self.config.trimap_dilation * 2,
                confidence=ben2_confidence,
                depth_edges=depth_edges,
            )
            unk2 = unknown_mask(wider_trimap)
            if depth_edges is not None:
                alpha = depth_aware_refine(alpha, depth_edges, unk2)
                alpha = smooth_alpha_boundary(alpha)
            report2 = score_alpha(alpha, depth_edges=depth_edges, confidence=ben2_confidence)
            if report2.score > report.score:
                report = report2
                meta["quality_retry"] = str(report2)

        # Stage E — Foreground decontamination
        with timer("stage_E_decontam", meta["timings_ms"]):
            image_np = np.array(image.convert("RGB"))
            foreground = estimate_foreground(image_np, alpha)

        # Compose final RGBA
        rgba = compose_rgba(foreground, alpha)
        meta["timings_ms"]["total"] = round((time.perf_counter() - t_total) * 1000, 1)

        return MattingResult(
            alpha=alpha,
            foreground=foreground,
            rgba=rgba,
            quality_score=report.score,
            metadata=meta,
        )

    def process_path(self, input_path: str | Path, output_path: str | Path) -> MattingResult:
        image = Image.open(input_path)
        result = self.process(image)
        result.save(output_path, premultiplied=self.config.premultiplied)
        return result

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _build_segmenters(self) -> None:
        if self._segmenters:
            return
        for sid in self.config.segmenters:
            if sid == SegmenterID.BIREFNET_HR:
                self._segmenters.append(
                    BiRefNetSegmenter(device=self._device, fp16=self._fp16)
                )
            elif sid == SegmenterID.BEN2:
                self._segmenters.append(
                    BEN2Segmenter(device=self._device, fp16=self._fp16)
                )
            elif sid == SegmenterID.INSPYRENET:
                self._segmenters.append(
                    InSPyReNetSegmenter(device=self._device)
                )
