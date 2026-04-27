from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from backgrounder.config import PipelineConfig, SegmenterID, Device
from backgrounder.models import (
    BiRefNetSegmenter,
    BEN2Segmenter,
    InSPyReNetSegmenter,
    DepthAnythingV2Small,
    ViTMatteRefiner,
    SDMatteRefiner,
    SAM2Segmenter,
    OWLv2Localizer,
)
from backgrounder.models.base import BaseSegmenter
from backgrounder.result import MattingResult
from backgrounder.stages import (
    ensemble_predict,
    generate_trimap,
    expert_refine,
    score_alpha,
    tile_process,
    sam2_refine,
    transparency_refine,
    uncertainty_gated_sharpen,
)
from backgrounder.utils import (
    resolve_device,
    compute_depth_edges,
    estimate_foreground,
    compose_rgba,
    timer,
)


class BackgroundRemovalPipeline:
    """
    Modular background removal pipeline.

    Phase 1 stages
    ──────────────
    B. Coarse: parallel ensemble (BiRefNet_HR + BEN2_Base [+ InSPyReNet])
    C. Trimap: BEN2 confidence + disagreement + depth edges → unknown band
    D. Refine: closed-form matting + uncertainty-gated sharpening
    E. Decontam: foreground colour-spill removal
    F. Judge: quality score; retry with wider trimap if below threshold

    Phase 2 additions
    ─────────────────
    A. Classify: CLIP subject classifier → adjust segmenter weights + pick expert
    D. Expert routing: portrait/fur → ViTMatte; product/glass → depth-only
    4K: SAHI-style tiling for images larger than config.tile_size

    Phase 3 additions
    ─────────────────
    G. SAM 2.1 refinement: precise mask via point/box prompts when quality is low
       or scene is complex_multi.  Box prompts supplied by OWLv2 if enabled.
    H. Transparency: glass/smoke post-processing — soften alpha in high-uncertainty
       + depth-boundary regions; apply guided-filter edge-aware smoothing.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._device = resolve_device(self.config.device.value)
        self._fp16 = self.config.fp16

        self._segmenters: List[BaseSegmenter] = []
        self._segmenter_ids: List[SegmenterID] = []  # parallel to _segmenters
        self._depth_model: Optional[DepthAnythingV2Small] = None
        self._vitmatte: Optional[ViTMatteRefiner] = None
        self._sdmatte: Optional[SDMatteRefiner] = None
        self._classifier = None
        self._sam2: Optional[SAM2Segmenter] = None
        self._owlv2: Optional[OWLv2Localizer] = None
        self._ready = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> "BackgroundRemovalPipeline":
        """Eagerly load all models. Call once at startup."""
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

        if self.config.use_vitmatte and self.config.vitmatte_allow_nc:
            self._vitmatte = ViTMatteRefiner(
                device=self._device,
                fp16=self._fp16,
                allow_nc_weights=True,
            )
            self._vitmatte.load()

        if self.config.use_sdmatte:
            if not self.config.sdmatte_repo_path or not self.config.sdmatte_checkpoint_path:
                raise ValueError(
                    "use_sdmatte=True requires sdmatte_repo_path and sdmatte_checkpoint_path."
                )
            self._sdmatte = SDMatteRefiner(
                repo_path=self.config.sdmatte_repo_path,
                checkpoint_path=self.config.sdmatte_checkpoint_path,
                variant=self.config.sdmatte_variant,
                device=self._device,
                pretrained_model_name_or_path=self.config.sdmatte_pretrained_model_name_or_path,
                prompt_mode=self.config.sdmatte_prompt_mode,
                input_size=self.config.sdmatte_input_size,
            )
            self._sdmatte.load()

        if self.config.use_classifier:
            from backgrounder.classifier import SubjectClassifier
            self._classifier = SubjectClassifier(device=self._device)
            self._classifier.load()

        if self.config.use_sam2:
            self._sam2 = SAM2Segmenter(
                device=self._device,
                fp16=self._fp16,
                model_id=self.config.sam2_model_id,
            )
            self._sam2.load()

        if self.config.use_owlv2:
            self._owlv2 = OWLv2Localizer(
                device=self._device,
                model_id=self.config.owlv2_model_id,
            )
            self._owlv2.load()

        self._ready = True
        return self

    def process(self, image: Image.Image) -> MattingResult:
        """Run the full pipeline on a single PIL image."""
        if not self._ready:
            self.load()

        # 4K tiling: delegate tile processing to a simpler single-model call.
        W, H = image.size
        if self.config.tile_size > 0 and (W > self.config.tile_size or H > self.config.tile_size):
            return self._process_tiled(image)

        return self._process_single(image)

    def process_path(self, input_path: str | Path, output_path: str | Path) -> MattingResult:
        image = Image.open(input_path)
        result = self.process(image)
        result.save(output_path, premultiplied=self.config.premultiplied)
        return result

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _process_single(self, image: Image.Image) -> MattingResult:
        meta: dict = {"device": self._device, "timings_ms": {}}
        t_total = time.perf_counter()

        # Stage A — Subject classification (Phase 2)
        classification = None
        seg_weights: Optional[List[float]] = None
        expert = "depth_only"
        subject_type = "generic"

        if self._classifier is not None:
            with timer("stage_A_classify", meta["timings_ms"]):
                if self.config.subject_type_override:
                    from backgrounder.classifier import (
                        EXPERT_MAP, SEGMENTER_WEIGHTS, ClassificationResult, SUBJECT_TYPES
                    )
                    st = self.config.subject_type_override
                    classification = ClassificationResult(
                        subject_type=st,
                        scores={t: 0.0 for t in SUBJECT_TYPES},
                        expert=EXPERT_MAP.get(st, "depth_only"),
                        segmenter_weights=SEGMENTER_WEIGHTS.get(st, SEGMENTER_WEIGHTS["generic"]),
                    )
                else:
                    classification = self._classifier.classify(image)

            expert = classification.expert
            subject_type = classification.subject_type
            meta["subject_type"] = subject_type
            meta["expert"] = expert

            # Map classifier weights to ordered list for ensemble.
            # Use SegmenterID.value (matches dict keys), NOT model.name (varies).
            seg_weights = [
                classification.segmenter_weights.get(sid.value, 1.0)
                for sid in self._segmenter_ids
            ]

        # Stage B — Coarse ensemble
        with timer("stage_B_ensemble", meta["timings_ms"]):
            alpha, uncertainty, outputs = ensemble_predict(
                self._segmenters,
                image,
                weights=seg_weights,
                parallel=True,
            )

        if subject_type != "transparent" and _looks_transparent(image, alpha):
            subject_type = "transparent"
            expert = "depth_only"
            meta["subject_type"] = subject_type
            meta["expert"] = expert
            meta["transparent_auto"] = True

        ben2_confidence: Optional[np.ndarray] = None
        for out in outputs:
            if out.model_name == "ben2_base" and out.confidence is not None:
                ben2_confidence = out.confidence
                break

        # Stage C — Depth
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

        # Stage D — Expert refinement (Phase 2: routed; Phase 1: depth-only)
        with timer("stage_D_refine", meta["timings_ms"]):
            use_closed_form = (
                self.config.use_closed_form_refine
                and subject_type != "transparent"
            )
            alpha = expert_refine(
                image=image,
                alpha=alpha,
                trimap=trimap,
                expert=expert,
                depth_edges=depth_edges,
                vitmatte=self._vitmatte,
                use_closed_form=use_closed_form,
                closed_form_max_pixels=self.config.closed_form_max_pixels,
            )

        if self.config.use_uncertainty_sharpen and subject_type != "transparent":
            with timer("stage_D2_uncertainty_sharpen", meta["timings_ms"]):
                alpha = uncertainty_gated_sharpen(
                    alpha,
                    uncertainty,
                    threshold=self.config.uncertainty_sharpen_threshold,
                    strength=self.config.uncertainty_sharpen_strength,
                )

        # Stage F — Quality judge + wider-trimap retry
        with timer("stage_F_judge", meta["timings_ms"]):
            report = score_alpha(alpha, depth_edges=depth_edges, confidence=ben2_confidence)
            meta["quality"] = str(report)

        if report.score < self.config.quality_threshold and self.config.max_refine_attempts > 0:
            wider_trimap = generate_trimap(
                alpha,
                dilation=self.config.trimap_dilation * 2,
                confidence=ben2_confidence,
                depth_edges=depth_edges,
            )
            alpha2 = expert_refine(
                image=image,
                alpha=alpha,
                trimap=wider_trimap,
                expert=expert,
                depth_edges=depth_edges,
                vitmatte=self._vitmatte,
                use_closed_form=use_closed_form,
                closed_form_max_pixels=self.config.closed_form_max_pixels,
            )
            if self.config.use_uncertainty_sharpen and subject_type != "transparent":
                alpha2 = uncertainty_gated_sharpen(
                    alpha2,
                    uncertainty,
                    threshold=self.config.uncertainty_sharpen_threshold,
                    strength=self.config.uncertainty_sharpen_strength,
                )
            report2 = score_alpha(alpha2, depth_edges=depth_edges, confidence=ben2_confidence)
            if report2.score > report.score:
                alpha, report = alpha2, report2
                meta["quality"] = str(report2)
                meta["quality_retry"] = str(report2)

        # Stage G0 — SDMatte diffusion refinement (optional, heavy CUDA path)
        if self._sdmatte is not None and report.score < self.config.sdmatte_quality_trigger:
            with timer("stage_G0_sdmatte", meta["timings_ms"]):
                self._sdmatte.is_transparent = subject_type == "transparent"
                alpha_sdmatte = self._sdmatte.refine(image, alpha, trimap)
                report_sdmatte = score_alpha(
                    alpha_sdmatte, depth_edges=depth_edges, confidence=ben2_confidence
                )
                if report_sdmatte.score >= report.score:
                    alpha = alpha_sdmatte
                    report = report_sdmatte
                    meta["quality"] = str(report_sdmatte)
                    meta["sdmatte_used"] = True
                    meta["quality_sdmatte"] = str(report_sdmatte)

        # Stage G — SAM 2.1 refinement (Phase 3, optional)
        if self._sam2 is not None:
            _need_sam2 = (
                report.score < self.config.sam2_quality_trigger
                or subject_type == "complex_multi"
            )
            if _need_sam2:
                with timer("stage_G_sam2", meta["timings_ms"]):
                    alpha_sam2 = sam2_refine(
                        image=image,
                        alpha=alpha,
                        sam2=self._sam2,
                        owlv2=self._owlv2,
                        subject_type=subject_type,
                    )
                    report_sam2 = score_alpha(
                        alpha_sam2, depth_edges=depth_edges, confidence=ben2_confidence
                    )
                    if report_sam2.score >= report.score:
                        alpha = alpha_sam2
                        report = report_sam2
                        meta["quality"] = str(report_sam2)
                        meta["sam2_used"] = True
                        meta["quality_sam2"] = str(report_sam2)

        # Stage H — Transparency post-processing (Phase 3)
        if subject_type == "transparent":
            with timer("stage_H_transparency", meta["timings_ms"]):
                alpha = transparency_refine(
                    alpha=alpha,
                    image=image,
                    uncertainty=uncertainty,
                    depth_edges=depth_edges,
                )

        # Stage E — Foreground decontamination
        with timer("stage_E_decontam", meta["timings_ms"]):
            image_np = np.array(image.convert("RGB"))
            if subject_type == "transparent":
                foreground = image_np
            else:
                foreground = estimate_foreground(image_np, alpha)

        rgba = compose_rgba(foreground, alpha)
        meta["timings_ms"]["total"] = round((time.perf_counter() - t_total) * 1000, 1)

        return MattingResult(
            alpha=alpha,
            foreground=foreground,
            rgba=rgba,
            quality_score=report.score,
            metadata=meta,
        )

    def _process_tiled(self, image: Image.Image) -> MattingResult:
        """Process 4K+ images by tiling over the primary segmenter."""
        # Use only the first (strongest) segmenter per tile to keep memory stable.
        primary = self._segmenters[0]

        def _tile_fn(tile: Image.Image) -> np.ndarray:
            return primary.predict(tile).alpha

        alpha = tile_process(
            image,
            _tile_fn,
            tile_size=self.config.tile_size,
            overlap=self.config.tile_overlap,
        )

        # Run depth + refinement + decontam on the full merged alpha.
        depth_edges: Optional[np.ndarray] = None
        if self._depth_model is not None:
            depth_map = self._depth_model.predict(image)
            depth_edges = compute_depth_edges(depth_map)

        from backgrounder.stages import generate_trimap
        trimap = generate_trimap(alpha, dilation=self.config.trimap_dilation, depth_edges=depth_edges)
        alpha = expert_refine(
            image,
            alpha,
            trimap,
            "depth_only",
            depth_edges,
            vitmatte=None,
            use_closed_form=self.config.use_closed_form_refine,
            closed_form_max_pixels=self.config.closed_form_max_pixels,
        )

        image_np = np.array(image.convert("RGB"))
        foreground = estimate_foreground(image_np, alpha)
        rgba = compose_rgba(foreground, alpha)
        report = score_alpha(alpha, depth_edges=depth_edges)

        return MattingResult(
            alpha=alpha,
            foreground=foreground,
            rgba=rgba,
            quality_score=report.score,
            metadata={"tiled": True, "tile_size": self.config.tile_size},
        )

    def _build_segmenters(self) -> None:
        if self._segmenters:
            return
        for sid in self.config.segmenters:
            if sid == SegmenterID.BIREFNET_HR:
                self._segmenters.append(BiRefNetSegmenter(device=self._device, fp16=self._fp16))
            elif sid == SegmenterID.BEN2:
                self._segmenters.append(BEN2Segmenter(device=self._device, fp16=self._fp16))
            elif sid == SegmenterID.INSPYRENET:
                self._segmenters.append(InSPyReNetSegmenter(device=self._device))
            self._segmenter_ids.append(sid)


def _looks_transparent(image: Image.Image, alpha: np.ndarray) -> bool:
    """Cheap post-ensemble heuristic for tinted glass/plastic objects."""
    fg = alpha > 0.05
    if fg.sum() < max(64, alpha.size * 0.02):
        return False

    semi = (alpha > 0.18) & (alpha < 0.92) & fg
    semi_ratio = float(semi.sum() / (fg.sum() + 1e-8))
    if semi_ratio < 0.18:
        return False

    rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = (maxc - minc) / (maxc + 1e-6)
    sat_mean = float(saturation[fg].mean())
    return sat_mean > 0.22
