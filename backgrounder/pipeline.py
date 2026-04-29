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
            if self._device != "cuda":
                import warnings
                warnings.warn(
                    f"SDMatte requires CUDA but device is '{self._device}' — skipping. "
                    "Install torch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124"
                )
            else:
                self._sdmatte = SDMatteRefiner(
                    cache_dir=self.config.sdmatte_cache_dir,
                    device=self._device,
                    variant=self.config.sdmatte_variant,
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

    # Maximum long-edge resolution before we auto-downsample.
    # Keeps segmenter preprocessing fast and VRAM predictable.
    _MAX_SIDE = 2048

    def process(self, image: Image.Image) -> MattingResult:
        """Run the full pipeline on a single PIL image."""
        if not self._ready:
            self.load()

        # RGBA passthrough: if the image already has meaningful transparency,
        # the user supplied an already-extracted asset (logo, UI element,
        # screenshot from a transparent PNG). Re-segmenting it would composite
        # the alpha against black and confuse BiRefNet/BEN2 (semi-transparent
        # glow pixels become dark, get classified as background, and the
        # output alpha collapses). Preserve the existing alpha unchanged.
        if image.mode == "RGBA":
            rgba_arr = np.array(image)
            orig_alpha = rgba_arr[..., 3].astype(np.float32) / 255.0
            partial = (orig_alpha > 0.05) & (orig_alpha < 0.95)
            if partial.sum() > 0.02 * orig_alpha.size:
                print("[Backgrounder] RGBA input with existing transparency — passthrough", flush=True)
                return MattingResult(
                    alpha=orig_alpha,
                    foreground=rgba_arr[..., :3],
                    rgba=image,
                    quality_score=1.0,
                    metadata={
                        "passthrough": "rgba_input",
                        "device": self._device,
                        "timings_ms": {"total": 0.0},
                    },
                )

        orig_size = image.size  # (W, H)
        image, scale = self._maybe_downscale(image)

        # 4K tiling: delegate tile processing to a simpler single-model call.
        W, H = image.size
        if self.config.tile_size > 0 and (W > self.config.tile_size or H > self.config.tile_size):
            result = self._process_tiled(image)
        else:
            result = self._process_single(image)

        # Upscale alpha/rgba back to original resolution if we downscaled.
        if scale < 1.0:
            result = result.upscale_to(orig_size)
        return result

    def _maybe_downscale(self, image: Image.Image) -> tuple[Image.Image, float]:
        W, H = image.size
        max_side = max(W, H)
        if max_side <= self._MAX_SIDE:
            return image, 1.0
        scale = self._MAX_SIDE / max_side
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        print(f"[Backgrounder] Downscaling {W}×{H} → {new_w}×{new_h} (long edge capped at {self._MAX_SIDE}px)", flush=True)
        return image.resize((new_w, new_h), Image.LANCZOS), scale

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
        W, H = image.size
        print(f"[Backgrounder] Processing {W}×{H} image on {self._device}", flush=True)

        # Stage A — Subject classification (Phase 2)
        classification = None
        seg_weights: Optional[List[float]] = None
        expert = "depth_only"
        subject_type = "generic"

        if self._classifier is not None:
            print("[Stage A] Classifying subject type...", flush=True)
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
        tta_on = self.config.use_tta
        tta_tag = " +TTA" if tta_on else ""
        print(f"[Stage B] Running segmenter ensemble ({len(self._segmenters)} models{tta_tag})...", flush=True)
        with timer("stage_B_ensemble", meta["timings_ms"]):
            alpha, uncertainty, outputs = ensemble_predict(
                self._segmenters,
                image,
                weights=seg_weights,
                parallel=True,
                tta=tta_on,
            )
        if tta_on:
            meta["tta"] = True

        # Note: a previous auto-transparency heuristic was removed. It misfired
        # on white text, light-colored objects with anti-aliased edges, etc.
        # Users who actually have glass/water can pick "transparent" from the
        # subject-type override dropdown.

        ben2_confidence: Optional[np.ndarray] = None
        for out in outputs:
            if out.model_name == "ben2_base" and out.confidence is not None:
                ben2_confidence = out.confidence
                break

        # Stage C — Depth
        depth_edges: Optional[np.ndarray] = None
        if self._depth_model is not None:
            print("[Stage C] Estimating depth...", flush=True)
            with timer("stage_C_depth", meta["timings_ms"]):
                depth_map = self._depth_model.predict(image)
                depth_edges = compute_depth_edges(depth_map)

        # Stage C — Trimap
        # Hair/fur/plants need a wider unknown band to capture fine strands.
        # Text/logos need a tighter band — wide dilation pulls in solid background.
        _HAIR_TYPES = {"portrait", "animal_fur", "plant_thin"}
        trimap_dilation = self.config.trimap_dilation
        if subject_type in _HAIR_TYPES:
            trimap_dilation = max(trimap_dilation, 20)
        elif subject_type == "text_logo":
            trimap_dilation = min(trimap_dilation, 5)

        with timer("stage_C_trimap", meta["timings_ms"]):
            trimap = generate_trimap(
                alpha,
                dilation=trimap_dilation,
                confidence=ben2_confidence,
                depth_edges=depth_edges,
            )

        # Stage D — Expert refinement (Phase 2: routed; Phase 1: depth-only)
        print(f"[Stage D] Refining alpha (expert={expert})...", flush=True)
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

        # text_logo already has binary-snapped crisp edges from color_key;
        # uncertainty sharpening would distort those clean edges.
        if self.config.use_uncertainty_sharpen and subject_type not in ("transparent", "text_logo"):
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
                dilation=trimap_dilation * 2,
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
        # Hard subjects get a raised trigger so SDMatte runs unless quality is already excellent.
        # force_sdmatte bypasses the quality gate entirely (user opt-in from UI).
        _HARD_SUBJECTS = {"portrait", "animal_fur", "complex_multi", "plant_thin"}
        sdmatte_trigger = self.config.sdmatte_quality_trigger
        if subject_type in _HARD_SUBJECTS:
            sdmatte_trigger = min(sdmatte_trigger + 0.10, 0.85)

        if self._sdmatte is not None:
            meta["sdmatte_score_before"] = round(report.score, 3)
            meta["sdmatte_trigger"] = round(sdmatte_trigger, 3)
            run_sdmatte = self.config.force_sdmatte or report.score < sdmatte_trigger
            if run_sdmatte:
                reason = "forced" if self.config.force_sdmatte else f"score {report.score:.3f} < trigger {sdmatte_trigger:.2f}"
                print(f"[Stage G0] SDMatte refining ({reason})...", flush=True)
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
                    meta["sdmatte_forced"] = self.config.force_sdmatte
                    meta["sdmatte_score"] = round(report_sdmatte.score, 3)
                    meta["sdmatte_accepted"] = report_sdmatte.score >= report.score
            else:
                meta["sdmatte_skipped"] = f"quality {round(report.score,3)} >= trigger {round(sdmatte_trigger,3)}"

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
        total_ms = round((time.perf_counter() - t_total) * 1000, 1)
        meta["timings_ms"]["total"] = total_ms
        print(f"[Backgrounder] Done in {total_ms/1000:.1f}s  quality={report.score:.3f}  subject={subject_type}", flush=True)

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


