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
    SAM3Segmenter,
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
    sam3_refine,
    transparency_refine,
    despill_solid_background,
    remove_solid_background_spill,
    analyze_image_route,
    scrub_transparent_rgb,
    uncertainty_gated_sharpen,
    visual_alpha_fixes,
)
from backgrounder.analyze import compute_failure_map
from backgrounder.decontam import adaptive_trimap, unmix_foreground
from backgrounder.route import LAMBDA_BY_MODE, plan_routes
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
        self._sam3: Optional[SAM3Segmenter] = None
        self._sam3_load_error: Optional[str] = None
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

        if self.config.use_sam3:
            if self._device != "cuda":
                self._sam3_load_error = f"SAM 3.1 requires CUDA, current device is '{self._device}'"
            else:
                try:
                    self._sam3 = SAM3Segmenter(
                        device=self._device,
                        fp16=self._fp16,
                        model_version=self.config.sam3_model_version,
                        checkpoint_path=self.config.sam3_checkpoint_path,
                        confidence_threshold=self.config.sam3_confidence_threshold,
                    )
                    self._sam3.load()
                except Exception as exc:
                    import warnings
                    self._sam3_load_error = str(exc)
                    warnings.warn(f"SAM 3.1 disabled: {exc}")

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

    def process(self, image: Image.Image, progress=None) -> MattingResult:
        """
        Run the full pipeline on a single PIL image.

        progress : optional callable(fraction: float, desc: str) for UI feedback
                   (e.g. a Gradio gr.Progress). Called at each major stage.
        """
        _report = _ProgressReporter(progress)
        _report(0.02, "Loading models")
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
            transparent = orig_alpha < 0.99
            if transparent.sum() > 0.001 * orig_alpha.size or partial.sum() > 0:
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
            result = self._process_single(image, report=_report)

        # Upscale alpha/rgba back to original resolution if we downscaled.
        if scale < 1.0:
            result = result.upscale_to(orig_size)
        _report(1.0, "Done")
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

    def _process_single(self, image: Image.Image, report=None) -> MattingResult:
        report = report or _ProgressReporter(None)
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
            report(0.08, "Classifying subject")
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
            meta["expert"] = _display_expert_name(expert)
            top_scores = sorted(
                classification.scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            meta["neural_advisor_top"] = [
                {"type": name, "score": round(float(score), 4)}
                for name, score in top_scores
            ]

            # Map classifier weights to ordered list for ensemble.
            # Use SegmenterID.value (matches dict keys), NOT model.name (varies).
            seg_weights = [
                classification.segmenter_weights.get(sid.value, 1.0)
                for sid in self._segmenter_ids
            ]

        # Stage B — Coarse ensemble
        tta_on = self.config.use_tta
        tta_tag = " +TTA" if tta_on else ""
        report(0.15, "Segmenter ensemble")
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

        route = analyze_image_route(
            image=image,
            subject_type=subject_type,
            alpha=alpha,
            uncertainty=uncertainty,
        )
        meta.update(route.metadata())
        if (
            expert == "color_key"
            and subject_type != "text_glow"
            and not (route.clean_border and route.keyable)
        ):
            expert = "depth_only"
            meta["expert"] = _display_expert_name(expert)
            meta["expert_route_override"] = "color_key_disabled_without_clean_plate"

        # Note: a previous auto-transparency heuristic was removed. It misfired
        # on white text, light-colored objects with anti-aliased edges, etc.
        # Users who actually have glass/water can pick "transparent" from the
        # subject-type override dropdown.

        ben2_confidence: Optional[np.ndarray] = None
        for out in outputs:
            if out.model_name == "ben2_base" and out.confidence is not None:
                ben2_confidence = out.confidence
                break

        # Stage B2 — Failure-map analyzer + cost-sensitive route plan (Phase 1).
        # No-reference per-pixel difficulty from orthogonal signals (§2.1); the
        # plan (§2.2) then decides which heavy experts are worth their cost. This
        # is what keeps Max Quality fast on easy images: no failure region → no
        # expensive expert dispatched, regardless of mode.
        failure_map = None
        route_plan = None
        if self.config.use_region_router:
            try:
                with timer("stage_B2_failure_map", meta["timings_ms"]):
                    model_alphas = [o.alpha for o in outputs if o.alpha is not None]
                    report(0.32, "Failure-map + routing")
                    failure_map = compute_failure_map(
                        image,
                        alpha,
                        alpha_per_model=model_alphas if len(model_alphas) >= 2 else None,
                        bg_rgb=route.bg_rgb if route.clean_border else None,
                    )
                    route_plan = plan_routes(
                        failure_map,
                        mode=self.config.route_mode,
                        lam=LAMBDA_BY_MODE.get(self.config.route_mode),
                        available=self._available_experts(),
                    )
                meta.update(failure_map.metadata())
                meta.update(route_plan.metadata())
            except Exception as exc:  # analysis must never break a working cutout
                import warnings
                warnings.warn(f"Region router disabled (analysis failed): {exc}")
                failure_map = None
                route_plan = None

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
            # Wide band: ViTMatte needs room around every wispy strand.
            trimap_dilation = max(trimap_dilation, 28)
        elif subject_type in {"anime", "flat_cartoon", "text_logo", "sticker_logo", "text_glow"}:
            trimap_dilation = min(trimap_dilation, 5)

        with timer("stage_C_trimap", meta["timings_ms"]):
            trimap = generate_trimap(
                alpha,
                dilation=trimap_dilation,
                confidence=ben2_confidence,
                depth_edges=depth_edges,
            )

        # Stage D — Expert refinement (Phase 2: routed; Phase 1: depth-only)
        report(0.45, "Refining alpha")
        print(f"[Stage D] Refining alpha (expert={_display_expert_name(expert)})...", flush=True)
        with timer("stage_D_refine", meta["timings_ms"]):
            use_closed_form = (
                self.config.use_closed_form_refine
                and subject_type not in {
                    "transparent", "transparent_object", "product_glass",
                    "anime", "flat_cartoon", "text_logo", "sticker_logo", "text_glow",
                }
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
        if self.config.use_uncertainty_sharpen and subject_type not in (
            "transparent", "transparent_object", "product_glass",
            "text_logo", "sticker_logo", "text_glow",
            "anime", "flat_cartoon",
        ):
            with timer("stage_D2_uncertainty_sharpen", meta["timings_ms"]):
                alpha = uncertainty_gated_sharpen(
                    alpha,
                    uncertainty,
                    threshold=self.config.uncertainty_sharpen_threshold,
                    strength=self.config.uncertainty_sharpen_strength,
                )

        if subject_type == "anime" and self._sdmatte is None and route.use_cartoon_snap:
            with timer("stage_D4_cartoon_alpha_snap", meta["timings_ms"]):
                alpha = _snap_cartoon_alpha(alpha)
                meta["cartoon_alpha_snap"] = True
        elif subject_type == "anime":
            meta["cartoon_alpha_snap"] = False

        if route.use_graphic_alpha_normalize:
            with timer("stage_D5_graphic_alpha_normalize", meta["timings_ms"]):
                alpha = _normalize_graphic_alpha(alpha, subject_type=subject_type)
                meta["graphic_alpha_normalize"] = True

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
            if self.config.use_uncertainty_sharpen and subject_type not in {
                "transparent", "transparent_object", "product_glass",
            }:
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
        # Only a narrow set auto-routes through SDMatte; force_sdmatte remains
        # an explicit developer override.
        sdmatte_trigger = self.config.sdmatte_quality_trigger
        if route.allow_sdmatte_auto:
            sdmatte_trigger = min(sdmatte_trigger + 0.10, 0.85)

        if self._sdmatte is not None:
            meta["sdmatte_score_before"] = round(report.score, 3)
            meta["sdmatte_trigger"] = round(sdmatte_trigger, 3)
            auto_allowed = route.allow_sdmatte_auto
            # Router gate: when active, the cost-sensitive policy must also vote
            # for a fine-matte expert (sdmatte/zim) somewhere — a clean image with
            # no soft-edge region won't pay for diffusion even in Max Quality.
            router_wants_sdmatte = route_plan is not None and bool(
                {"sdmatte", "zim"} & set(route_plan.experts_used)
            )
            if self.config.use_region_router and route_plan is not None:
                run_sdmatte = self.config.force_sdmatte or (
                    auto_allowed and report.score < sdmatte_trigger and router_wants_sdmatte
                )
                meta["sdmatte_router_vote"] = router_wants_sdmatte
            else:
                run_sdmatte = self.config.force_sdmatte or (auto_allowed and report.score < sdmatte_trigger)
            if run_sdmatte:
                reason = "forced" if self.config.force_sdmatte else f"score {report.score:.3f} < trigger {sdmatte_trigger:.2f}"
                report(0.62, "SDMatte refining")
                print(f"[Stage G0] SDMatte refining ({reason})...", flush=True)
                with timer("stage_G0_sdmatte", meta["timings_ms"]):
                    self._sdmatte.is_transparent = subject_type == "transparent"
                    alpha_sdmatte = self._sdmatte.refine(image, alpha, trimap)
                    report_sdmatte = score_alpha(
                        alpha_sdmatte, depth_edges=depth_edges, confidence=ben2_confidence
                    )
                    # Always accept when forced (user override takes priority).
                    # When score-triggered: accept if score doesn't regress by more than 0.02.
                    accept = (
                        self.config.force_sdmatte
                        or report_sdmatte.score >= report.score - 0.02
                    )
                    if accept:
                        alpha = alpha_sdmatte
                        report = report_sdmatte
                        meta["quality"] = str(report_sdmatte)
                    meta["sdmatte_used"] = True
                    meta["sdmatte_forced"] = self.config.force_sdmatte
                    meta["sdmatte_score"] = round(report_sdmatte.score, 3)
                    meta["sdmatte_accepted"] = accept
            elif not auto_allowed:
                meta["sdmatte_skipped"] = f"subject {subject_type} uses non-diffusion path"
            elif (
                self.config.use_region_router
                and route_plan is not None
                and report.score < sdmatte_trigger
                and not router_wants_sdmatte
            ):
                meta["sdmatte_skipped"] = "router_no_fine_matte_region"
            else:
                meta["sdmatte_skipped"] = f"quality {round(report.score,3)} >= trigger {round(sdmatte_trigger,3)}"

        # Heavy-refiner cascade control: SDMatte, SAM 3.1 and SAM 2.1 are all
        # expensive full-frame mask models. Running all three on every image is
        # the main Max-Quality latency sink (spec §3 → route per-need, not always).
        # We gate the SAM stages on quality/complexity instead of an always-on
        # subject list, and treat SAM 2.1 as a fallback only when SAM 3.1 did not
        # already run (the two are redundant boundary refiners).
        sam3_executed = False

        # Stage G — SAM 3.1 refinement (Phase 3, optional)
        if self.config.use_sam3:
            meta["sam3_enabled"] = True
            if self._sam3_load_error:
                meta["sam3_skipped"] = self._sam3_load_error
            elif self._sam3 is None:
                meta["sam3_skipped"] = "not_loaded"
            elif _sam3_should_skip(subject_type, route):
                meta["sam3_skipped"] = "route_uses_keying_or_text_cleanup"
            else:
                # Fire only when the matte is actually weak or the scene is
                # genuinely hard — not for every portrait/product with a good
                # base score (that was the always-on cascade).
                _hard_scene = (
                    subject_type in {"complex_multi", "busy_scene"}
                    or meta.get("route_background") == "busy"
                )
                if self.config.use_region_router and route_plan is not None:
                    # Router gate: SAM 3.1 (the costliest expert) runs only when
                    # the policy actually selected it for a disagreement/instability
                    # region, or the scene is independently hard.
                    router_wants_sam3 = "sam3" in route_plan.experts_used
                    _need_sam3 = router_wants_sam3 or _hard_scene
                    meta["sam3_router_vote"] = router_wants_sam3
                else:
                    _need_sam3 = (
                        report.score < self.config.sam3_quality_trigger or _hard_scene
                    )
                if _need_sam3:
                    sam3_executed = True
                    with timer("stage_G_sam3", meta["timings_ms"]):
                        report(0.78, "SAM 3.1 refining")
                        alpha_sam3, sam3_meta = sam3_refine(
                            image=image,
                            alpha=alpha,
                            sam3=self._sam3,
                            owlv2=self._owlv2,
                            subject_type=subject_type,
                        )
                        meta.update(sam3_meta)
                        report_sam3 = score_alpha(
                            alpha_sam3,
                            depth_edges=depth_edges,
                            confidence=ben2_confidence,
                        )
                        accept = (
                            sam3_meta.get("sam3_status") == "applied"
                            and report_sam3.score >= report.score - 0.12
                        )
                        meta["sam3_used"] = bool(accept)
                        meta["sam3_accepted"] = bool(accept)
                        meta["sam3_score"] = round(report_sam3.score, 3)
                        if accept:
                            alpha = alpha_sam3
                            report = report_sam3
                            meta["quality"] = str(report_sam3)
                            meta["quality_sam3"] = str(report_sam3)
                else:
                    meta["sam3_skipped"] = (
                        f"quality {round(report.score,3)} >= trigger "
                        f"{round(self.config.sam3_quality_trigger,3)} and scene not hard"
                    )
        else:
            meta["sam3_enabled"] = False

        # SAM 2.1 is the boundary-refiner fallback. Skip it whenever SAM 3.1
        # already ran this image — running both is redundant work for no gain.
        if self._sam2 is not None and not sam3_executed:
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
        elif self._sam2 is not None and sam3_executed:
            meta["sam2_skipped"] = "sam3_already_refined_boundary"

        # Stage H — Transparency post-processing (Phase 3)
        if subject_type in {"transparent", "transparent_object", "product_glass"}:
            with timer("stage_H_transparency", meta["timings_ms"]):
                alpha = transparency_refine(
                    alpha=alpha,
                    image=image,
                    uncertainty=uncertainty,
                    depth_edges=depth_edges,
                )

        if self.config.use_solid_background_cleanup and route.use_solid_background_cleanup:
            with timer("stage_H2_solid_bg_spill_cleanup", meta["timings_ms"]):
                alpha, cleanup_meta = remove_solid_background_spill(
                    image=image,
                    alpha=alpha,
                    subject_type=subject_type,
                )
                meta.update(cleanup_meta)
        elif self.config.use_solid_background_cleanup:
            meta["solid_bg_spill_cleanup"] = "skipped_route"

        with timer("stage_H3_visual_qa", meta["timings_ms"]):
            alpha, visual_meta = visual_alpha_fixes(
                image=image,
                alpha=alpha,
                subject_type=subject_type,
                route_meta=meta,
            )
            meta.update(visual_meta)

        # Stage E — Foreground decontamination
        report(0.92, "Foreground decontamination")
        with timer("stage_E_decontam", meta["timings_ms"]):
            image_np = np.array(image.convert("RGB"))

            # On a known clean plate, recover the true foreground F by closed-form
            # unmixing (spec §2.3 L1) rather than the chroma-only despill. Falls
            # back to despill when the unmix core is disabled.
            def _plate_recover(bg_rgb) -> np.ndarray:
                if self.config.use_decontam_unmix:
                    meta["foreground_decontam"] = "unmix_l1"
                    return unmix_foreground(image_np, alpha, bg_rgb=bg_rgb)
                return despill_solid_background(image_np, alpha, bg_rgb)

            if subject_type in {"transparent", "transparent_object", "product_glass"}:
                foreground = image_np
            elif (
                subject_type in {"portrait", "animal_fur", "plant_thin"}
                and meta.get("route_background") == "busy"
            ):
                foreground = image_np
                meta["foreground_decontam"] = "skipped_natural_busy_scene"
            elif meta.get("solid_bg_spill_cleanup") == "applied" and meta.get("solid_bg_bg_rgb"):
                foreground = _plate_recover(meta["solid_bg_bg_rgb"])
            elif subject_type == "anime":
                foreground = image_np
            elif meta.get("solid_bg_cleanup") == "applied" and meta.get("bg_rgb"):
                foreground = _plate_recover(meta["bg_rgb"])
            else:
                foreground = estimate_foreground(image_np, alpha, subject_type=subject_type)

            if (
                subject_type in {"anime", "flat_cartoon", "sticker_logo", "solid_screen_keying"}
                and meta.get("solid_bg_spill_cleanup") != "applied"
                and meta.get("route_bg_rgb")
            ):
                foreground = _plate_recover(meta["route_bg_rgb"])
                meta["busy_graphic_despill"] = True

        rgba = scrub_transparent_rgb(compose_rgba(foreground, alpha))
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
        rgba = scrub_transparent_rgb(compose_rgba(foreground, alpha))
        report = score_alpha(alpha, depth_edges=depth_edges)

        return MattingResult(
            alpha=alpha,
            foreground=foreground,
            rgba=rgba,
            quality_score=report.score,
            metadata={"tiled": True, "tile_size": self.config.tile_size},
        )

    def _available_experts(self) -> list[str]:
        """Experts the router may dispatch given which models are actually loaded."""
        avail = ["base", "unmix", "crisp"]  # CPU / closed-form, always available
        if self._sdmatte is not None:
            avail.append("sdmatte")
        if self._sam3 is not None and not self._sam3_load_error:
            avail.append("sam3")
        # "zim" is a Phase-2 expert; not wired yet, so never offered in v1.
        return avail

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


def _snap_cartoon_alpha(alpha: np.ndarray) -> np.ndarray:
    """Hard-cut illustration mattes to remove semi-transparent backdrop rims."""
    snapped = alpha.copy()
    snapped = np.where(snapped < 0.62, 0.0, snapped)
    snapped = np.where(snapped > 0.82, 1.0, snapped)
    return np.clip(snapped, 0.0, 1.0).astype(np.float32)


def _normalize_graphic_alpha(alpha: np.ndarray, subject_type: str) -> np.ndarray:
    """
    Make graphic/sticker/cartoon cutouts opaque without the old outline-eating snap.

    Dedicated segmentation models often return confident-looking cartoon bodies
    at alpha 0.45-0.85. For graphics that is visually wrong: stickers and game
    assets should be opaque inside, with only a narrow antialias band at edges.
    This contrast curve lifts the body while leaving tiny edge transparency.
    """
    if subject_type == "text_glow":
        return alpha.astype(np.float32)

    low = 0.12
    high = 0.72
    if subject_type in {"text_logo", "sticker_logo", "solid_screen_keying"}:
        low = 0.08
        high = 0.62
    elif subject_type in {"anime", "flat_cartoon"}:
        low = 0.10
        high = 0.66

    t = np.clip((alpha - low) / (high - low + 1e-6), 0.0, 1.0)
    lifted = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    lifted = np.where(lifted > 0.94, 1.0, lifted)
    lifted = np.where(lifted < 0.025, 0.0, lifted)
    return np.clip(np.maximum(alpha * 0.25, lifted), 0.0, 1.0).astype(np.float32)


def _sam3_should_skip(subject_type: str, route) -> bool:
    if subject_type in {"text_logo", "text_glow", "document_screenshot", "solid_screen_keying"}:
        return True
    return bool(route.clean_border and route.keyable)


def _display_expert_name(expert: str) -> str:
    return "crisp_edges" if expert == "depth_only" else expert


class _ProgressReporter:
    """
    Thin adapter around an optional progress callback.

    Accepts either a 2-arg callable(fraction, desc) or a Gradio-style
    ``gr.Progress`` (callable as ``progress(fraction, desc=...)``). Monotonic:
    never reports a fraction lower than one already reported, so optional stages
    that are skipped don't make the bar jump backwards. All errors are swallowed
    — progress reporting must never break a cutout.
    """

    def __init__(self, cb) -> None:
        self._cb = cb
        self._last = 0.0

    def __call__(self, fraction: float, desc: str = "") -> None:
        if self._cb is None:
            return
        frac = max(self._last, min(float(fraction), 1.0))
        self._last = frac
        try:
            self._cb(frac, desc=desc)
        except TypeError:
            try:
                self._cb(frac, desc)
            except Exception:
                pass
        except Exception:
            pass
