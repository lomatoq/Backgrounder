from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Device(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class SegmenterID(str, Enum):
    BIREFNET_HR = "birefnet_hr"
    BEN2 = "ben2"
    INSPYRENET = "inspyrenet"


@dataclass
class PipelineConfig:
    # Which segmenters to run in parallel (Stage B).
    # All are MIT/Apache-2.0 — safe for commercial use.
    segmenters: List[SegmenterID] = field(
        default_factory=lambda: [SegmenterID.BIREFNET_HR, SegmenterID.BEN2]
    )

    # Depth Anything V2-Small (Apache-2.0), opt-in only.
    # In practice it can over-expand trimaps and make normal cutouts worse.
    use_depth: bool = False
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"

    device: Device = Device.AUTO

    # fp16 is only applied on CUDA; MPS/CPU use fp32 to avoid numerics issues.
    fp16: bool = True

    # Trimap unknown-band half-width in pixels.
    trimap_dilation: int = 10

    # Quality judge: re-run refinement if score < threshold (0 = never, 1 = always).
    quality_threshold: float = 0.55
    max_refine_attempts: int = 2

    # Phase 4A: classical matting refinement, no extra model weights.
    use_closed_form_refine: bool = False
    closed_form_max_pixels: int = 65_536
    use_uncertainty_sharpen: bool = True
    uncertainty_sharpen_threshold: float = 0.15
    uncertainty_sharpen_strength: float = 0.60
    use_solid_background_cleanup: bool = True

    # Phase 0 (spec §2.3 Level 1): closed-form foreground unmixing.
    # When a clean-plate background colour is known (flat/CG keyable route), emit
    # the recovered foreground F instead of the observed composite I, killing the
    # solid-colour rim. No model weights, no training.
    use_decontam_unmix: bool = True

    # Phase 1 (spec §2.1-2.5): failure-map analyzer + cost-sensitive per-region
    # router. When on, the failure map drives the adaptive trimap band and the
    # router decides which heavy experts (SDMatte/SAM3) are worth running — so a
    # clean image stays fast even in Max Quality, while hard images get the full
    # budget. route_mode picks λ: "fast" (large λ) / "smart" / "max" (λ→0).
    use_region_router: bool = True
    route_mode: str = "smart"

    # Lazy-load heavy experts (ViTMatte/SDMatte/SAM2/SAM3/OWLv2) on first actual
    # use and unload them afterwards, instead of holding all of them in VRAM for
    # the whole session. Essential on 12 GB GPUs where Max Quality otherwise
    # OOM-thrashes (SDMatte alone is ~5 GB). Set False to keep models resident
    # (faster for batch runs on large GPUs).
    lazy_load_experts: bool = True

    # Stage 1b: after SAM 3.1 establishes the silhouette for a hair/fur subject,
    # run the matting expert on the silhouette's edge band to recover soft strands
    # (SAM's hard mask alone has no soft hair). Interior stays solid.
    use_seg_edge_matte: bool = True

    # Export premultiplied alpha (avoids fringing on composition).
    premultiplied: bool = False

    # Phase 2: subject classifier drives segmenter weights + expert routing.
    use_classifier: bool = True
    # Override auto-detected subject type (one of classifier.SUBJECT_TYPES or None).
    subject_type_override: str | None = None

    # Phase 2: ViTMatte trimap-based refiner for hair/fur.
    # Weights trained on Adobe Composition-1k — NON-COMMERCIAL use only.
    # Set vitmatte_allow_nc=True to confirm you accept that restriction.
    use_vitmatte: bool = False
    vitmatte_allow_nc: bool = False   # user must opt-in to NC weights

    # Phase 3: SDMatte diffusion refiner (MIT, ICCV 2025).
    # Bundled — no external repo needed. Auto-downloads weights (~5 GB) and
    # SD 2.1 architecture configs to sdmatte_cache_dir on first use.
    # Requires: CUDA, diffusers, accelerate, safetensors.
    use_sdmatte: bool = False
    # When True, run SDMatte regardless of quality score (user "force" mode).
    force_sdmatte: bool = False
    sdmatte_cache_dir: str = "~/.cache/backgrounder/sdmatte"
    sdmatte_variant: str = "sdmatte"  # "sdmatte" or "sdmatte_plus"
    sdmatte_prompt_mode: str = "trimap"  # bbox, mask, trimap, point
    sdmatte_input_size: int = 1024
    sdmatte_quality_trigger: float = 0.72

    # Test-Time Augmentation: run ensemble on original + horizontal flip, average.
    # Improves quality on low-contrast / dark-on-dark subjects (~2× Stage-B time).
    use_tta: bool = False

    # Phase 2: tile large images for 4K+ support.
    # 0 = disabled; >0 = tile_size in pixels.
    tile_size: int = 0
    tile_overlap: int = 128

    # Phase 3: SAM 2.1 boundary refinement (Apache-2.0).
    # Activated when quality score < sam2_quality_trigger OR subject == complex_multi.
    use_sam2: bool = False
    sam2_model_id: str = "facebook/sam2.1-hiera-base-plus"
    sam2_quality_trigger: float = 0.60

    # Phase 4: SAM 3.1 concept/box mask refinement (Meta SAM 3 repo).
    # Optional because it requires the external facebookresearch/sam3 package,
    # CUDA, and access to the facebook/sam3.1 checkpoint on Hugging Face.
    use_sam3: bool = False
    sam3_model_version: str = "sam3.1"
    sam3_checkpoint_path: str | None = None
    # Fire SAM 3.1 only when the matte is genuinely weak (or the scene is hard).
    # The old 0.88 meant "almost always run", which—stacked on SDMatte+SAM2—was
    # the Max-Quality 10-minute latency sink.
    sam3_quality_trigger: float = 0.75
    sam3_confidence_threshold: float = 0.30

    # Phase 3: OWLv2 open-vocabulary localizer (Apache-2.0).
    # Supplies bounding-box prompts to SAM 2.1 for multi-object scenes.
    use_owlv2: bool = False
    owlv2_model_id: str = "google/owlv2-base-patch16-ensemble"
