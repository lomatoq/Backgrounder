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

    # Depth Anything V2-Small (Apache-2.0) for low-contrast refinement.
    use_depth: bool = True
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
    use_closed_form_refine: bool = True
    closed_form_max_pixels: int = 65_536
    use_uncertainty_sharpen: bool = True
    uncertainty_sharpen_threshold: float = 0.15
    uncertainty_sharpen_strength: float = 0.60

    # Export premultiplied alpha (avoids fringing on composition).
    premultiplied: bool = False

    # Phase 2: subject classifier drives segmenter weights + expert routing.
    use_classifier: bool = True
    # Override auto-detected subject type (one of classifier.SUBJECT_TYPES or None).
    subject_type_override: str | None = None

    # Phase 2: ViTMatte trimap-based refiner for hair/fur.
    # Weights trained on Adobe Composition-1k — NON-COMMERCIAL use only.
    # Set vitmatte_allow_nc=True to confirm you accept that restriction.
    use_vitmatte: bool = True
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
    sdmatte_input_size: int = 768
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

    # Phase 3: OWLv2 open-vocabulary localizer (Apache-2.0).
    # Supplies bounding-box prompts to SAM 2.1 for multi-object scenes.
    use_owlv2: bool = False
    owlv2_model_id: str = "google/owlv2-base-patch16-ensemble"
