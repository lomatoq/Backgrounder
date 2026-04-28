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
    # Weights are NC (Adobe Composition-1k) — set True only if you accept that.
    use_vitmatte: bool = False
    vitmatte_allow_nc: bool = False

    # Phase 3: SDMatte / LiteSDMatte diffusion refiner.
    # Setup (all three paths required):
    #   1. sdmatte_repo_path    — clone https://github.com/vivoCameraResearch/SDMatte
    #   2. sdmatte_model_path   — download HF model locally:
    #                             huggingface-cli download LongfeiHuang/LiteSDMatte --local-dir <path>
    #   3. sdmatte_checkpoint_path — path to LiteSDMatte.pth (usually inside sdmatte_model_path)
    # Also requires: detectron2, diffusers, accelerate, CUDA device.
    use_sdmatte: bool = False
    sdmatte_repo_path: str | None = None
    sdmatte_model_path: str | None = None        # local dir with vae/, unet/, tokenizer/ subfolders
    sdmatte_checkpoint_path: str | None = None
    sdmatte_variant: str = "lite"  # "lite" or "sdmatte"
    sdmatte_prompt_mode: str = "bbox"  # bbox, mask, trimap, point
    sdmatte_input_size: int = 1024
    sdmatte_quality_trigger: float = 0.72

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
