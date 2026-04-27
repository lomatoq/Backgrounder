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

    # Export premultiplied alpha (avoids fringing on composition).
    premultiplied: bool = False
