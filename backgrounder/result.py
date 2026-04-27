from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


@dataclass
class MattingResult:
    """Final output of the pipeline."""

    # float32 [0, 1], H×W
    alpha: np.ndarray

    # uint8 RGB, H×W×3 — colour-decontaminated foreground
    foreground: np.ndarray

    # Composed RGBA image ready to save
    rgba: Image.Image

    # Quality judge score in [0, 1]; higher is better
    quality_score: float

    # Diagnostic metadata (model names, stage timings, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path, *, premultiplied: bool = False) -> None:
        """Save to PNG.  premultiplied=True uses Pillow 'RGBa' mode."""
        path = Path(path)
        if premultiplied:
            a = (self.alpha * 255).astype(np.uint8)
            rgb = (self.foreground.astype(np.float32) * self.alpha[:, :, None]).astype(np.uint8)
            img = Image.fromarray(
                np.dstack([rgb, a]), mode="RGBa"
            ).convert("RGBA")
        else:
            img = self.rgba
        img.save(path)

    @property
    def size(self) -> tuple[int, int]:
        return self.rgba.size  # (W, H)


@dataclass
class SegmentationOutput:
    """Intermediate output from a single segmentation model."""

    # float32 [0, 1], H×W
    alpha: np.ndarray

    # float32 [0, 1], H×W; 1 = model is certain, 0 = uncertain.
    # Optional — not all models provide this.
    confidence: Optional[np.ndarray] = None

    model_name: str = ""

    def __post_init__(self) -> None:
        self.alpha = np.clip(self.alpha, 0.0, 1.0).astype(np.float32)
        if self.confidence is not None:
            self.confidence = np.clip(self.confidence, 0.0, 1.0).astype(np.float32)
