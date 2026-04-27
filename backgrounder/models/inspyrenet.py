from __future__ import annotations
from typing import Optional

import numpy as np
from PIL import Image

from backgrounder.result import SegmentationOutput
from .base import BaseSegmenter


class InSPyReNetSegmenter(BaseSegmenter):
    """
    InSPyReNet via the `transparent-background` pip package (MIT).

    Install extra: pip install transparent-background
    Res2Net backbone — more ANE-friendly on Apple Silicon than Swin-L.
    Strong baseline for high-res inputs thanks to laplacian pyramid blending.
    """

    def __init__(self, device: str = "cpu", jit: bool = False) -> None:
        self._device = device
        self._jit = jit
        self._remover = None

    @property
    def name(self) -> str:
        return "inspyrenet"

    def _load(self) -> None:
        try:
            from transparent_background import Remover
        except ImportError as exc:
            raise ImportError(
                "transparent-background is not installed.  "
                "Run: pip install transparent-background"
            ) from exc

        # 'fast' uses the lighter InSPyReNet-Res2Net50 variant
        self._remover = Remover(mode="fast", jit=self._jit, device=self._device)

    def _predict(self, image: Image.Image) -> SegmentationOutput:
        rgb = image.convert("RGB")
        # transparent_background returns the RGBA composite directly
        rgba = self._remover.process(rgb, type="rgba")
        alpha = np.array(rgba)[:, :, 3].astype(np.float32) / 255.0
        return SegmentationOutput(alpha=alpha, model_name=self.name)
