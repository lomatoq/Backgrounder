from __future__ import annotations
from typing import Optional

import numpy as np
from PIL import Image

from backgrounder.result import SegmentationOutput
from .base import BaseSegmenter

# MIT licensed.  BEN2_Base uses a MVANet-decoder with Confidence-Guided Matting.
# The free (non-API) path returns an RGBA image; alpha channel is the CGM mask.
_MODEL_ID = "PramaLLC/BEN2"


class BEN2Segmenter(BaseSegmenter):
    """
    BEN2_Base — MIT, commercial OK.

    Key advantage: the alpha channel is the Confidence-Guided Matting mask,
    which directly encodes model uncertainty — pixels near 0.5 are uncertain,
    pixels near 0 or 1 are definite.  We expose this as a confidence map.
    """

    def __init__(self, device: str = "cpu", fp16: bool = False) -> None:
        self._device = device
        self._fp16 = fp16
        self._model = None

    @property
    def name(self) -> str:
        return "ben2_base"

    def _load(self) -> None:
        import torch
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(_MODEL_ID, trust_remote_code=True)
        if self._device != "cpu":
            self._model = self._model.to(self._device)
        if self._fp16 and self._device.startswith("cuda"):
            self._model = self._model.half()
        self._model.eval()

    def _predict(self, image: Image.Image) -> SegmentationOutput:
        rgb = image.convert("RGB")
        orig_wh = rgb.size

        # BEN2 returns an RGBA PIL image; alpha is the CGM segmentation mask.
        result = self._model.inference(image=rgb, refine_foreground=False)

        if isinstance(result, Image.Image):
            rgba = result.convert("RGBA") if result.mode != "RGBA" else result
        elif isinstance(result, (list, tuple)):
            # Some versions return (fg_image, mask_image)
            candidate = result[1] if len(result) > 1 else result[0]
            rgba = candidate.convert("RGBA") if isinstance(candidate, Image.Image) else result[0].convert("RGBA")
        else:
            raise RuntimeError(f"Unexpected BEN2 output type: {type(result)}")

        # Resize if model changed the resolution
        if rgba.size != orig_wh:
            rgba = rgba.resize(orig_wh, Image.LANCZOS)

        alpha = np.array(rgba)[:, :, 3].astype(np.float32) / 255.0

        # Confidence: distance from 0.5 → 1 at certain, 0 at maximally uncertain.
        confidence = (2.0 * np.abs(alpha - 0.5)).astype(np.float32)

        return SegmentationOutput(alpha=alpha, confidence=confidence, model_name=self.name)

    def unload(self) -> None:
        import torch
        del self._model
        self._model = None
        self._loaded = False
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
