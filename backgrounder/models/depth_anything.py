from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from PIL import Image

from backgrounder.utils import choose_dtype

# Apache-2.0 licensed — commercial OK.
# Only the Small variant; Base/Large/Giant are CC-BY-NC.
_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2Small:
    """
    Depth Anything V2 — Small (25 M params, Apache-2.0).

    Returns a normalised depth map and its Sobel-magnitude edge map.
    Used by the pipeline as an orthogonal cue to RGB for low-contrast scenes.
    """

    def __init__(self, device: str = "cpu", fp16: bool = False, model_id: str = _MODEL_ID) -> None:
        self._device = device
        self._dtype = choose_dtype(device, fp16)
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._loaded = False

    def load(self) -> "DepthAnythingV2Small":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor

        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self._model_id)
        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return normalised depth map float32 [0, 1], H×W (original resolution)."""
        if not self._loaded:
            self.load()

        orig_wh = image.size
        rgb = image.convert("RGB")

        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {
            k: v.to(device=self._device, dtype=self._dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model(**inputs)

        depth = outputs.predicted_depth.squeeze().float().cpu().numpy()
        depth_pil = Image.fromarray(depth).resize(orig_wh, Image.BILINEAR)
        depth_np = np.array(depth_pil).astype(np.float32)

        lo, hi = depth_np.min(), depth_np.max()
        return (depth_np - lo) / (hi - lo + 1e-8)

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        self._loaded = False
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
