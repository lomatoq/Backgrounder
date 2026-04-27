from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from PIL import Image

from backgrounder.utils import choose_dtype

# ViTMatte — code: Apache-2.0 / MIT.
# Pre-trained weights: trained on Adobe Composition-1k — NON-COMMERCIAL.
# For commercial deployment, retrain on your own licensed dataset.
# Set allow_nc_weights=True only if you accept the NC restriction.
_MODEL_SMALL = "hustvl/vitmatte-small-composition-1k"
_MODEL_BASE = "hustvl/vitmatte-base-composition-1k"


class ViTMatteRefiner:
    """
    ViTMatte trimap-based alpha refiner.

    Takes (image, trimap) and returns a refined alpha only in the unknown band.
    Definite fg/bg pixels are kept from the coarse alpha — ViTMatte only
    fills in the uncertain boundary region.

    ⚠  Pre-trained weights are NC (Adobe Composition-1k).
       Pass allow_nc_weights=True to acknowledge.
    """

    def __init__(
        self,
        device: str = "cpu",
        fp16: bool = False,
        model_id: str = _MODEL_SMALL,
        allow_nc_weights: bool = False,
    ) -> None:
        if not allow_nc_weights:
            raise RuntimeError(
                "ViTMatte weights are trained on Adobe Composition-1k (non-commercial). "
                "Pass allow_nc_weights=True to use them, or retrain on licensed data."
            )
        self._device = device
        self._dtype = choose_dtype(device, fp16)
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._loaded = False

    def load(self) -> "ViTMatteRefiner":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self._processor = VitMatteImageProcessor.from_pretrained(self._model_id)
        self._model = VitMatteForImageMatting.from_pretrained(self._model_id)
        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

    def refine(
        self,
        image: Image.Image,
        coarse_alpha: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        """
        Return refined alpha float32 [0,1].

        Only the unknown band (trimap==128) is replaced; fg/bg keep coarse values.
        """
        if not self._loaded:
            self.load()

        rgb = image.convert("RGB")
        orig_wh = rgb.size

        # ViTMatte expects trimap as a single-channel image with values {0, 128, 255}.
        trimap_pil = Image.fromarray(trimap, mode="L")

        inputs = self._processor(images=rgb, trimaps=trimap_pil, return_tensors="pt")
        inputs = {k: v.to(device=self._device, dtype=self._dtype) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        refined = outputs.alphas.squeeze().float().cpu().numpy()

        # Resize to original resolution.
        if (refined.shape[1], refined.shape[0]) != orig_wh:
            refined_pil = Image.fromarray((refined * 255).astype(np.uint8), mode="L")
            refined = np.array(refined_pil.resize(orig_wh, Image.LANCZOS)).astype(np.float32) / 255.0

        # Composite: keep coarse outside unknown band, use ViTMatte inside.
        unknown = (trimap == 128).astype(np.float32)
        if unknown.shape != refined.shape:
            unknown_pil = Image.fromarray((unknown * 255).astype(np.uint8), mode="L")
            unknown = np.array(unknown_pil.resize(orig_wh, Image.LANCZOS)).astype(np.float32) / 255.0

        result = coarse_alpha * (1.0 - unknown) + refined * unknown
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        self._loaded = False
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
