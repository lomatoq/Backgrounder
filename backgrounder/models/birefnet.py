from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from backgrounder.result import SegmentationOutput
from backgrounder.utils import choose_dtype
from .base import BaseSegmenter

# MIT licensed; HR-matting variant trained at 2048×2048 on Freepik H200×4.
_MODEL_ID = "ZhengPeng7/BiRefNet_HR-matting"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _make_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class BiRefNetSegmenter(BaseSegmenter):
    """
    BiRefNet_HR-matting — MIT, commercial OK.

    Trained at 2048×2048. We default to 1024 for speed; pass input_size=2048
    for maximum detail on high-res inputs.
    """

    def __init__(
        self,
        device: str = "cpu",
        fp16: bool = False,
        input_size: int = 1024,
        model_id: str = _MODEL_ID,
    ) -> None:
        self._device = device
        self._dtype = choose_dtype(device, fp16)
        self._input_size = input_size
        self._model_id = model_id
        self._model: Optional[torch.nn.Module] = None
        self._transform = _make_transform(input_size)

    @property
    def name(self) -> str:
        return f"birefnet_hr@{self._input_size}"

    def _load(self) -> None:
        from transformers import AutoModelForImageSegmentation

        self._model = AutoModelForImageSegmentation.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

    def _predict(self, image: Image.Image) -> SegmentationOutput:
        orig_wh = image.size
        rgb = image.convert("RGB")
        tensor = self._transform(rgb).unsqueeze(0).to(device=self._device, dtype=self._dtype)

        with torch.inference_mode():
            output = self._model(tensor)

        # BiRefNet returns a list of multi-scale preds; last is finest.
        if isinstance(output, (list, tuple)):
            pred = output[-1]
        elif hasattr(output, "logits"):
            pred = output.logits
        else:
            pred = output

        alpha_t = pred.sigmoid().squeeze().float().cpu().numpy()
        # Resize back to original resolution.
        alpha_pil = Image.fromarray((alpha_t * 255).astype(np.uint8), mode="L")
        alpha = np.array(alpha_pil.resize(orig_wh, Image.LANCZOS)).astype(np.float32) / 255.0

        return SegmentationOutput(alpha=alpha, model_name=self.name)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
