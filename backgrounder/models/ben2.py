from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from backgrounder.result import SegmentationOutput
from .base import BaseSegmenter

# MIT licensed.  BEN2_Base uses a MVANet-decoder with Confidence-Guided Matting.
_MODEL_ID = "PramaLLC/BEN2"


class BEN2Segmenter(BaseSegmenter):
    """
    BEN2_Base — MIT, commercial OK.

    BEN2 does not use a standard HuggingFace model_type, so we load its
    custom class directly from the downloaded repo files via snapshot_download.

    Key advantage: confidence map — pixels near 0.5 are uncertain,
    pixels near 0 or 1 are definite.
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
        from huggingface_hub import snapshot_download

        # Download the full repo (weights + model code).
        repo_dir = Path(snapshot_download(_MODEL_ID))

        # Dynamically import BEN2 class from the repo's own Python file.
        model_file = repo_dir / "BEN2.py"
        if not model_file.exists():
            # Some versions use a different filename.
            candidates = list(repo_dir.glob("*.py"))
            candidates = [f for f in candidates if f.name != "__init__.py"]
            if not candidates:
                raise FileNotFoundError(f"No Python model file found in {repo_dir}")
            model_file = candidates[0]

        spec = importlib.util.spec_from_file_location("_ben2_module", model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_ben2_module"] = module
        spec.loader.exec_module(module)

        # Discover the model class — it has a `loadcheckpoints` method.
        import inspect, torch.nn as nn
        BEN2Class = None
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, nn.Module) and hasattr(obj, "loadcheckpoints"):
                BEN2Class = obj
                break
        if BEN2Class is None:
            # Fallback: try common names
            for name in ("BEN2", "BEN2model", "BEN2_model", "Model"):
                if hasattr(module, name):
                    BEN2Class = getattr(module, name)
                    break
        if BEN2Class is None:
            raise AttributeError(
                f"Could not find BEN2 model class in {model_file}. "
                f"Available: {[n for n, _ in inspect.getmembers(module, inspect.isclass)]}"
            )
        self._model = BEN2Class()
        self._model.loadcheckpoints(str(repo_dir))

        if self._device != "cpu":
            self._model = self._model.to(self._device)
        if self._fp16 and self._device.startswith("cuda"):
            self._model = self._model.half()
        self._model.eval()

    def _predict(self, image: Image.Image) -> SegmentationOutput:
        import torch

        rgb = image.convert("RGB")
        orig_wh = rgb.size

        with torch.inference_mode():
            result = self._model.inference(image=rgb, refine_foreground=False)

        if isinstance(result, Image.Image):
            rgba = result.convert("RGBA") if result.mode != "RGBA" else result
        elif isinstance(result, (list, tuple)):
            candidate = result[1] if len(result) > 1 else result[0]
            rgba = candidate.convert("RGBA") if isinstance(candidate, Image.Image) else result[0].convert("RGBA")
        else:
            raise RuntimeError(f"Unexpected BEN2 output type: {type(result)}")

        if rgba.size != orig_wh:
            rgba = rgba.resize(orig_wh, Image.LANCZOS)

        alpha = np.array(rgba)[:, :, 3].astype(np.float32) / 255.0
        confidence = (2.0 * np.abs(alpha - 0.5)).astype(np.float32)

        return SegmentationOutput(alpha=alpha, confidence=confidence, model_name=self.name)

    def unload(self) -> None:
        import torch
        del self._model
        self._model = None
        self._loaded = False
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
