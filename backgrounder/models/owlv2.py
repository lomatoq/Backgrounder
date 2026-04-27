from __future__ import annotations
from typing import List

import numpy as np
from PIL import Image

# OWLv2 — Google Research — Apache-2.0 licensed.
_MODEL_ID = "google/owlv2-base-patch16-ensemble"

# Text queries used to locate foreground subjects per scene type.
_QUERIES: dict[str, list[str]] = {
    "portrait":      ["a person", "a human face"],
    "animal_fur":    ["an animal", "a pet", "a dog", "a cat"],
    "product":       ["a product", "an object on plain background"],
    "plant_thin":    ["a plant", "a flower", "grass", "thin branches"],
    "transparent":   ["a glass object", "a transparent bottle", "a crystal"],
    "vehicle":       ["a vehicle", "a car", "a motorcycle", "a bicycle"],
    "anime":         ["an anime character", "a cartoon character"],
    "complex_multi": ["a person", "an animal", "an object"],
    "generic":       ["an object", "a subject"],
}


class OWLv2Localizer:
    """
    OWLv2 open-vocabulary object detector.
    google/owlv2-base-patch16-ensemble — Apache-2.0 licensed.

    Returns bounding boxes for the detected foreground subjects, used as
    SAM 2.1 prompts in Stage G for complex / multi-object scenes.
    """

    def __init__(
        self,
        device: str = "cpu",
        score_threshold: float = 0.12,
        model_id: str = _MODEL_ID,
    ) -> None:
        self._device = device
        self._score_threshold = score_threshold
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._loaded = False

    def load(self) -> "OWLv2Localizer":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self._processor = Owlv2Processor.from_pretrained(self._model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(self._model_id)
        self._model = self._model.to(self._device)
        self._model.eval()

    def detect(
        self,
        image: Image.Image,
        subject_type: str = "generic",
        max_boxes: int = 5,
    ) -> List[List[float]]:
        """
        Return list of [x1, y1, x2, y2] bounding boxes (pixel coords),
        sorted by confidence, capped at max_boxes.  Empty list if nothing
        passes the score threshold.
        """
        if not self._loaded:
            self.load()

        import torch

        texts = [_QUERIES.get(subject_type, _QUERIES["generic"])]
        W, H = image.size

        inputs = self._processor(
            text=texts,
            images=image.convert("RGB"),
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            box_threshold=self._score_threshold,
            text_threshold=self._score_threshold * 0.5,
            target_sizes=[(H, W)],
        )

        boxes  = results[0]["boxes"].cpu().numpy()   # (N, 4) xyxy
        scores = results[0]["scores"].cpu().numpy()  # (N,)

        if len(boxes) == 0:
            return []

        order = np.argsort(scores)[::-1][:max_boxes]
        return boxes[order].tolist()

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        self._loaded = False
        if self._device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()
