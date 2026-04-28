from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Subject types that drive expert routing and segmenter weights.
SUBJECT_TYPES = [
    "portrait",       # person with hair — ViTMatte refiner
    "animal_fur",     # furry animal / pet — ViTMatte refiner
    "product",        # product on clean background — BEN2-heavy
    "plant_thin",     # plant, grass, thin branches — ViTMatte refiner
    "transparent",    # glass, smoke, water — depth-heavy
    "vehicle",        # car, bike — BiRefNet-heavy
    "anime",          # cartoon / illustration
    "complex_multi",  # multiple objects / occlusion
    "generic",        # fallback
]

# Text prompts fed to CLIP for zero-shot classification.
_PROMPTS: Dict[str, str] = {
    "portrait":      "a portrait photo of a person with visible hair",
    "animal_fur":    "a photo of a furry animal or pet",
    "product":       "a product photo on a plain or white background",
    "plant_thin":    "a photo of a plant, tree, grass, or thin branches",
    "transparent":   "a photo of transparent or glass objects, smoke, or water",
    "vehicle":       "a photo of a car, motorcycle, or vehicle",
    "anime":         "an anime drawing or cartoon illustration",
    "complex_multi": "a complex scene with multiple foreground objects",
    "generic":       "a photo of an object or person",
}

# Segmenter weight overrides per subject type.
# Keys match SegmenterID values; absent keys → equal weight (normalised later).
SEGMENTER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "portrait":      {"birefnet_hr": 0.55, "ben2": 0.45, "inspyrenet": 0.00},
    "animal_fur":    {"birefnet_hr": 0.55, "ben2": 0.35, "inspyrenet": 0.10},
    "product":       {"birefnet_hr": 0.30, "ben2": 0.55, "inspyrenet": 0.15},
    "plant_thin":    {"birefnet_hr": 0.50, "ben2": 0.30, "inspyrenet": 0.20},
    "transparent":   {"birefnet_hr": 0.60, "ben2": 0.40, "inspyrenet": 0.00},
    "vehicle":       {"birefnet_hr": 0.50, "ben2": 0.35, "inspyrenet": 0.15},
    "anime":         {"birefnet_hr": 0.40, "ben2": 0.30, "inspyrenet": 0.30},
    "complex_multi": {"birefnet_hr": 0.40, "ben2": 0.45, "inspyrenet": 0.15},
    "generic":       {"birefnet_hr": 0.40, "ben2": 0.40, "inspyrenet": 0.20},
}

# Which Stage-D expert to use per subject type.
EXPERT_MAP: Dict[str, str] = {
    "portrait":      "vitmatte",
    "animal_fur":    "vitmatte",
    "plant_thin":    "vitmatte",
    "product":       "depth_only",
    "transparent":   "depth_only",
    "vehicle":       "depth_only",
    "anime":         "depth_only",
    "complex_multi": "depth_only",
    "generic":       "depth_only",
}


@dataclass
class ClassificationResult:
    subject_type: str           # one of SUBJECT_TYPES
    scores: Dict[str, float]    # raw CLIP scores per type
    expert: str                 # "vitmatte" | "depth_only"
    segmenter_weights: Dict[str, float]


class SubjectClassifier:
    """
    Zero-shot CLIP-based subject classifier.

    openai/clip-vit-base-patch32 — MIT/research-permissive, ~150 MB.
    Falls back to "generic" gracefully if CLIP is unavailable.
    """

    _MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model = None
        self._processor = None
        self._text_features = None
        self._loaded = False

    def load(self) -> "SubjectClassifier":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self._MODEL_ID)
        self._model = CLIPModel.from_pretrained(self._MODEL_ID).to(self._device)
        self._model.eval()

        # Pre-compute text features once.
        prompts = [_PROMPTS[t] for t in SUBJECT_TYPES]
        text_inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}
        with torch.inference_mode():
            text_feats = self._model.get_text_features(**text_inputs)
            if not isinstance(text_feats, torch.Tensor):
                text_feats = text_feats.pooler_output
            self._text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)

    def classify(self, image: Image.Image) -> ClassificationResult:
        if not self._loaded:
            self.load()

        import torch

        rgb = image.convert("RGB")
        # Resize to 224 — CLIP's native resolution.
        thumb = rgb.resize((224, 224), Image.BILINEAR)
        inputs = self._processor(images=thumb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            image_features = self._model.get_image_features(**inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)

        probs = logits.squeeze().cpu().numpy()
        scores = {t: float(probs[i]) for i, t in enumerate(SUBJECT_TYPES)}
        subject_type = max(scores, key=scores.__getitem__)

        return ClassificationResult(
            subject_type=subject_type,
            scores=scores,
            expert=EXPERT_MAP[subject_type],
            segmenter_weights=SEGMENTER_WEIGHTS[subject_type],
        )

    @staticmethod
    def fallback() -> ClassificationResult:
        return ClassificationResult(
            subject_type="generic",
            scores={t: 1.0 / len(SUBJECT_TYPES) for t in SUBJECT_TYPES},
            expert="depth_only",
            segmenter_weights=SEGMENTER_WEIGHTS["generic"],
        )
