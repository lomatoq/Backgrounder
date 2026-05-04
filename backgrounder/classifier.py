from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from PIL import Image


# Subject types that drive expert routing and segmenter weights.
SUBJECT_TYPES = [
    "portrait",
    "animal_fur",
    "product",
    "product_opaque",
    "product_glass",
    "plant_thin",
    "transparent",
    "transparent_object",
    "vehicle",
    "anime",
    "flat_cartoon",
    "complex_multi",
    "text_logo",
    "sticker_logo",
    "text_glow",
    "document_screenshot",
    "solid_screen_keying",
    "busy_scene",
    "generic",
]


_PROMPTS: Dict[str, str] = {
    "portrait": "a photo of a person, human face, selfie, portrait, or full body person",
    "animal_fur": "a photo of a furry animal, pet, cat, dog, or wildlife",
    "product": "a product photo of an object, appliance, toy, gadget, item, or 3D render",
    "product_opaque": "an opaque product photo, toy, gadget, package, shoe, tool, or hard-edged object",
    "product_glass": "a reflective or transparent product, glass bottle, jewelry, crystal, acrylic, or shiny translucent object",
    "plant_thin": "a photo of a plant, flower, tree, grass, leaves, or thin branches",
    "transparent": "a photo of transparent glass, crystal, water, smoke, or translucent material",
    "transparent_object": "a transparent object with see-through material, glass, plastic, smoke, liquid, or reflections",
    "vehicle": "a photo of a car, motorcycle, truck, bicycle, or other vehicle",
    "anime": "an anime drawing, cartoon illustration, or digital character art",
    "flat_cartoon": "a flat-color cartoon, sticker, mascot, game sprite, or outlined illustration",
    "complex_multi": "a group photo, multiple people, many separate foreground objects, or occluded scene",
    "text_logo": "text, typography, logo, brand mark, icon, or graphic design on a plain background",
    "sticker_logo": "a sticker, logo, decal, icon, badge, mascot mark, or graphic cutout with outline",
    "text_glow": "glowing text, neon typography, gold text, transparent text effects, shadows, or luminous logo text",
    "document_screenshot": "a screenshot, document, UI panel, chart, table, code block, meme, or flat screen capture",
    "solid_screen_keying": "an object photographed or drawn on a solid blue, green, red, or chroma key background",
    "busy_scene": "a natural photo with a busy background, cluttered scene, room, street, landscape, or textured backdrop",
    "generic": "a photo of an inanimate object, furniture, food, or scene with no people, animals, or vehicles",
}


_PRODUCT_WEIGHTS = {"birefnet_hr": 0.30, "ben2": 0.55, "inspyrenet": 0.15}
_NATURAL_WEIGHTS = {"birefnet_hr": 0.55, "ben2": 0.35, "inspyrenet": 0.10}
_CARTOON_WEIGHTS = {"birefnet_hr": 0.40, "ben2": 0.30, "inspyrenet": 0.30}
_TEXT_WEIGHTS = {"birefnet_hr": 0.25, "ben2": 0.65, "inspyrenet": 0.10}
_GENERIC_WEIGHTS = {"birefnet_hr": 0.40, "ben2": 0.40, "inspyrenet": 0.20}


SEGMENTER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "portrait": {"birefnet_hr": 0.55, "ben2": 0.45, "inspyrenet": 0.00},
    "animal_fur": _NATURAL_WEIGHTS,
    "product": _PRODUCT_WEIGHTS,
    "product_opaque": _PRODUCT_WEIGHTS,
    "product_glass": {"birefnet_hr": 0.50, "ben2": 0.40, "inspyrenet": 0.10},
    "plant_thin": {"birefnet_hr": 0.50, "ben2": 0.30, "inspyrenet": 0.20},
    "transparent": {"birefnet_hr": 0.60, "ben2": 0.40, "inspyrenet": 0.00},
    "transparent_object": {"birefnet_hr": 0.60, "ben2": 0.40, "inspyrenet": 0.00},
    "vehicle": {"birefnet_hr": 0.50, "ben2": 0.35, "inspyrenet": 0.15},
    "anime": _CARTOON_WEIGHTS,
    "flat_cartoon": _CARTOON_WEIGHTS,
    "complex_multi": {"birefnet_hr": 0.50, "ben2": 0.40, "inspyrenet": 0.10},
    "text_logo": _TEXT_WEIGHTS,
    "sticker_logo": _TEXT_WEIGHTS,
    "text_glow": {"birefnet_hr": 0.35, "ben2": 0.50, "inspyrenet": 0.15},
    "document_screenshot": _PRODUCT_WEIGHTS,
    "solid_screen_keying": _PRODUCT_WEIGHTS,
    "busy_scene": {"birefnet_hr": 0.50, "ben2": 0.40, "inspyrenet": 0.10},
    "generic": _GENERIC_WEIGHTS,
}


EXPERT_MAP: Dict[str, str] = {
    "portrait": "vitmatte",
    "animal_fur": "vitmatte",
    "plant_thin": "vitmatte",
    "product": "depth_only",
    "product_opaque": "depth_only",
    "product_glass": "depth_only",
    "transparent": "depth_only",
    "transparent_object": "depth_only",
    "vehicle": "depth_only",
    "anime": "depth_only",
    "flat_cartoon": "depth_only",
    "complex_multi": "depth_only",
    "text_logo": "color_key",
    "sticker_logo": "color_key",
    "text_glow": "color_key",
    "document_screenshot": "depth_only",
    "solid_screen_keying": "color_key",
    "busy_scene": "depth_only",
    "generic": "depth_only",
}


@dataclass
class ClassificationResult:
    subject_type: str
    scores: Dict[str, float]
    expert: str
    segmenter_weights: Dict[str, float]


class SubjectClassifier:
    """
    Zero-shot CLIP-based subject/material advisor.

    The final route is still decided by the CG analyzer in stages/router.py.
    CLIP can suggest image family/material, but it cannot bypass hard safety
    rules such as "no chroma cleanup on busy natural borders".
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

        thumb = image.convert("RGB").resize((224, 224), Image.BILINEAR)
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
        subject_type = _resolve_close_subject(scores, subject_type)

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


def _resolve_close_subject(scores: Dict[str, float], winner: str) -> str:
    winner_score = scores[winner]

    # In background removal, a real person should usually win over nearby UI,
    # text, table, or graphic cues. CLIP often sees a logo/table and a person
    # with almost equal confidence; choosing portrait is the safer cutout route.
    if (
        winner in {
            "text_logo", "sticker_logo", "text_glow", "document_screenshot",
            "solid_screen_keying", "product", "product_opaque", "busy_scene",
            "generic",
        }
        and scores.get("portrait", 0.0) >= winner_score * 0.86
    ):
        return "portrait"

    # Glass/transparent product prompts overlap heavily. Prefer the more
    # specific glass/product route when it is close to generic transparent.
    if winner == "transparent" and scores.get("product_glass", 0.0) >= winner_score * 0.88:
        return "product_glass"

    return winner
