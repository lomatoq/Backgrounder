from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes


_NATURAL_EDGE_TYPES = {
    "portrait",
    "animal_fur",
    "plant_thin",
    "transparent",
    "transparent_object",
    "product_glass",
}
_FLAT_ASSET_TYPES = {
    "anime",
    "flat_cartoon",
    "text_logo",
    "sticker_logo",
    "text_glow",
    "solid_screen_keying",
    "product",
    "product_opaque",
    "vehicle",
    "document_screenshot",
    "generic",
}
_CARTOON_TYPES = {"anime", "flat_cartoon", "sticker_logo"}
_TEXT_TYPES = {"text_logo", "text_glow"}
_SDMATTE_AUTO_TYPES = {
    "animal_fur",
    "complex_multi",
    "plant_thin",
    "transparent",
    "transparent_object",
    "product_glass",
}


@dataclass(frozen=True)
class CGFeatures:
    background_kind: str
    image_family: str
    material_hint: str
    clean_border: bool
    keyable: bool
    bg_rgb: tuple[float, float, float]
    border_p95: float
    border_p99: float
    bg_saturation: float
    bg_value: float
    uncertain_band_ratio: float | None
    alpha_hole_ratio: float | None
    alpha_halo_ratio: float | None
    transparent_cue: float

    def metadata(self) -> dict:
        data = {
            "cg_background": self.background_kind,
            "cg_image_family": self.image_family,
            "cg_material_hint": self.material_hint,
            "cg_clean_border": self.clean_border,
            "cg_keyable": self.keyable,
            "cg_bg_rgb": [round(float(v), 1) for v in self.bg_rgb],
            "cg_border_p95": round(float(self.border_p95), 2),
            "cg_border_p99": round(float(self.border_p99), 2),
            "cg_bg_saturation": round(float(self.bg_saturation), 3),
            "cg_bg_value": round(float(self.bg_value), 3),
            "cg_transparent_cue": round(float(self.transparent_cue), 4),
        }
        if self.uncertain_band_ratio is not None:
            data["cg_uncertain_band_ratio"] = round(float(self.uncertain_band_ratio), 5)
        if self.alpha_hole_ratio is not None:
            data["cg_alpha_hole_ratio"] = round(float(self.alpha_hole_ratio), 5)
        if self.alpha_halo_ratio is not None:
            data["cg_alpha_halo_ratio"] = round(float(self.alpha_halo_ratio), 5)
        return data


@dataclass(frozen=True)
class ImageRoute:
    route_id: str
    background_kind: str
    image_family: str
    material_hint: str
    use_solid_background_cleanup: bool
    use_cartoon_snap: bool
    use_graphic_alpha_normalize: bool
    allow_sdmatte_auto: bool
    clean_border: bool
    keyable: bool
    bg_rgb: tuple[float, float, float] | None
    border_p95: float | None
    border_p99: float | None
    uncertain_band_ratio: float | None
    alpha_hole_ratio: float | None
    alpha_halo_ratio: float | None
    neural_subject_type: str
    notes: tuple[str, ...] = ()

    def metadata(self) -> dict:
        data = {
            "route_id": self.route_id,
            "route_background": self.background_kind,
            "route_image_family": self.image_family,
            "route_material_hint": self.material_hint,
            "route_clean_border": self.clean_border,
            "route_keyable": self.keyable,
            "route_use_solid_bg_cleanup": self.use_solid_background_cleanup,
            "route_use_cartoon_snap": self.use_cartoon_snap,
            "route_use_graphic_alpha_normalize": self.use_graphic_alpha_normalize,
            "route_allow_sdmatte_auto": self.allow_sdmatte_auto,
            "route_neural_subject_type": self.neural_subject_type,
            "route_notes": list(self.notes),
        }
        if self.bg_rgb is not None:
            data["route_bg_rgb"] = [round(float(v), 1) for v in self.bg_rgb]
        if self.border_p95 is not None:
            data["route_border_p95"] = round(float(self.border_p95), 2)
        if self.border_p99 is not None:
            data["route_border_p99"] = round(float(self.border_p99), 2)
        if self.uncertain_band_ratio is not None:
            data["route_uncertain_band_ratio"] = round(float(self.uncertain_band_ratio), 5)
        if self.alpha_hole_ratio is not None:
            data["route_alpha_hole_ratio"] = round(float(self.alpha_hole_ratio), 5)
        if self.alpha_halo_ratio is not None:
            data["route_alpha_halo_ratio"] = round(float(self.alpha_halo_ratio), 5)
        return data


def analyze_image_route(
    image: Image.Image,
    subject_type: str,
    alpha: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> ImageRoute:
    features = analyze_cg_features(image, subject_type, alpha=alpha, uncertainty=uncertainty)
    notes: list[str] = []

    use_solid_cleanup = False
    use_cartoon_snap = False
    use_graphic_alpha_normalize = False
    allow_sdmatte = subject_type in _SDMATTE_AUTO_TYPES

    if features.clean_border and features.keyable and subject_type in _FLAT_ASSET_TYPES:
        route_id = f"flat_key_{features.image_family}"
        use_solid_cleanup = True
        use_graphic_alpha_normalize = features.image_family in {"flat_cartoon", "logo_text", "solid_screen_keying"}
        allow_sdmatte = False
        notes.append("clean keyable border; connected chroma cleanup allowed")
    elif subject_type in _NATURAL_EDGE_TYPES or features.material_hint == "transparent":
        route_id = f"natural_{features.material_hint}"
        use_solid_cleanup = False
        use_cartoon_snap = False
        allow_sdmatte = subject_type in _SDMATTE_AUTO_TYPES
        notes.append("natural/transparent material; chroma cleanup blocked")
    elif subject_type in _CARTOON_TYPES:
        route_id = f"cartoon_{features.background_kind}"
        use_solid_cleanup = False
        use_cartoon_snap = False
        use_graphic_alpha_normalize = True
        allow_sdmatte = False
        notes.append("graphic cutout; normalize alpha opacity without old binary snap")
    elif subject_type in _TEXT_TYPES:
        route_id = f"text_{features.background_kind}"
        use_solid_cleanup = features.keyable
        use_graphic_alpha_normalize = subject_type != "text_glow"
        allow_sdmatte = False
        notes.append("text/logo route; color key only on clean keyable plate")
    elif subject_type in {"solid_screen_keying", "document_screenshot"}:
        route_id = f"graphic_{features.background_kind}"
        use_solid_cleanup = features.keyable
        use_graphic_alpha_normalize = True
        allow_sdmatte = False
        notes.append("graphic/screenshot route; normalize opacity")
    elif subject_type == "busy_scene" or features.background_kind == "busy":
        route_id = "busy_scene_safe"
        use_solid_cleanup = False
        allow_sdmatte = False
        notes.append("busy border; block screen-color cleanup")
    else:
        route_id = f"general_{features.image_family}"
        use_solid_cleanup = False
        use_graphic_alpha_normalize = features.image_family in {"flat_cartoon", "logo_text"} and subject_type != "text_glow"
        allow_sdmatte = subject_type in _SDMATTE_AUTO_TYPES
        notes.append("general route; conservative post-processing")

    if features.uncertain_band_ratio is not None and features.uncertain_band_ratio > 0.18:
        notes.append("large uncertain band")
    if features.alpha_hole_ratio is not None and features.alpha_hole_ratio > 0.01:
        notes.append("alpha holes detected")
    if features.alpha_halo_ratio is not None and features.alpha_halo_ratio > 0.02:
        notes.append("solid-background halo likely")

    return ImageRoute(
        route_id=route_id,
        background_kind=features.background_kind,
        image_family=features.image_family,
        material_hint=features.material_hint,
        use_solid_background_cleanup=use_solid_cleanup,
        use_cartoon_snap=use_cartoon_snap,
        use_graphic_alpha_normalize=use_graphic_alpha_normalize,
        allow_sdmatte_auto=allow_sdmatte,
        clean_border=features.clean_border,
        keyable=features.keyable,
        bg_rgb=features.bg_rgb,
        border_p95=features.border_p95,
        border_p99=features.border_p99,
        uncertain_band_ratio=features.uncertain_band_ratio,
        alpha_hole_ratio=features.alpha_hole_ratio,
        alpha_halo_ratio=features.alpha_halo_ratio,
        neural_subject_type=subject_type,
        notes=tuple(notes),
    )


def analyze_cg_features(
    image: Image.Image,
    subject_type: str,
    alpha: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> CGFeatures:
    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = img.shape[:2]
    border = _border_mask(h, w)
    samples = img[border]
    bg_color = np.median(samples, axis=0)
    border_dist = _rgb_distance(samples, bg_color)
    border_p95 = float(np.percentile(border_dist, 95))
    border_p99 = float(np.percentile(border_dist, 99))
    clean_border = border_p95 <= 34.0 and border_p99 <= 56.0

    bg_hsv = _rgb_to_hsv((bg_color / 255.0).reshape(1, 1, 3))[0, 0]
    bg_saturation = float(bg_hsv[1])
    bg_value = float(bg_hsv[2])
    keyable = clean_border and (bg_saturation >= 0.18 or border_p99 <= 10.0)

    transparent_cue = _transparent_cue(img)
    material_hint = _material_hint(subject_type, transparent_cue)
    image_family = _image_family(subject_type)

    if clean_border and keyable:
        background_kind = "clean_keyable"
    elif clean_border:
        background_kind = "clean_neutral"
    elif border_p95 <= 64.0 and border_p99 <= 96.0:
        background_kind = "gradient_or_mild_texture"
    else:
        background_kind = "busy"

    uncertain_ratio = _uncertain_ratio(alpha, uncertainty)
    hole_ratio = _alpha_hole_ratio(alpha)
    halo_ratio = _alpha_halo_ratio(img, alpha, bg_color) if alpha is not None and clean_border else None

    return CGFeatures(
        background_kind=background_kind,
        image_family=image_family,
        material_hint=material_hint,
        clean_border=clean_border,
        keyable=keyable,
        bg_rgb=(float(bg_color[0]), float(bg_color[1]), float(bg_color[2])),
        border_p95=border_p95,
        border_p99=border_p99,
        bg_saturation=bg_saturation,
        bg_value=bg_value,
        uncertain_band_ratio=uncertain_ratio,
        alpha_hole_ratio=hole_ratio,
        alpha_halo_ratio=halo_ratio,
        transparent_cue=transparent_cue,
    )


def _image_family(subject_type: str) -> str:
    if subject_type in {"anime", "flat_cartoon"}:
        return "flat_cartoon"
    if subject_type in {"text_logo", "sticker_logo", "text_glow"}:
        return "logo_text"
    if subject_type in {"product", "product_opaque", "product_glass"}:
        return "product"
    if subject_type in {"portrait", "animal_fur", "plant_thin"}:
        return "natural_fine_edge"
    if subject_type in {"transparent", "transparent_object"}:
        return "transparent"
    if subject_type == "document_screenshot":
        return "document_screenshot"
    if subject_type == "solid_screen_keying":
        return "solid_screen_keying"
    if subject_type == "busy_scene":
        return "busy_scene"
    return subject_type


def _material_hint(subject_type: str, transparent_cue: float) -> str:
    if subject_type in {"portrait", "animal_fur", "plant_thin"}:
        return "fine_edge"
    if subject_type in {"anime", "flat_cartoon", "sticker_logo", "text_logo", "text_glow"}:
        return "graphic"
    if subject_type in {"product", "product_opaque", "vehicle"}:
        return "hard_edge"
    if subject_type in {"transparent", "transparent_object", "product_glass"} or transparent_cue > 0.24:
        return "transparent"
    return "unknown"


def _uncertain_ratio(alpha: np.ndarray | None, uncertainty: np.ndarray | None) -> float | None:
    if alpha is None:
        return None
    alpha_band = (alpha > 0.05) & (alpha < 0.95)
    if uncertainty is not None:
        alpha_band |= uncertainty > 0.15
    return float(alpha_band.mean())


def _alpha_hole_ratio(alpha: np.ndarray | None) -> float | None:
    if alpha is None:
        return None
    solid = alpha > 0.72
    if solid.mean() < 0.01:
        return 0.0
    filled = binary_fill_holes(solid)
    holes = filled & ~solid
    return float(holes.mean())


def _alpha_halo_ratio(img: np.ndarray, alpha: np.ndarray, bg_color: np.ndarray) -> float:
    edge = (alpha > 0.05) & (alpha < 0.88)
    if not edge.any():
        return 0.0
    dist = _rgb_distance(img.reshape(-1, 3), bg_color).reshape(alpha.shape)
    bg_like_edge = edge & (dist < 70.0)
    return float(bg_like_edge.mean())


def _transparent_cue(img: np.ndarray) -> float:
    rgb = np.clip(img / 255.0, 0.0, 1.0)
    hsv = _rgb_to_hsv(rgb)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    low_sat_mid_val = (sat < 0.18) & (val > 0.18) & (val < 0.96)
    highlight = val > 0.94
    return float(low_sat_mid_val.mean() * 0.7 + highlight.mean() * 0.3)


def _border_mask(h: int, w: int) -> np.ndarray:
    border_px = max(8, min(h, w) // 24)
    border = np.zeros((h, w), dtype=bool)
    border[:border_px, :] = True
    border[-border_px:, :] = True
    border[:, :border_px] = True
    border[:, -border_px:] = True
    return border


def _rgb_distance(pixels: np.ndarray, color: np.ndarray) -> np.ndarray:
    diff = pixels.astype(np.float32) - color.astype(np.float32)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    r, g, b = np.moveaxis(rgb[..., :3], -1, 0)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    safe_delta = np.where(delta > 1e-6, delta, 1.0)

    hue = np.zeros_like(maxc, dtype=np.float32)
    red_is_max = (maxc == r) & (delta > 1e-6)
    green_is_max = (maxc == g) & (delta > 1e-6)
    blue_is_max = (maxc == b) & (delta > 1e-6)

    hue = np.where(red_is_max, ((g - b) / safe_delta) % 6.0, hue)
    hue = np.where(green_is_max, ((b - r) / safe_delta) + 2.0, hue)
    hue = np.where(blue_is_max, ((r - g) / safe_delta) + 4.0, hue)
    hue = (hue / 6.0) % 1.0

    sat = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0.0)
    val = maxc
    return np.stack([hue, sat, val], axis=-1).astype(np.float32)
