from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_propagation


_GRAPHIC_TYPES = {
    "anime",
    "flat_cartoon",
    "sticker_logo",
    "text_logo",
    "text_glow",
    "document_screenshot",
    "solid_screen_keying",
}


def refine_solid_background_alpha(
    image: Image.Image,
    alpha: np.ndarray,
    subject_type: str,
) -> tuple[np.ndarray, dict]:
    """
    Remove flat-background halos from a neural alpha matte.

    Dedicated segmenters often keep a soft ring of solid blue/green backdrop
    around cartoons, logos, and products. For those images, a classic keyer is
    the right second opinion: estimate the clean plate from the border, compute
    per-pixel color distance from that plate, and only cut into non-core alpha
    pixels so similarly colored foreground interiors are preserved.
    """
    if subject_type in {"portrait", "animal_fur", "plant_thin", "transparent", "transparent_object", "product_glass"}:
        return alpha, {"solid_bg_cleanup": "skipped_subject"}

    img = np.array(image.convert("RGB")).astype(np.float32)
    h, w = alpha.shape
    border_px = max(8, min(h, w) // 24)

    border = np.zeros((h, w), dtype=bool)
    border[:border_px, :] = True
    border[-border_px:, :] = True
    border[:, :border_px] = True
    border[:, -border_px:] = True

    samples = img[border]
    bg_color = np.median(samples, axis=0)
    border_dist = _rgb_distance(samples, bg_color)
    bg_p95 = float(np.percentile(border_dist, 95))
    bg_p99 = float(np.percentile(border_dist, 99))

    # Not a clean plate: avoid damaging natural / busy backgrounds.
    if bg_p95 > 36.0 or bg_p99 > 58.0:
        return alpha, {
            "solid_bg_cleanup": "skipped_busy_border",
            "bg_p95": round(bg_p95, 2),
            "bg_p99": round(bg_p99, 2),
        }

    dist = _rgb_distance(img.reshape(-1, 3), bg_color).reshape(h, w)

    black_clip = max(bg_p95 * 2.0 + 18.0, 34.0)
    white_clip = max(black_clip + 44.0, bg_p95 * 4.0, 92.0)
    key_alpha = _smoothstep(black_clip, white_clip, dist)

    # Preserve confident foreground interiors. The cleanup acts on edge and
    # outside pixels only, which is where the backdrop-colored halo lives.
    core = binary_erosion(alpha > 0.992, structure=np.ones((5, 5), dtype=bool))
    edge_or_outside = ~core

    cleaned = alpha.copy()
    cleaned[edge_or_outside] = np.minimum(cleaned[edge_or_outside], key_alpha[edge_or_outside])

    # Matte controls equivalent to keyer black/white clip. Keep this crisp for
    # illustration/product cutouts; the previous blur reintroduced a dark rim.
    cleaned = np.where(cleaned < 0.10, 0.0, cleaned)
    cleaned = np.where(cleaned > 0.94, 1.0, cleaned)

    changed = float(np.mean(np.abs(cleaned - alpha)))
    return np.clip(cleaned, 0.0, 1.0).astype(np.float32), {
        "solid_bg_cleanup": "applied",
        "bg_rgb": [round(float(v), 1) for v in bg_color],
        "bg_p95": round(bg_p95, 2),
        "black_clip": round(black_clip, 2),
        "white_clip": round(white_clip, 2),
        "alpha_delta_mean": round(changed, 5),
    }


def despill_solid_background(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    bg_color: list[float] | tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """
    Recover edge foreground colors after keying a solid background.

    This is intentionally simpler than the generic foreground estimator: for a
    clean plate, direct inverse compositing against the known background avoids
    the smeared dark rim caused by Gaussian foreground propagation.
    """
    img = image_rgb.astype(np.float32)
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    recovered = img.copy()
    edge = (a > 0.08) & (a < 0.98)
    if edge.any():
        denom = np.maximum(a[..., None], 0.18)
        inv = (img - bg * (1.0 - a[..., None])) / denom
        blend = np.clip((a - 0.08) / 0.35, 0.0, 1.0)[..., None]
        recovered = np.where(edge[..., None], blend * inv + (1.0 - blend) * img, img)

    return np.clip(recovered, 0, 255).astype(np.uint8)


def remove_solid_background_spill(
    image: Image.Image,
    alpha: np.ndarray,
    subject_type: str,
) -> tuple[np.ndarray, dict]:
    """
    Final export cleanup for solid-color backgrounds.

    This runs after all neural refiners. It does not try to solve the whole
    matte. It only removes leftover pixels whose original RGB is still close to
    the detected clean-plate color, which is exactly the blue rim failure mode.
    """
    if subject_type in {"portrait", "animal_fur", "plant_thin", "transparent", "transparent_object", "product_glass"}:
        return alpha, {"solid_bg_spill_cleanup": "skipped_subject"}

    img = np.array(image.convert("RGB")).astype(np.float32)
    h, w = alpha.shape
    border = _border_mask(h, w)

    samples = img[border]
    bg_color = np.median(samples, axis=0)
    border_dist = _rgb_distance(samples, bg_color)
    bg_p95 = float(np.percentile(border_dist, 95))
    bg_p99 = float(np.percentile(border_dist, 99))

    if bg_p95 > 34.0 or bg_p99 > 56.0:
        return alpha, {
            "solid_bg_spill_cleanup": "skipped_busy_border",
            "solid_bg_p95": round(bg_p95, 2),
            "solid_bg_p99": round(bg_p99, 2),
        }

    black_clip = max(bg_p99 + 8.0, 22.0)
    white_clip = max(black_clip + 78.0, 108.0)
    keep, chroma_meta = _solid_background_keep(
        image_rgb=img,
        bg_color=bg_color,
        black_clip=black_clip,
        white_clip=white_clip,
    )
    connected_bg, candidate, connected_meta = _border_connected_background(
        keep=keep,
        border=border,
        subject_type=subject_type,
    )
    protect, protect_meta = _foreground_protect_mask(
        image_rgb=img,
        alpha=alpha,
        bg_color=bg_color,
        subject_type=subject_type,
    )

    cleaned = alpha.copy()
    editable = candidate & ~protect
    cleaned[editable] = np.minimum(cleaned[editable], keep[editable])
    cleaned[connected_bg & ~protect & (keep < 0.16)] = 0.0
    cleaned = np.where(cleaned < 0.06, 0.0, cleaned)
    cleaned = np.where(cleaned > 0.985, 1.0, cleaned)

    changed = float(np.mean(np.abs(cleaned - alpha)))
    return cleaned.astype(np.float32), {
        "solid_bg_spill_cleanup": "applied",
        "solid_bg_bg_rgb": [round(float(v), 1) for v in bg_color],
        "solid_bg_p95": round(bg_p95, 2),
        "solid_bg_black_clip": round(black_clip, 2),
        "solid_bg_white_clip": round(white_clip, 2),
        "solid_bg_alpha_delta_mean": round(changed, 5),
        **chroma_meta,
        **connected_meta,
        **protect_meta,
    }


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


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _solid_background_keep(
    image_rgb: np.ndarray,
    bg_color: np.ndarray,
    black_clip: float,
    white_clip: float,
) -> tuple[np.ndarray, dict]:
    # Two-key strategy:
    # 1) RGB distance catches exact clean-plate pixels.
    # 2) screen-color keying catches blue/green/red spill mixed into antialiased
    #    edges where Euclidean distance is already too large.
    h, w = image_rgb.shape[:2]
    dist = _rgb_distance(image_rgb.reshape(-1, 3), bg_color).reshape(h, w)
    rgb_keep = _smoothstep(black_clip, white_clip, dist)
    chroma_keep, chroma_meta = _screen_color_keep(image_rgb, bg_color)
    keep = np.minimum(rgb_keep, chroma_keep)
    return keep.astype(np.float32), chroma_meta


def _border_connected_background(
    keep: np.ndarray,
    border: np.ndarray,
    subject_type: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Treat the clean-plate key as background only when it is connected to border.

    This is the main safety rail: a blue logo detail, costume part, or dark
    cartoon outline may be chroma-similar, but if it is enclosed by foreground
    it is not the background. A tiny dilation catches antialiased spill just
    inside the connected background without punching holes through the subject.
    """
    seed = border & (keep < 0.22)
    if not seed.any():
        empty = np.zeros_like(keep, dtype=bool)
        return empty, empty, {
            "solid_bg_connected": False,
            "solid_bg_connected_ratio": 0.0,
            "solid_bg_candidate_ratio": 0.0,
        }

    if subject_type in _GRAPHIC_TYPES:
        background_like_threshold = 0.58
        halo_threshold = 0.74
        halo_iters = 1
    else:
        background_like_threshold = 0.50
        halo_threshold = 0.66
        halo_iters = 1

    background_like = keep < background_like_threshold
    connected = binary_propagation(seed, mask=background_like)

    if halo_iters > 0:
        halo_shell = binary_dilation(
            connected,
            structure=np.ones((3, 3), dtype=bool),
            iterations=halo_iters,
        )
        halo_shell &= keep < halo_threshold
    else:
        halo_shell = np.zeros_like(connected, dtype=bool)

    candidate = connected | halo_shell
    return connected, candidate, {
        "solid_bg_connected": True,
        "solid_bg_connected_ratio": round(float(connected.mean()), 5),
        "solid_bg_candidate_ratio": round(float(candidate.mean()), 5),
        "solid_bg_connected_threshold": background_like_threshold,
        "solid_bg_halo_threshold": halo_threshold,
    }


def _screen_color_keep(
    image_rgb: np.ndarray,
    bg_color: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    CorridorKey-style screen-color cleanup generalized from the clean border.

    It is deliberately post-model and conservative: neutral/low-saturation
    backgrounds fall back to RGB distance, while saturated screen colors get a
    hue/dominance matte that removes leftover blue/green/red edge spill.
    """
    rgb = np.clip(image_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    bg = np.clip(bg_color.astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = _rgb_to_hsv(rgb)
    bg_hsv = _rgb_to_hsv(bg.reshape(1, 1, 3))[0, 0]

    bg_hue = float(bg_hsv[0])
    bg_sat = float(bg_hsv[1])
    bg_val = float(bg_hsv[2])
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    if bg_sat < 0.18:
        return np.ones(image_rgb.shape[:2], dtype=np.float32), {
            "solid_bg_screen_color": "neutral",
            "solid_bg_screen_saturation": round(bg_sat, 3),
        }

    hue_diff = _hue_distance(h, bg_hue)
    hue_close = 1.0 - _smoothstep(0.018, 0.155, hue_diff)
    sat_gate = _smoothstep(0.10, max(0.22, bg_sat * 0.72), s)
    # Bright screens often share hue with dark ink/outline pixels. Key only
    # pixels that are reasonably close to the screen brightness; pure RGB
    # distance still removes exact background pixels.
    value_floor = max(0.16, min(0.52, bg_val * 0.48))
    value_full = max(value_floor + 0.16, min(0.86, bg_val * 0.82))
    val_gate = _smoothstep(value_floor, value_full, v)

    key_channel = int(np.argmax(bg))
    other_channels = [c for c in range(3) if c != key_channel]
    bg_dominance = float(bg[key_channel] - max(bg[other_channels[0]], bg[other_channels[1]]))
    dominance_bgness = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    if bg_dominance > 0.045:
        px_dominance = rgb[..., key_channel] - np.maximum(
            rgb[..., other_channels[0]],
            rgb[..., other_channels[1]],
        )
        dominance_bgness = _smoothstep(
            max(0.02, bg_dominance * 0.16),
            max(0.08, bg_dominance * 0.92),
            px_dominance,
        )

    # Hue catches off-axis screen colors; dominance catches canonical blue/green
    # screen leakage. Low-value / low-saturation foreground outlines are gated
    # away so black line art is not eaten.
    hue_bgness = hue_close * sat_gate * val_gate
    screen_bgness = np.maximum(hue_bgness * 0.82, hue_close * dominance_bgness * sat_gate * val_gate)
    keep = 1.0 - np.clip(screen_bgness, 0.0, 1.0)

    return keep.astype(np.float32), {
        "solid_bg_screen_color": _screen_color_label(bg_hue, bg_sat),
        "solid_bg_screen_saturation": round(bg_sat, 3),
        "solid_bg_screen_value": round(bg_val, 3),
        "solid_bg_value_floor": round(value_floor, 3),
        "solid_bg_value_full": round(value_full, 3),
        "solid_bg_key_channel": ["red", "green", "blue"][key_channel],
        "solid_bg_key_dominance": round(bg_dominance, 3),
    }


def _foreground_protect_mask(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    bg_color: np.ndarray,
    subject_type: str,
) -> tuple[np.ndarray, dict]:
    """
    Preserve high-confidence foreground strokes while keying clean plates.

    Cartoon/logo cutouts often have a deliberate dark or colored outline that is
    connected to the background. A pure connected-component keyer cannot tell
    that outline from screen-color spill, so we protect confident foreground
    core pixels plus high-alpha, high-contrast ink/stroke pixels.
    """
    dist = _rgb_distance(image_rgb.reshape(-1, 3), bg_color).reshape(alpha.shape)
    bg_luma = float(np.dot(bg_color, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
    px_luma = (
        image_rgb[..., 0] * 0.2126
        + image_rgb[..., 1] * 0.7152
        + image_rgb[..., 2] * 0.0722
    )
    foreground_contrast = (dist > 30.0) | (np.abs(px_luma - bg_luma) > 18.0)
    core = binary_erosion(
        (alpha > 0.992) & foreground_contrast,
        structure=np.ones((3, 3), dtype=bool),
    )

    if subject_type not in _GRAPHIC_TYPES:
        return core, {
            "solid_bg_protect_ratio": round(float(core.mean()), 5),
            "solid_bg_protect_mode": "core",
        }

    # Keep intentional line art / colored stickers that the model is already
    # confident about. Exact background pixels remain editable, even if a base
    # segmenter assigned them too much alpha.
    contrast_stroke = (dist > 26.0) | (px_luma < bg_luma * 0.88)
    stroke = (alpha > 0.78) & contrast_stroke
    stroke = binary_dilation(stroke, structure=np.ones((3, 3), dtype=bool), iterations=1)
    stroke &= (alpha > 0.48) & foreground_contrast

    protect = core | stroke
    return protect, {
        "solid_bg_protect_ratio": round(float(protect.mean()), 5),
        "solid_bg_protect_mode": "core_stroke",
    }


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


def _hue_distance(hue: np.ndarray, key_hue: float) -> np.ndarray:
    diff = np.abs(hue - key_hue)
    return np.minimum(diff, 1.0 - diff).astype(np.float32)


def _screen_color_label(hue: float, saturation: float) -> str:
    if saturation < 0.18:
        return "neutral"
    deg = (hue * 360.0) % 360.0
    if deg < 18 or deg >= 342:
        return "red"
    if deg < 48:
        return "orange"
    if deg < 75:
        return "yellow"
    if deg < 165:
        return "green"
    if deg < 195:
        return "cyan"
    if deg < 255:
        return "blue"
    if deg < 292:
        return "purple"
    return "magenta"
