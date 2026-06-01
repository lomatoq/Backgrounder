from __future__ import annotations

from PIL import Image
import numpy as np
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
_TEXT_TYPES = {"text_logo", "text_glow"}


def visual_alpha_fixes(
    image: Image.Image,
    alpha: np.ndarray,
    subject_type: str,
    route_meta: dict,
) -> tuple[np.ndarray, dict]:
    """
    Last-mile visual QA before foreground composition.

    This stage is deliberately image-domain, not model-domain. It catches cases
    where the numeric quality score is high but a human still sees a leftover
    screen/checker/background fringe.
    """
    meta: dict = {}
    fixed = alpha.astype(np.float32).copy()

    fixed, checker_meta = remove_checkerboard_background(image, fixed, subject_type)
    meta.update(checker_meta)

    fixed, residue_meta = remove_graphic_border_residue(
        image=image,
        alpha=fixed,
        subject_type=subject_type,
        route_meta=route_meta,
    )
    meta.update(residue_meta)

    fixed, portrait_meta = remove_portrait_lower_surface(
        image=image,
        alpha=fixed,
        subject_type=subject_type,
        route_meta=route_meta,
    )
    meta.update(portrait_meta)

    report = visual_quality_report(image, fixed, route_meta)
    meta.update({f"visual_{k}": v for k, v in report.items()})
    return np.clip(fixed, 0.0, 1.0).astype(np.float32), meta


def remove_checkerboard_background(
    image: Image.Image,
    alpha: np.ndarray,
    subject_type: str,
) -> tuple[np.ndarray, dict]:
    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = alpha.shape
    border = _border_mask(h, w)

    # JPEG-compressed checkerboards are rarely perfectly gray; the channels
    # often drift by 15-20 values near text/glow edges.
    neutral = _neutral_mask(img, tolerance=22.0)
    luma = _luma(img)
    border_neutral = neutral & border
    border_neutral_ratio = float(border_neutral.sum()) / max(1, int(border.sum()))
    if border_neutral_ratio < 0.35:
        return alpha, {"checkerboard_cleanup": "skipped_not_neutral_border"}

    values = luma[border_neutral]
    if values.size < 64:
        return alpha, {"checkerboard_cleanup": "skipped_few_samples"}

    lo = float(np.percentile(values, 25))
    hi = float(np.percentile(values, 75))
    if hi - lo < 18.0:
        return alpha, {"checkerboard_cleanup": "skipped_no_two_tones"}

    near_lo = np.abs(luma - lo) < 13.0
    near_hi = np.abs(luma - hi) < 13.0
    bg_like = neutral & (near_lo | near_hi)
    seed = border & bg_like
    if not seed.any():
        return alpha, {"checkerboard_cleanup": "skipped_no_seed"}

    connected = binary_propagation(seed, mask=bg_like)
    halo = binary_dilation(connected, structure=np.ones((3, 3), dtype=bool), iterations=1)
    halo &= bg_like

    # Do not let a confident foreground object get punched out just because it
    # is gray. Exact checker pixels remain removable even if a model assigned
    # them high alpha; non-checker high-alpha cores are protected.
    protect = binary_erosion((alpha > 0.985) & ~bg_like, structure=np.ones((3, 3), dtype=bool))
    editable = (connected | halo) & ~protect
    if subject_type in _TEXT_TYPES:
        editable |= bg_like & (alpha > 0.025)
        expanded = _text_checker_shadow_mask(
            img=img,
            alpha=alpha,
            luma=luma,
            bg_like=bg_like,
            lo=lo,
            hi=hi,
        )
        editable |= expanded & (alpha > 0.025)

    fixed = alpha.copy()
    fixed[editable] = 0.0

    fixed = np.where(fixed < 0.025, 0.0, fixed)
    changed = float(np.mean(np.abs(fixed - alpha)))
    status = "applied" if changed > 0.0001 else "no_change"
    return fixed.astype(np.float32), {
        "checkerboard_cleanup": status,
        "checkerboard_border_neutral_ratio": round(border_neutral_ratio, 4),
        "checkerboard_luma_lo": round(lo, 2),
        "checkerboard_luma_hi": round(hi, 2),
        "checkerboard_connected_ratio": round(float(connected.mean()), 5),
        "checkerboard_alpha_delta_mean": round(changed, 5),
        "checkerboard_text_expanded_ratio": round(float((editable & ~bg_like).mean()), 5),
    }


def _text_checker_shadow_mask(
    img: np.ndarray,
    alpha: np.ndarray,
    luma: np.ndarray,
    bg_like: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    JPEG/screenshot checkerboards often leave dark neutral pixels inside glyph
    counters that are no longer close enough to the two sampled tones. Only use
    this wider key for bright text over a dark checker plate.
    """
    if hi >= 150.0:
        return np.zeros(alpha.shape, dtype=bool)

    foregroundish = (alpha > 0.45) & ~bg_like
    if int(foregroundish.sum()) < 32:
        return np.zeros(alpha.shape, dtype=bool)
    bright_ref = float(np.percentile(luma[foregroundish], 70))
    if bright_ref < 135.0:
        return np.zeros(alpha.shape, dtype=bool)

    neutral_wide = _neutral_mask(img, tolerance=38.0)
    span = max(18.0, hi - lo)
    lower = max(0.0, lo - 18.0)
    upper = min(150.0, hi + max(34.0, span * 0.85))
    return neutral_wide & (luma >= lower) & (luma <= upper)


def remove_graphic_border_residue(
    image: Image.Image,
    alpha: np.ndarray,
    subject_type: str,
    route_meta: dict,
) -> tuple[np.ndarray, dict]:
    if subject_type not in _GRAPHIC_TYPES:
        return alpha, {"graphic_border_residue_cleanup": "skipped_subject"}

    bg_rgb = route_meta.get("route_bg_rgb") or route_meta.get("cg_bg_rgb")
    if not bg_rgb:
        return alpha, {"graphic_border_residue_cleanup": "skipped_no_bg"}

    bg = np.asarray(bg_rgb, dtype=np.float32)
    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = alpha.shape
    border = _border_mask(h, w)

    dist = _rgb_distance(img.reshape(-1, 3), bg).reshape(h, w)
    bg_sat = _rgb_to_hsv(np.clip(bg / 255.0, 0.0, 1.0).reshape(1, 1, 3))[0, 0, 1]

    if bg_sat < 0.16:
        return alpha, {"graphic_border_residue_cleanup": "skipped_neutral_bg"}

    # This is a conservative "human-visible residue" cleanup: only semi-alpha
    # pixels close to the estimated backdrop, and only if connected to border.
    bg_like = dist < 72.0
    seed = border & bg_like
    if not seed.any():
        return alpha, {"graphic_border_residue_cleanup": "skipped_no_seed"}

    connected = binary_propagation(seed, mask=bg_like)
    edge_residue = connected & (alpha > 0.02) & (alpha < 0.72)
    if not edge_residue.any():
        return alpha, {
            "graphic_border_residue_cleanup": "no_residue",
            "graphic_border_bg_rgb": [round(float(v), 1) for v in bg],
        }

    fixed = alpha.copy()
    keep = np.clip((dist - 26.0) / 70.0, 0.0, 1.0)
    fixed[edge_residue] = np.minimum(fixed[edge_residue], keep[edge_residue])
    fixed = np.where(fixed < 0.025, 0.0, fixed)

    changed = float(np.mean(np.abs(fixed - alpha)))
    return fixed.astype(np.float32), {
        "graphic_border_residue_cleanup": "applied",
        "graphic_border_bg_rgb": [round(float(v), 1) for v in bg],
        "graphic_border_residue_ratio": round(float(edge_residue.mean()), 5),
        "graphic_border_alpha_delta_mean": round(changed, 5),
    }


def remove_portrait_lower_surface(
    image: Image.Image,
    alpha: np.ndarray,
    subject_type: str,
    route_meta: dict | None = None,
) -> tuple[np.ndarray, dict]:
    if subject_type != "portrait":
        return alpha, {"portrait_lower_surface_cleanup": "skipped_subject"}

    # This heuristic removes a table/counter/floor the salient-object models
    # wrongly keep under a person. But it samples the "surface" colour from the
    # widest lower foreground — which on a person in dark clothing is the *suit
    # itself*, so it keys out the body into speckle. When a strong segmentation
    # (SAM 3.1 / SAM 2.1) has already produced a confident silhouette, trust it
    # and skip this destructive re-key.
    route_meta = route_meta or {}
    if route_meta.get("sam3_used") or route_meta.get("sam2_used"):
        return alpha, {"portrait_lower_surface_cleanup": "skipped_confident_segmentation"}

    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = alpha.shape
    fg = alpha > 0.02
    if not fg.any():
        return alpha, {"portrait_lower_surface_cleanup": "skipped_empty"}

    y = np.arange(h)
    yy = y[:, None]
    lower = yy >= int(h * 0.52)
    bottom_rows = y >= h - max(10, h // 24)
    bottom_band = bottom_rows[:, None]
    row_coverage = fg.mean(axis=1)
    lower_rows = row_coverage[int(h * 0.58):]
    lower_row_peak = float(lower_rows.max()) if lower_rows.size else 0.0
    bottom_fg_ratio = float(fg[bottom_rows].mean()) if bottom_rows.any() else 0.0

    # A portrait/person should not normally occupy most of the image width at
    # the very bottom. When it does, it is often a table/counter/floor prop that
    # salient-object models incorrectly keep with the person.
    if lower_row_peak < 0.58 or bottom_fg_ratio < 0.24:
        return alpha, {
            "portrait_lower_surface_cleanup": "skipped_no_wide_lower_surface",
            "portrait_lower_row_peak": round(lower_row_peak, 4),
            "portrait_bottom_fg_ratio": round(bottom_fg_ratio, 4),
        }

    side = np.zeros((h, w), dtype=bool)
    side[:, : max(8, w // 28)] = True
    side[:, -max(8, w // 28):] = True
    sample_mask = fg & lower & (bottom_band | side)
    samples = img[sample_mask]
    if samples.shape[0] < 128:
        return alpha, {"portrait_lower_surface_cleanup": "skipped_few_surface_samples"}

    surface_rgb = np.median(samples, axis=0)
    dist = _rgb_distance(img.reshape(-1, 3), surface_rgb).reshape(h, w)
    surface_spread = float(np.percentile(_rgb_distance(samples, surface_rgb), 75))
    color_cut = float(np.clip(surface_spread * 1.7 + 24.0, 48.0, 96.0))
    surface_like = dist < color_cut
    protect = _portrait_protect_mask(img, alpha)

    seed = fg & lower & (bottom_band | side) & surface_like
    if not seed.any():
        return alpha, {"portrait_lower_surface_cleanup": "skipped_no_surface_seed"}

    connected = binary_propagation(seed, mask=(fg & lower & surface_like))
    connected = binary_dilation(connected, structure=np.ones((3, 3), dtype=bool), iterations=1)
    connected &= fg & lower & (dist < color_cut + 18.0)

    wide_rows = np.where((y >= int(h * 0.55)) & (row_coverage > 0.58))[0]
    broad_connected = np.zeros_like(fg, dtype=bool)
    if wide_rows.size:
        surface_start = max(int(h * 0.52), int(wide_rows[0]) - max(2, h // 90))
        broad_zone = yy >= surface_start
        broad_seed = fg & broad_zone & (bottom_band | side) & ~protect
        broad_mask = fg & broad_zone & ~protect
        if broad_seed.any():
            broad_connected = binary_propagation(broad_seed, mask=broad_mask)
    else:
        surface_start = h

    editable = (connected | broad_connected) & ~protect
    if not editable.any():
        return alpha, {
            "portrait_lower_surface_cleanup": "no_editable_surface",
            "portrait_lower_surface_rgb": [round(float(v), 1) for v in surface_rgb],
        }

    fixed = alpha.copy()
    fixed[editable] = 0.0
    fixed = np.where(fixed < 0.025, 0.0, fixed)
    changed = float(np.mean(np.abs(fixed - alpha)))
    return fixed.astype(np.float32), {
        "portrait_lower_surface_cleanup": "applied",
        "portrait_lower_surface_rgb": [round(float(v), 1) for v in surface_rgb],
        "portrait_lower_row_peak": round(lower_row_peak, 4),
        "portrait_bottom_fg_ratio": round(bottom_fg_ratio, 4),
        "portrait_lower_surface_color_cut": round(color_cut, 2),
        "portrait_lower_surface_start_y": int(surface_start),
        "portrait_lower_surface_ratio": round(float(editable.mean()), 5),
        "portrait_lower_surface_alpha_delta_mean": round(changed, 5),
    }


def visual_quality_report(
    image: Image.Image,
    alpha: np.ndarray,
    route_meta: dict,
) -> dict:
    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = alpha.shape
    bg_rgb = route_meta.get("route_bg_rgb") or route_meta.get("cg_bg_rgb")
    report: dict = {
        "alpha_empty_ratio": round(float((alpha < 0.025).mean()), 5),
        "alpha_solid_ratio": round(float((alpha > 0.975).mean()), 5),
        "semi_alpha_ratio": round(float(((alpha > 0.025) & (alpha < 0.975)).mean()), 5),
    }
    if bg_rgb:
        bg = np.asarray(bg_rgb, dtype=np.float32)
        dist = _rgb_distance(img.reshape(-1, 3), bg).reshape(h, w)
        residue = (alpha > 0.025) & (alpha < 0.88) & (dist < 72.0)
        report["bg_colored_residue_ratio"] = round(float(residue.mean()), 5)
    return report


def scrub_transparent_rgb(rgba: Image.Image, alpha_cutoff: int = 2) -> Image.Image:
    arr = np.asarray(rgba.convert("RGBA")).copy()
    transparent = arr[..., 3] <= alpha_cutoff
    arr[transparent, :3] = 0
    arr[transparent, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


def _border_mask(h: int, w: int) -> np.ndarray:
    border_px = max(8, min(h, w) // 24)
    border = np.zeros((h, w), dtype=bool)
    border[:border_px, :] = True
    border[-border_px:, :] = True
    border[:, :border_px] = True
    border[:, -border_px:] = True
    return border


def _neutral_mask(rgb: np.ndarray, tolerance: float) -> np.ndarray:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return (mx - mn) <= tolerance


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _portrait_protect_mask(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    hsv = _rgb_to_hsv(np.clip(rgb / 255.0, 0.0, 1.0))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    luma = _luma(rgb)
    skin_hue = (h < 0.14) | (h > 0.92)
    skin = skin_hue & (s > 0.10) & (s < 0.72) & (v > 0.34) & (luma > 82.0)
    white_cloth = (s < 0.34) & (v > 0.50)
    dark_cloth = (v < 0.18) & (alpha > 0.72)
    protect = (skin | white_cloth | dark_cloth) & (alpha > 0.24)
    return binary_dilation(protect, structure=np.ones((3, 3), dtype=bool), iterations=1)


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
