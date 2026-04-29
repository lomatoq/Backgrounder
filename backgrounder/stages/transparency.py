from __future__ import annotations
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from backgrounder.stages.expert_refine import multiscale_guided_filter


def transparency_refine(
    alpha: np.ndarray,
    image: Image.Image,
    uncertainty: np.ndarray,
    depth_edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stage H: glass / transparency post-processing.

    Steps:
    1. Physics-based transmittance solver: estimates per-pixel alpha from
       the compositing equation pixel = alpha·glass + (1-alpha)·background.
       Works best where glass has a distinct color from the background.
    2. Glass-candidate softening via uncertainty × depth-edge mask.
    3. Multi-scale guided-filter smoothing: fine scale preserves the glass
       boundary; coarse scale suppresses noisy speckle in the glass body.
    """
    alpha = alpha.copy()

    # 1. Transmittance solver — physics-based refinement for colored glass.
    alpha = _transmittance_refine(image, alpha)

    # 2. Glass-candidate softening. Limit to the ambiguous semi-transparent band
    # to avoid turning hard chair legs and product rims into grey haze.
    if depth_edges is not None:
        semi = (alpha > 0.08) & (alpha < 0.92)
        glass_mask = semi & (uncertainty > 0.20) & (depth_edges > 0.15)
        if glass_mask.any():
            strength = np.clip(uncertainty * depth_edges * 4.0, 0.0, 0.55)
            alpha[glass_mask] = (
                alpha[glass_mask] * (1.0 - strength[glass_mask])
                + 0.5 * strength[glass_mask]
            )

    # 3. Multi-scale edge-preserving guided-filter smoothing.
    # Gradient-weighted blend of fine→coarse scales simultaneously sharpens the
    # glass outline and suppresses noisy speckle inside the glass body.
    guide = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    smoothed = multiscale_guided_filter(guide, alpha)
    ambiguous = (alpha > 0.03) & (alpha < 0.97)
    alpha[ambiguous] = 0.65 * alpha[ambiguous] + 0.35 * smoothed[ambiguous]

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _transmittance_refine(
    image: Image.Image,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Physics-based alpha estimation for semi-transparent objects.

    The compositing equation for a transparent object over background:

        pixel = alpha * object_color + (1 - alpha) * background_color

    Rearranged per channel:

        alpha_c = (pixel_c - bg_c) / (obj_c - bg_c)

    object_color is estimated spatially: for each pixel we box-average the RGB
    of nearby opaque (alpha > 0.85) neighbours within a 33×33 window.
    background_color is the median of confirmed-background pixels
    (border strip ∪ neural alpha < 0.05).

    The per-channel estimates are combined with nanmedian for robustness;
    channels where |obj_c - bg_c| < 10/255 are excluded (insufficient contrast
    to solve reliably — common for perfectly clear glass on white background,
    which falls back to the neural alpha gracefully).

    A conservative 45 / 55 blend (transmittance / neural) is applied only in
    the uncertain band 0.12 < alpha < 0.88, leaving confirmed opaque and
    transparent pixels exactly as the neural segmenter produced them.
    """
    img = np.array(image.convert("RGB")).astype(np.float32)
    h, w = img.shape[:2]
    border_px = max(8, min(h, w) // 20)

    # Background colour: border pixels ∪ neural-confirmed background
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_px, :] = True
    border_mask[-border_px:, :] = True
    border_mask[:, :border_px] = True
    border_mask[:, -border_px:] = True
    bg_mask = border_mask | (alpha < 0.05)
    bg_pixels = img[bg_mask] if bg_mask.any() else img.reshape(-1, 3)
    bg_color = np.median(bg_pixels, axis=0)  # (3,)

    # Spatially-varying object colour via box-filter of opaque neighbours
    opaque = (alpha > 0.85).astype(np.float32)
    if opaque.sum() < 16:
        return alpha  # not enough anchors for the solve

    obj_color_map = np.zeros_like(img)
    win = 33
    for c in range(3):
        numer = uniform_filter(img[:, :, c] * opaque, size=win)
        denom = uniform_filter(opaque, size=win)
        obj_color_map[:, :, c] = numer / (denom + 1e-6)

    # Per-channel alpha: alpha_c = (pixel - bg) / (obj - bg)
    denom_ch = obj_color_map - bg_color   # (H, W, 3)
    numer_ch = img           - bg_color   # (H, W, 3)

    significant = np.abs(denom_ch) > 10.0
    alpha_ch = np.where(
        significant,
        numer_ch / (denom_ch + np.sign(denom_ch + 1e-9) * 1e-6),
        np.nan,
    )

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        alpha_solved = np.nanmedian(alpha_ch, axis=2).astype(np.float32)

    no_info = np.all(~significant, axis=2)
    alpha_solved = np.where(no_info, alpha, alpha_solved)

    # Conservative blend — only in uncertain zone
    uncertain = (alpha > 0.12) & (alpha < 0.88)
    result = alpha.copy()
    result[uncertain] = np.clip(
        0.45 * alpha_solved[uncertain] + 0.55 * alpha[uncertain],
        0.0, 1.0,
    )
    return result.astype(np.float32)
