from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from backgrounder.stages.depth_refine import smooth_alpha_boundary

if TYPE_CHECKING:
    from backgrounder.models.vitmatte import ViTMatteRefiner


def expert_refine(
    image: Image.Image,
    alpha: np.ndarray,
    trimap: np.ndarray,
    expert: str,
    depth_edges: Optional[np.ndarray],
    vitmatte: Optional["ViTMatteRefiner"] = None,
    use_closed_form: bool = True,
    closed_form_max_pixels: int = 65_536,
) -> np.ndarray:
    """
    Route to the appropriate Stage-D refiner based on subject type.

    expert = "vitmatte"   → ViTMatte (hair/fur); fallback: medium guided filter
    expert = "depth_only" → tight guided filter for crisp hard edges
    expert = "color_key"  → background-color-distance extraction (text/logos)
    """
    unknown = (trimap == 128).astype(np.float32)
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    if expert == "color_key":
        # Background-color-keying: best for text/logos on near-solid backgrounds.
        # The neural coarse alpha is used only to identify which pixels are
        # definitely background — then we re-extract alpha from color distance.
        alpha_ck = _color_key_extract(image, alpha)
        # Blend: 75% color-key, 25% neural. Keeps correct mask shape on images
        # where the background is not perfectly uniform.
        alpha = 0.75 * alpha_ck + 0.25 * alpha
        # Guided filter to anti-alias text/logo edges.
        alpha_gf = _guided_filter(image_np, alpha, r=2, eps=5e-5)
        alpha = np.where((alpha > 0.03) & (alpha < 0.97), alpha_gf, alpha)
        # Snap to binary — text edges should be crisp.
        alpha = np.where(alpha > 0.88, 1.0, alpha)
        alpha = np.where(alpha < 0.12, 0.0, alpha)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    elif expert == "vitmatte" and vitmatte is not None:
        alpha = vitmatte.refine(image, alpha, trimap)
        # Light guided-filter pass to clean residual blur at hair boundary.
        alpha_gf = _guided_filter(image_np, alpha, r=4, eps=1e-3)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)
        return smooth_alpha_boundary(alpha, sigma=0.5)

    elif expert == "vitmatte":
        # ViTMatte not loaded — guided filter is the next best for hair/fur.
        alpha_gf = _guided_filter(image_np, alpha, r=6, eps=5e-4)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)
        return smooth_alpha_boundary(alpha, sigma=0.5)

    else:
        # depth_only — hard-edged objects (products, vehicles, generic).
        # use_closed_form=True → tighter params (sharper); False → relaxed.
        r   = 3   if use_closed_form else 6
        eps = 1e-4 if use_closed_form else 8e-4
        alpha_gf = _guided_filter(image_np, alpha, r=r, eps=eps)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)

        # Snap nearly-solid pixels to binary: removes semi-transparent fringe
        # that should be fully FG or BG on hard-edged objects.
        alpha = np.where((alpha > 0.92) & (unknown > 0.5), 1.0, alpha)
        alpha = np.where((alpha < 0.08) & (unknown > 0.5), 0.0, alpha)

        # No Gaussian smoothing for hard edges — preserves crisp product boundaries.
        return smooth_alpha_boundary(alpha, sigma=0.0)


def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    r: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    def box(x: np.ndarray) -> np.ndarray:
        return uniform_filter(x.astype(np.float64), size=2 * r + 1)

    I = guide.mean(axis=2)
    mean_I  = box(I)
    mean_p  = box(src)
    mean_Ip = box(I * src)
    mean_II = box(I * I)
    cov_Ip  = mean_Ip - mean_I * mean_p
    var_I   = mean_II - mean_I ** 2
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return np.clip(box(a) * I + box(b), 0.0, 1.0).astype(np.float32)


def _color_key_extract(
    image: Image.Image,
    alpha_coarse: np.ndarray,
) -> np.ndarray:
    """
    Background-color-keying for text / logos on near-solid backgrounds.

    Algorithm:
      1. Build a background pixel set: border strip + pixels where coarse
         alpha < 0.05 (both confirmed background by the neural segmenter).
      2. Estimate background color as the median of those pixels.
      3. Compute per-pixel Euclidean RGB distance from the background color.
      4. Find a soft threshold: pixels within the 90th-percentile distance of
         confirmed background → transparent; beyond → opaque.
      5. Return a smooth [0, 1] alpha via a linear ramp around the threshold.

    This is dramatically more accurate than neural segmenters for the case of
    high-contrast text / logos on flat-color or near-flat backgrounds.
    """
    img = np.array(image.convert("RGB")).astype(np.float32)
    h, w = img.shape[:2]
    border_px = max(8, min(h, w) // 20)

    # Background pixel set
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_px, :] = True
    border_mask[-border_px:, :] = True
    border_mask[:, :border_px] = True
    border_mask[:, -border_px:] = True
    bg_mask = border_mask | (alpha_coarse < 0.05)

    bg_pixels = img[bg_mask] if bg_mask.any() else img.reshape(-1, 3)
    bg_color = np.median(bg_pixels, axis=0)

    # Per-pixel RGB distance from background
    dist = np.sqrt(((img - bg_color) ** 2).sum(axis=2))

    # Threshold from confirmed-background pixel distances
    bg_dist = dist[bg_mask]
    # 90th percentile of bg distances = where background "ends"
    bg_threshold = float(np.percentile(bg_dist, 90)) if len(bg_dist) > 10 else 20.0
    bg_threshold = max(bg_threshold, 12.0)  # floor to avoid near-zero on perfect solid bg

    # Ramp: 0 at threshold, 1 at 3× threshold
    spread = max(bg_threshold * 2.0, 20.0)
    alpha = np.clip((dist - bg_threshold) / (spread + 1e-6), 0.0, 1.0)
    return alpha.astype(np.float32)
