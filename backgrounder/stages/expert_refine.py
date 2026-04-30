from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, sobel

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
    expert = "depth_only" → multi-scale guided filter for crisp hard edges
    expert = "color_key"  → background-color-distance extraction (text/logos)
    """
    unknown = (trimap == 128).astype(np.float32)
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    if expert == "color_key":
        alpha_ck = _color_key_extract(image, alpha)
        alpha = 0.75 * alpha_ck + 0.25 * alpha
        # Fine-scale guided filter for text edges — don't blend scales here,
        # text edges need maximum sharpness.
        alpha_gf = _guided_filter(image_np, alpha, r=2, eps=5e-5)
        alpha = np.where((alpha > 0.03) & (alpha < 0.97), alpha_gf, alpha)
        alpha = np.where(alpha > 0.88, 1.0, alpha)
        alpha = np.where(alpha < 0.12, 0.0, alpha)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    elif expert == "vitmatte" and vitmatte is not None:
        alpha = vitmatte.refine(image, alpha, trimap)
        # Standard guided filter after ViTMatte — ViTMatte already handles
        # fine strand detail so a medium-scale pass is sufficient.
        alpha_gf = _guided_filter(image_np, alpha, r=4, eps=1e-3)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)
        return smooth_alpha_boundary(alpha, sigma=0.5)

    elif expert == "vitmatte":
        # ViTMatte not loaded: multi-scale guided filter as best alternative.
        alpha_gf = multiscale_guided_filter(image_np, alpha)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)
        return smooth_alpha_boundary(alpha, sigma=0.5)

    else:
        # depth_only — hard-edged objects (products, vehicles, generic).
        # Multi-scale guided filter: fine scale for crisp product edges,
        # coarse scale for smooth object bodies without halos.
        alpha_gf = multiscale_guided_filter(image_np, alpha)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)

        alpha = np.where((alpha > 0.92) & (unknown > 0.5), 1.0, alpha)
        alpha = np.where((alpha < 0.08) & (unknown > 0.5), 0.0, alpha)
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


def multiscale_guided_filter(
    guide: np.ndarray,  # H×W×3 float32 [0,1]
    src: np.ndarray,    # H×W   float32 [0,1]
) -> np.ndarray:
    """
    Gradient-weighted multi-scale guided filter fusion.

    Three guided filters at different radii are merged per-pixel using the
    IMAGE gradient (not alpha gradient) to select scale:

      fine   (r=2,  eps=1e-5) — dominates at real object boundaries
      medium (r=6,  eps=5e-4) — semi-transparent / soft-shadow regions
      coarse (r=12, eps=3e-3) — smooth glass bodies, uniform backgrounds

    Scale selection is driven by the GUIDE (image) gradient, not the alpha.
    The alpha can be noisy everywhere, which would make the alpha gradient
    high everywhere and force fine-scale everywhere (defeating the purpose).
    The image has real structure only at true object edges, so the image
    gradient correctly identifies where fine-scale sharpness is needed.

    Blend weights (quadratic, sum to 1 exactly):
        w_fine   = g²        → 1 at crisp image edges, 0 in smooth areas
        w_coarse = (1-g)²    → 1 in smooth areas, 0 at edges
        w_medium = 2g(1-g)   → peaks at g=0.5
    """
    fine   = _guided_filter(guide, src, r=2,  eps=1e-5)
    medium = _guided_filter(guide, src, r=6,  eps=5e-4)
    coarse = _guided_filter(guide, src, r=12, eps=3e-3)

    # Gradient from the IMAGE (guide), not from the alpha.
    I = guide.mean(axis=2).astype(np.float64)
    gx = np.abs(sobel(I, axis=1))
    gy = np.abs(sobel(I, axis=0))
    g = np.sqrt(gx ** 2 + gy ** 2)
    g = (g / (g.max() + 1e-8)).astype(np.float32)

    w_fine   = g * g
    w_coarse = (1.0 - g) * (1.0 - g)
    w_medium = 2.0 * g * (1.0 - g)  # = 1 - w_fine - w_coarse

    return np.clip(
        w_fine * fine + w_medium * medium + w_coarse * coarse,
        0.0, 1.0,
    ).astype(np.float32)


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
