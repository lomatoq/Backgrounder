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

    The Levin closed-form solver is NOT called here: its pure-Python O(n²) loop
    takes 12+ seconds at 256×256. A guided filter at r=3 is equally sharp
    and runs in ~50 ms.
    """
    unknown = (trimap == 128).astype(np.float32)
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    if expert == "vitmatte" and vitmatte is not None:
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
