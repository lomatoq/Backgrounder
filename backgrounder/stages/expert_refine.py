from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from backgrounder.stages.depth_refine import closed_form_matting_refine, smooth_alpha_boundary

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
    Route to the appropriate Stage-D expert based on subject type.

    expert = "vitmatte"   → ViTMatte (hair/fur); fallback: guided filter
    expert = "depth_only" → closed-form matting OR guided filter
    """
    unknown = (trimap == 128).astype(np.float32)
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    if expert == "vitmatte" and vitmatte is not None:
        alpha = vitmatte.refine(image, alpha, trimap)
        # Light guided-filter pass to clean residual blur.
        alpha_gf = _guided_filter(image_np, alpha, r=4, eps=1e-3)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)
    elif expert == "vitmatte":
        # ViTMatte not loaded — guided filter is the next best thing for hair.
        alpha_gf = _guided_filter(image_np, alpha, r=6, eps=5e-4)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)
    elif use_closed_form:
        alpha = closed_form_matting_refine(
            image=image,
            alpha=alpha,
            trimap=trimap,
            max_solve_pixels=closed_form_max_pixels,
        )
    else:
        alpha_gf = _guided_filter(image_np, alpha, r=6, eps=8e-4)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)

    return smooth_alpha_boundary(alpha)


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
