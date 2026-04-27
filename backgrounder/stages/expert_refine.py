from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

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

    expert = "vitmatte"   → ViTMatte trimap-based refiner (hair, fur, thin structures)
    expert = "depth_only" → closed-form matting in the trimap unknown band

    Falls back to depth_only gracefully if vitmatte is not loaded.
    """
    if expert == "vitmatte" and vitmatte is not None:
        alpha = vitmatte.refine(image, alpha, trimap)
    elif use_closed_form:
        alpha = closed_form_matting_refine(
            image=image,
            alpha=alpha,
            trimap=trimap,
            max_solve_pixels=closed_form_max_pixels,
        )

    return smooth_alpha_boundary(alpha)
