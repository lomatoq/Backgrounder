from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

from backgrounder.stages.depth_refine import depth_aware_refine, smooth_alpha_boundary

if TYPE_CHECKING:
    from backgrounder.models.vitmatte import ViTMatteRefiner


def expert_refine(
    image: Image.Image,
    alpha: np.ndarray,
    trimap: np.ndarray,
    expert: str,
    depth_edges: Optional[np.ndarray],
    vitmatte: Optional["ViTMatteRefiner"] = None,
) -> np.ndarray:
    """
    Route to the appropriate Stage-D expert based on subject type.

    expert = "vitmatte"   → ViTMatte trimap-based refiner (hair, fur, thin structures)
    expert = "depth_only" → depth-aware boundary fusion (low-contrast, product, glass)

    Falls back to depth_only gracefully if vitmatte is not loaded.
    """
    unknown = (trimap == 128).astype(np.float32)

    if expert == "vitmatte" and vitmatte is not None:
        alpha = vitmatte.refine(image, alpha, trimap)
        # Additionally apply light depth refinement on top for low-contrast edges.
        if depth_edges is not None:
            alpha = depth_aware_refine(alpha, depth_edges, unknown, depth_weight=0.15)
    else:
        # depth_only path
        if depth_edges is not None:
            alpha = depth_aware_refine(alpha, depth_edges, unknown)

    return smooth_alpha_boundary(alpha)
