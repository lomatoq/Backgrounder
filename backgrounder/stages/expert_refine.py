from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

from backgrounder.stages.depth_refine import smooth_alpha_boundary
from backgrounder.utils import guided_filter, rgb_edge_magnitude

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
    Stage D — Route to the appropriate boundary refiner.

    expert = "vitmatte"   → ViTMatte trimap-based refiner (hair, fur, thin)
    expert = "depth_only" → RGB-guided filter  +  conditional depth correction

    Phase 4A redesign:
    - Guided filter (RGB-guided) is now the PRIMARY smoother for all subjects.
      It follows colour edges, so it sharpens at real object boundaries without
      spreading blur across them the way depth-only Gaussian did.
    - Depth edges are applied ONLY where the RGB contrast is too low to be
      relied on (white-on-white, glass, haze).  Everywhere else depth is
      skipped entirely to avoid smearing already-clean edges.
    """
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    unknown = (trimap == 128).astype(np.float32)

    if expert == "vitmatte" and vitmatte is not None:
        alpha = vitmatte.refine(image, alpha, trimap)
        # Light guided-filter pass on top to clean up any residual blur.
        alpha = guided_filter(image_np, alpha, r=5, eps=1e-3)
    else:
        # Primary: edge-preserving guided filter in the unknown band.
        alpha_smooth = guided_filter(image_np, alpha, r=6, eps=8e-4)
        # Only update unknown-band pixels; keep confident fg/bg untouched.
        alpha = np.where(unknown > 0.5, alpha_smooth, alpha)

        # Secondary: depth correction where RGB contrast is too low.
        if depth_edges is not None:
            alpha = _conditional_depth_correct(
                alpha, image_np, depth_edges, unknown
            )

    return smooth_alpha_boundary(alpha, sigma=0.3)


# ── helpers ────────────────────────────────────────────────────────────────

def _conditional_depth_correct(
    alpha: np.ndarray,
    image_np: np.ndarray,
    depth_edges: np.ndarray,
    trimap_unknown: np.ndarray,
    rgb_thresh: float = 0.10,
    depth_weight: float = 0.30,
) -> np.ndarray:
    """
    Pull alpha boundaries toward depth discontinuities only in regions where
    the RGB image offers little contrast (low_rgb_contrast & unknown band).

    For pixels with clear RGB edges the model is already reliable — applying
    depth there would smear sharp cuts into blurry halos.
    """
    rgb_mag = rgb_edge_magnitude(image_np)
    low_rgb = rgb_mag < rgb_thresh
    apply_mask = (trimap_unknown > 0.5) & low_rgb

    if not apply_mask.any():
        return alpha

    # Depth-driven boundary: sigmoid-sharpen depth edges to 0/1.
    depth_boundary = 1.0 / (1.0 + np.exp(-4.0 * (depth_edges - 0.30)))
    snap = np.where(alpha >= 0.5, depth_boundary, 1.0 - depth_boundary)

    blend = apply_mask.astype(np.float32) * depth_edges * depth_weight
    return np.clip(alpha * (1.0 - blend) + snap * blend, 0.0, 1.0).astype(np.float32)
