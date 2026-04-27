from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def depth_aware_refine(
    alpha: np.ndarray,
    depth_edges: np.ndarray,
    trimap_unknown: np.ndarray,
    sharpness: float = 4.0,
    depth_weight: float = 0.35,
) -> np.ndarray:
    """
    Boundary-selective fusion: in the trimap unknown band, pull alpha boundaries
    toward depth discontinuities.

    Where depth edges are strong (ΔRGB ≈ 0, "white on white"), the depth signal
    is the only reliable boundary cue — so we sharpen alpha transitions there.

    Parameters
    ----------
    alpha         : ensemble alpha float32 [0,1]
    depth_edges   : normalised Sobel depth magnitude [0,1]
    trimap_unknown: 1 where unknown, 0 elsewhere
    sharpness     : sigmoid steepness for depth-driven hard boundary
    depth_weight  : blending weight toward depth boundary in unknown region
    """
    # Depth-driven boundary: sigmoid-sharpen depth edges to produce a crisp 0/1 map.
    depth_boundary = 1.0 / (1.0 + np.exp(-sharpness * (depth_edges - 0.3)))

    # In areas of strong depth discontinuity, snap alpha toward 0 or 1.
    # We use the alpha's own sign to decide which way to snap.
    snap_direction = np.where(alpha >= 0.5, depth_boundary, 1.0 - depth_boundary)

    # Blend only inside the unknown band, proportional to depth edge strength.
    blend_factor = trimap_unknown * depth_edges * depth_weight
    refined = alpha * (1.0 - blend_factor) + snap_direction * blend_factor

    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def smooth_alpha_boundary(alpha: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Light Gaussian smoothing on the alpha boundary to remove jaggedness."""
    uncertain = (alpha > 0.05) & (alpha < 0.95)
    smoothed = gaussian_filter(alpha, sigma=sigma)
    return np.where(uncertain, smoothed, alpha).astype(np.float32)
