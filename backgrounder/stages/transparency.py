from __future__ import annotations
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def transparency_refine(
    alpha: np.ndarray,
    image: Image.Image,
    uncertainty: np.ndarray,
    depth_edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stage H: glass / transparency post-processing.

    Glass and transparent objects are characterised by:
    - High segmenter disagreement (ensemble uncertainty > 0.20)
    - Real depth discontinuity at their boundary (object has depth)
    - Semi-transparent alpha that should be fractional, not binary

    Steps:
    1. Identify glass-candidate pixels: high uncertainty + depth boundary.
    2. Blend alpha toward partial transparency in those regions.
    3. Apply a fast guided filter for edge-preserving alpha smoothing,
       which removes halos and softens glass edges naturally.
    """
    alpha = alpha.copy()

    # 1. Glass-candidate softening
    if depth_edges is not None:
        glass_mask = (uncertainty > 0.20) & (depth_edges > 0.15)
        if glass_mask.any():
            # Blend strength: proportional to uncertainty × depth-edge magnitude
            strength = np.clip(uncertainty * depth_edges * 4.0, 0.0, 0.55)
            # Target: 0.5 (partial transparency) where glass is detected
            alpha = alpha * (1.0 - strength) + 0.5 * strength

    # 2. Edge-preserving guided-filter smoothing
    guide = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    alpha = _guided_filter(guide, alpha, r=8, eps=1e-3)

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _guided_filter(
    guide: np.ndarray,   # H×W×3  float32 [0, 1]
    src: np.ndarray,     # H×W    float32 [0, 1]
    r: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Fast approximate guided filter via box (uniform) filters.
    He et al. "Guided Image Filtering", ECCV 2010.
    """
    def box(x: np.ndarray) -> np.ndarray:
        return uniform_filter(x, size=2 * r + 1)

    # Greyscale guide
    I = guide.mean(axis=2)

    mean_I  = box(I)
    mean_p  = box(src)
    mean_Ip = box(I * src)
    mean_II = box(I * I)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I ** 2

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    return box(a) * I + box(b)
