from __future__ import annotations
from typing import Optional

import numpy as np
from PIL import Image
from backgrounder.utils import guided_filter


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
    alpha = guided_filter(guide, alpha, r=8, eps=1e-3)

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)
