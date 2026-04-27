from __future__ import annotations
from typing import List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from backgrounder.result import SegmentationOutput


def generate_trimap(
    alpha: np.ndarray,
    dilation: int = 10,
    confidence: Optional[np.ndarray] = None,
    depth_edges: Optional[np.ndarray] = None,
    depth_edge_expand: int = 5,
) -> np.ndarray:
    """
    Trimap: 255 = definite foreground, 0 = definite background, 128 = unknown.

    confidence    : BEN2 confidence map [0,1]; low-confidence pixels → unknown.
    depth_edges   : normalised Sobel depth magnitude; strong edges expand the band.
    """
    struct = np.ones((2 * dilation + 1, 2 * dilation + 1), dtype=bool)

    fg = alpha > 0.9
    certain_fg = binary_erosion(fg, structure=struct)
    uncertain_boundary = binary_dilation(fg, structure=struct)

    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap[uncertain_boundary] = 128
    trimap[certain_fg] = 255

    # Widen unknown band in high-uncertainty regions from BEN2 confidence.
    if confidence is not None:
        low_conf = confidence < 0.4
        small_struct = np.ones((2 * (dilation // 2) + 1, 2 * (dilation // 2) + 1), dtype=bool)
        extra_unknown = binary_dilation(low_conf, structure=small_struct)
        # Only open up pixels that were already definite — don't shrink unknowns.
        trimap[extra_unknown & (trimap == 255)] = 128
        trimap[extra_unknown & (trimap == 0) & uncertain_boundary] = 128

    # Expand the unknown band at depth discontinuities to catch low-contrast edges.
    if depth_edges is not None:
        strong_depth = depth_edges > 0.25
        expand_struct = np.ones((2 * depth_edge_expand + 1, 2 * depth_edge_expand + 1), dtype=bool)
        depth_zone = binary_dilation(strong_depth, structure=expand_struct)
        trimap[depth_zone & (trimap != 128)] = 128

    return trimap


def unknown_mask(trimap: np.ndarray) -> np.ndarray:
    return (trimap == 128).astype(np.float32)
