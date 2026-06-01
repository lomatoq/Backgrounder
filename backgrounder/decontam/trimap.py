"""
Adaptive-width trimap (spec §2.4).

A fixed dilation kernel is the classic cascade killer: too narrow and wispy hair
falls outside the unknown band (matting can never recover it); too wide and the
band swallows solid background, which the refiner then has to hallucinate through.

The band width must follow the local difficulty: wide where the edge is soft or
high-frequency (hair, glass, text strokes), narrow on clean hard edges.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

__all__ = ["adaptive_trimap"]


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def adaptive_trimap(
    alpha: np.ndarray,
    *,
    base_dilation: int = 10,
    min_dilation: int = 3,
    max_dilation: int = 32,
    difficulty: Optional[np.ndarray] = None,
    fg_threshold: float = 0.9,
    transition_eps: float = 0.05,
) -> np.ndarray:
    """
    Trimap with a spatially-varying unknown band.

    Encoding: 255 = definite foreground, 0 = definite background, 128 = unknown.

    Parameters
    ----------
    difficulty :
        Optional per-pixel difficulty in [0, 1] (e.g. a failure-map channel such
        as transition mass or stroke energy — spec §2.1). Where it is high the
        band is dilated out to ``max_dilation``; where it is low it stays at
        ``min_dilation``. When omitted, an intrinsic transition-mass estimate from
        α is used so the function is still adaptive on its own.

    The band is the union of a narrow base band everywhere and a wide band
    restricted to high-difficulty zones — that gives wide coverage exactly around
    hair/glass/text while keeping clean edges tight.
    """
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    fg = a > fg_threshold
    if not fg.any():
        # Nothing confidently foreground — mark the whole soft region unknown.
        trimap = np.zeros(a.shape, dtype=np.uint8)
        trimap[a > transition_eps] = 128
        return trimap

    if difficulty is None:
        difficulty = _intrinsic_difficulty(a, transition_eps)
    difficulty = np.clip(difficulty.astype(np.float32), 0.0, 1.0)

    narrow = max(1, min(min_dilation, base_dilation))
    wide = max(narrow + 1, max_dilation)

    certain_fg = binary_erosion(fg, structure=_disk(narrow))
    base_band = binary_dilation(fg, structure=_disk(base_dilation))
    wide_band = binary_dilation(fg, structure=_disk(wide))

    # High-difficulty zone: dilate the difficulty mask itself so the wide band is
    # only opened where strands/strokes actually live.
    hard_zone = binary_dilation(difficulty > 0.35, structure=_disk(base_dilation))

    unknown = base_band | (wide_band & hard_zone)

    trimap = np.zeros(a.shape, dtype=np.uint8)
    trimap[unknown] = 128
    trimap[certain_fg] = 255
    return trimap


def _intrinsic_difficulty(alpha: np.ndarray, eps: float) -> np.ndarray:
    """Transition mass: soft (partial-alpha) regions are intrinsically hard."""
    transition = (alpha > eps) & (alpha < 1.0 - eps)
    # Smear so isolated soft pixels form a coherent difficulty zone.
    return binary_dilation(transition, structure=_disk(2)).astype(np.float32)
