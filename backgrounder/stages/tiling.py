from __future__ import annotations
from typing import Callable, Tuple

import numpy as np
from PIL import Image


def tile_process(
    image: Image.Image,
    process_fn: Callable[[Image.Image], np.ndarray],
    tile_size: int = 1024,
    overlap: int = 128,
) -> np.ndarray:
    """
    SAHI-style tiling for images larger than tile_size.

    Splits the image into overlapping tiles, processes each independently,
    and merges with Gaussian-feathered blending to hide tile boundaries.

    Returns alpha float32 [0,1], H×W at the original resolution.
    """
    W, H = image.size

    if W <= tile_size and H <= tile_size:
        return process_fn(image)

    alpha_acc = np.zeros((H, W), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)
    feather = _make_feather(tile_size, overlap)

    stride = tile_size - overlap
    xs = list(range(0, W - overlap, stride))
    ys = list(range(0, H - overlap, stride))

    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)
            # Shift origin so tiles never start past the edge.
            x0 = max(0, x1 - tile_size)
            y0 = max(0, y1 - tile_size)

            tile = image.crop((x0, y0, x1, y1))
            tile_alpha = process_fn(tile)

            tw, th = x1 - x0, y1 - y0
            w = feather[:th, :tw]

            alpha_acc[y0:y1, x0:x1] += tile_alpha * w
            weight_acc[y0:y1, x0:x1] += w

    result = alpha_acc / np.maximum(weight_acc, 1e-8)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _make_feather(tile_size: int, overlap: int) -> np.ndarray:
    """
    2-D Gaussian weight mask that falls to ~0 at the tile edges.
    This blends tile boundaries seamlessly.
    """
    from scipy.ndimage import gaussian_filter

    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    # Taper the border region.
    ramp = np.ones(tile_size, dtype=np.float32)
    ramp[:overlap] = np.linspace(0, 1, overlap)
    ramp[-overlap:] = np.linspace(1, 0, overlap)
    mask *= ramp[np.newaxis, :]
    mask *= ramp[:, np.newaxis]
    return gaussian_filter(mask, sigma=overlap / 4)
