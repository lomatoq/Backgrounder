from __future__ import annotations

import numpy as np
from PIL import Image

from backgrounder.stages.depth_refine import closed_form_matting_refine


def _flat_scene():
    """Flat low-contrast image; a centre block is 'unknown' at coarse alpha 0.7."""
    img = Image.fromarray(np.full((64, 64, 3), 80, dtype=np.uint8), mode="RGB")
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 0.7
    trimap = np.zeros((64, 64), dtype=np.uint8)       # 0 = definite background
    trimap[16:48, 16:48] = 128                         # unknown block
    return img, alpha, trimap


def test_edge_gate_keeps_flat_low_contrast_region_from_collapsing() -> None:
    img, alpha, trimap = _flat_scene()
    gated = closed_form_matting_refine(img, alpha, trimap, edge_gate=True)
    # No colour edge → matting has no information → coarse alpha (0.7) preserved
    # instead of collapsing toward 0 (the dark-suit speckle failure).
    assert gated[32, 32] > 0.6


def test_definite_regions_stay_anchored() -> None:
    img, alpha, trimap = _flat_scene()
    out = closed_form_matting_refine(img, alpha, trimap, edge_gate=True)
    # Definite-background pixels (trimap 0) are never edited.
    assert out[0, 0] == 0.0
