from __future__ import annotations

import numpy as np

from backgrounder.models.sam3 import _fuse_sam3_mask


def test_interior_speckle_solidified() -> None:
    # Confident SAM body with sub-0.08 salt holes deep inside (the dark-suit
    # failure). After fusion the interior must be opaque, not transparent.
    h = w = 100
    sam_mask = np.zeros((h, w), dtype=bool)
    sam_mask[20:80, 20:80] = True

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[20:80, 20:80] = 0.9
    # scatter near-zero speckle deep inside the body
    for (y, x) in [(50, 50), (45, 60), (60, 40), (55, 55)]:
        alpha[y, x] = 0.0
    alpha[40, 62] = 0.03

    refined, meta = _fuse_sam3_mask(alpha, sam_mask)
    assert meta["sam3_status"] == "applied"
    # All the deep-interior speckle is now opaque, not transparent.
    for (y, x) in [(50, 50), (45, 60), (60, 40), (55, 55), (40, 62)]:
        assert refined[y, x] > 0.9, (y, x, refined[y, x])


def test_background_outside_mask_still_removed() -> None:
    # Semi-transparent pixels outside the SAM mask are cleared (not solidified).
    h = w = 100
    sam_mask = np.zeros((h, w), dtype=bool)
    sam_mask[30:70, 30:70] = True
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[30:70, 30:70] = 0.9
    alpha[5:10, 5:10] = 0.5  # stray semi-transparent blob in the background

    refined, _ = _fuse_sam3_mask(alpha, sam_mask)
    assert refined[7, 7] == 0.0
    assert refined[50, 50] > 0.9  # interior stays solid
