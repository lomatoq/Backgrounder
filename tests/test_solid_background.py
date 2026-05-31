from __future__ import annotations

import numpy as np
from PIL import Image

from backgrounder.stages.solid_background import remove_solid_background_spill


def test_solid_blue_cleanup_preserves_dark_cartoon_outline() -> None:
    bg = [50, 85, 239]
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:] = bg

    # Opaque "bad neural matte" everywhere: the cleanup must identify the
    # connected screen plate by itself.
    alpha = np.ones((96, 96), dtype=np.float32)

    # Foreground block, a dark blue outline, and a bright blue antialias halo.
    img[26:70, 30:70] = [245, 210, 80]
    img[24:72, 24:26] = [28, 38, 90]
    img[24:72, 26:30] = [60, 100, 230]

    cleaned, meta = remove_solid_background_spill(Image.fromarray(img), alpha, "anime")

    assert meta["solid_bg_spill_cleanup"] == "applied"
    assert cleaned[:8, :8].mean() == 0.0
    assert cleaned[24:72, 26:30].mean() == 0.0
    assert cleaned[24:72, 24:26].mean() == 1.0
    assert cleaned[32:64, 36:64].mean() == 1.0


def test_connected_key_does_not_punch_enclosed_blue_foreground_detail() -> None:
    bg = [50, 85, 239]
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:] = bg
    alpha = np.ones((96, 96), dtype=np.float32)

    # A yellow foreground island fully encloses a blue detail with the exact
    # screen color. A global chroma key would punch this out; connected keying
    # must keep it because it is not connected to the border plate.
    img[20:76, 20:76] = [245, 210, 80]
    img[42:54, 42:54] = bg

    cleaned, meta = remove_solid_background_spill(Image.fromarray(img), alpha, "anime")

    assert meta["solid_bg_spill_cleanup"] == "applied"
    assert cleaned[:8, :8].mean() == 0.0
    assert cleaned[42:54, 42:54].mean() == 1.0
    assert cleaned[24:72, 24:72].mean() > 0.95


def test_large_enclosed_screen_hole_is_removed_for_flat_cartoon() -> None:
    bg = [50, 85, 239]
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:] = bg
    alpha = np.ones((128, 128), dtype=np.float32)

    img[22:106, 22:106] = [245, 210, 80]
    img[48:82, 52:78] = bg

    cleaned, meta = remove_solid_background_spill(Image.fromarray(img), alpha, "flat_cartoon")

    assert meta["solid_bg_spill_cleanup"] == "applied"
    assert meta["solid_bg_enclosed_hole_cleanup"] == "applied"
    assert cleaned[:8, :8].mean() == 0.0
    assert cleaned[52:78, 56:74].mean() == 0.0
    assert cleaned[28:44, 30:98].mean() == 1.0


def test_busy_border_skips_solid_background_cleanup() -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    alpha = np.ones((96, 96), dtype=np.float32)

    cleaned, meta = remove_solid_background_spill(Image.fromarray(img), alpha, "product")

    assert meta["solid_bg_spill_cleanup"] == "skipped_busy_border"
    np.testing.assert_array_equal(cleaned, alpha)
