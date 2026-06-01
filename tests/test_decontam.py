from __future__ import annotations

import numpy as np

from backgrounder.decontam import adaptive_trimap, unmix_foreground


def _composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Build the observed image I = α·F + (1−α)·B (uint8)."""
    a = alpha[..., None].astype(np.float32)
    f = np.asarray(fg, dtype=np.float32).reshape(1, 1, 3)
    b = np.asarray(bg, dtype=np.float32).reshape(1, 1, 3)
    img = a * f + (1.0 - a) * b
    return np.clip(img, 0, 255).astype(np.uint8)


def test_blue_halo_removed_recovers_foreground_color() -> None:
    # Red foreground composited over a blue screen with a soft alpha ramp — the
    # classic semi-transparent blue rim. Unmixing must recover red, not blue.
    fg_color = [220, 40, 40]
    bg_color = [40, 80, 230]
    h, w = 32, 64
    alpha = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    img = _composite(fg_color, bg_color, alpha)

    fg = unmix_foreground(img, alpha, bg_rgb=bg_color).astype(np.float32)

    # Mid-band pixel (α≈0.5) where the blue rim is worst.
    mid = fg[:, w // 2].mean(axis=0)
    assert abs(mid[0] - fg_color[0]) < 8, mid
    assert abs(mid[2] - fg_color[2]) < 8, mid
    # Blue must no longer dominate the recovered edge colour.
    assert mid[0] > mid[2] + 80, mid


def test_dark_outline_preserved() -> None:
    # An opaque dark outline must survive untouched — unmixing only acts on the
    # partial-coverage band, never on confident foreground.
    bg_color = [40, 80, 230]
    outline = [18, 18, 24]
    img = np.empty((24, 24, 3), dtype=np.uint8)
    img[:] = outline
    alpha = np.ones((24, 24), dtype=np.float32)

    fg = unmix_foreground(img, alpha, bg_rgb=bg_color).astype(np.float32)
    assert np.allclose(fg, np.array(outline, dtype=np.float32), atol=2), fg.mean(axis=(0, 1))


def test_enclosed_blue_foreground_preserved() -> None:
    # A genuinely blue, opaque foreground region (e.g. a logo detail the same hue
    # as the screen). Despill must not eat it because it is fully opaque.
    bg_color = [40, 80, 230]
    blue_fg = [30, 70, 220]
    img = np.empty((24, 24, 3), dtype=np.uint8)
    img[:] = blue_fg
    alpha = np.ones((24, 24), dtype=np.float32)

    fg = unmix_foreground(img, alpha, bg_rgb=bg_color).astype(np.float32)
    mean = fg.mean(axis=(0, 1))
    assert mean[2] > 180, mean  # blue channel preserved


def test_adaptive_trimap_widens_band_in_hard_regions() -> None:
    # Hard-edged foreground square. With high difficulty on the left half the
    # unknown band there must be wider than on the low-difficulty right half.
    h, w = 80, 80
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[20:60, 20:60] = 1.0

    difficulty = np.zeros((h, w), dtype=np.float32)
    difficulty[:, : w // 2] = 1.0  # left half is "hard"

    trimap = adaptive_trimap(
        alpha, base_dilation=4, min_dilation=2, max_dilation=20, difficulty=difficulty
    )

    row = 40  # crosses the square
    unknown = trimap[row] == 128
    left_band = unknown[:20].sum()    # band outside the left edge (x<20)
    right_band = unknown[60:].sum()   # band outside the right edge (x>=60)
    assert left_band > right_band + 4, (left_band, right_band)


def test_unmix_without_known_plate_estimates_background() -> None:
    # No bg_rgb supplied: the plate is estimated from α<0.05 pixels.
    fg_color = [200, 60, 50]
    bg_color = [50, 90, 220]
    h, w = 32, 64
    alpha = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    img = _composite(fg_color, bg_color, alpha)

    fg = unmix_foreground(img, alpha).astype(np.float32)
    mid = fg[:, w // 2].mean(axis=0)
    assert mid[0] > mid[2] + 60, mid
