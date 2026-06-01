from __future__ import annotations

import numpy as np
from PIL import Image

from backgrounder.analyze import FAILURE_CHANNELS, compute_failure_map


def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def test_channel_order_and_shape() -> None:
    img = _img(np.full((40, 40, 3), 127))
    alpha = np.zeros((40, 40), dtype=np.float32)
    fm = compute_failure_map(img, alpha)
    assert fm.stack.shape == (40, 40, 5)
    assert FAILURE_CHANNELS == ("U_dis", "U_trans", "U_tta", "U_flat", "U_text")
    assert fm.stack.min() >= 0.0 and fm.stack.max() <= 1.0


def test_u_dis_high_where_models_disagree() -> None:
    img = _img(np.full((40, 40, 3), 127))
    a1 = np.zeros((40, 40), dtype=np.float32)
    a2 = np.zeros((40, 40), dtype=np.float32)
    a2[:, 20:] = 1.0  # second model claims the right half is foreground
    fm = compute_failure_map(img, (a1 + a2) / 2, alpha_per_model=[a1, a2])
    u_dis = fm.channel("U_dis")
    assert u_dis[:, 25:].mean() > 0.8
    assert u_dis[:, :15].mean() < 0.1


def test_u_trans_high_in_soft_edge_band() -> None:
    img = _img(np.full((40, 64, 3), 127))
    alpha = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :].repeat(40, axis=0)
    fm = compute_failure_map(img, alpha)
    u_trans = fm.channel("U_trans")
    # Mid ramp is partial-alpha (soft); the fully opaque/transparent ends are not.
    assert u_trans[:, 28:36].mean() > 0.7
    assert u_trans[:, :3].mean() < 0.3
    assert u_trans[:, -3:].mean() < 0.3


def test_u_tta_reflects_variance() -> None:
    img = _img(np.full((32, 32, 3), 127))
    alpha = np.zeros((32, 32), dtype=np.float32)
    var = np.zeros((32, 32), dtype=np.float32)
    var[10:20, 10:20] = 0.25  # std 0.5 → scaled to 1.0
    fm = compute_failure_map(img, alpha, tta_variance=var)
    u = fm.channel("U_tta")
    assert u[15, 15] > 0.8
    assert u[0, 0] < 0.05


def test_u_flat_high_on_border_connected_plate() -> None:
    bg = [40, 80, 230]
    img = np.empty((64, 64, 3), dtype=np.uint8)
    img[:] = bg
    img[24:40, 24:40] = [220, 60, 60]  # foreground block, not plate-coloured
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[24:40, 24:40] = 1.0
    fm = compute_failure_map(_img(img), alpha, bg_rgb=bg)
    u_flat = fm.channel("U_flat")
    assert u_flat[:6, :6].mean() > 0.6      # flat border plate
    assert u_flat[28:36, 28:36].mean() < 0.3  # textured-coloured foreground


def test_u_text_high_on_thin_strokes() -> None:
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    # Thin black strokes (a crude glyph): high-frequency, thin structures.
    img[10:54, 20:23] = 0
    img[10:13, 20:44] = 0
    alpha = np.zeros((64, 64), dtype=np.float32)
    fm = compute_failure_map(_img(img), alpha)
    u_text = fm.channel("U_text")
    flat_region = _img(np.full((64, 64, 3), 200))
    fm_flat = compute_failure_map(flat_region, alpha)
    assert u_text.max() > 0.4
    assert u_text.mean() > fm_flat.channel("U_text").mean() + 0.02


def test_region_labels_drop_speckle_and_mark_dominant() -> None:
    img = _img(np.full((64, 64, 3), 127))
    a1 = np.zeros((64, 64), dtype=np.float32)
    a2 = np.zeros((64, 64), dtype=np.float32)
    a2[16:48, 16:48] = 1.0  # large disagreement block → a real region

    # A tiny TTA-variance speckle that is the dominant channel locally but is far
    # below min_region_area → must be dropped back to "no region".
    var = np.zeros((64, 64), dtype=np.float32)
    var[2:4, 2:4] = 0.25

    fm = compute_failure_map(
        img, (a1 + a2) / 2, alpha_per_model=[a1, a2], tta_variance=var, min_region_area=64
    )
    dis_id = FAILURE_CHANNELS.index("U_dis") + 1
    assert (fm.region_labels == dis_id).sum() > 100   # big region kept
    assert fm.region_labels[2:4, 2:4].max() == 0      # speckle dropped
