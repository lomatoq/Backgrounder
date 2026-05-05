from __future__ import annotations

import numpy as np
from PIL import Image

from backgrounder.stages.visual_qa import (
    remove_checkerboard_background,
    remove_graphic_border_residue,
    remove_portrait_lower_surface,
    scrub_transparent_rgb,
)


def test_checkerboard_background_is_removed_when_model_keeps_it() -> None:
    h = w = 80
    yy, xx = np.indices((h, w))
    checker = ((xx // 8) + (yy // 8)) % 2 == 0
    img = np.where(checker[..., None], 222, 184).astype(np.uint8)
    img = np.broadcast_to(img, (h, w, 3)).copy()
    img[28:52, 24:56] = [235, 210, 70]

    alpha = np.ones((h, w), dtype=np.float32)
    fixed, meta = remove_checkerboard_background(Image.fromarray(img), alpha, "text_logo")

    assert meta["checkerboard_cleanup"] == "applied"
    assert fixed[:8, :8].mean() == 0.0
    assert fixed[-8:, -8:].mean() == 0.0
    assert fixed[32:48, 30:50].mean() == 1.0


def test_checkerboard_text_cleanup_removes_internal_holes_without_boosting_edges() -> None:
    h = w = 80
    yy, xx = np.indices((h, w))
    checker = ((xx // 8) + (yy // 8)) % 2 == 0
    dark = np.array([35, 31, 22], dtype=np.uint8)
    light = np.array([99, 92, 82], dtype=np.uint8)
    img = np.where(checker[..., None], light, dark).astype(np.uint8)
    img[28:52, 18:62] = [238, 215, 142]
    img[36:44, 36:44] = np.where(checker[36:44, 36:44, None], light, dark).astype(np.uint8)

    alpha = np.full((h, w), 0.12, dtype=np.float32)
    alpha[28:52, 18:62] = 0.35
    alpha[36:44, 36:44] = 0.72
    fixed, meta = remove_checkerboard_background(Image.fromarray(img), alpha, "text_glow")

    assert meta["checkerboard_cleanup"] == "applied"
    assert fixed[:8, :8].mean() == 0.0
    assert fixed[36:44, 36:44].mean() == 0.0
    assert fixed[30:34, 24:32].mean() == 0.35


def test_graphic_border_residue_removes_connected_background_color_halo() -> None:
    bg = [60, 42, 118]
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[:] = bg
    alpha = np.zeros((80, 80), dtype=np.float32)

    img[22:58, 22:58] = [88, 210, 74]
    alpha[22:58, 22:58] = 1.0

    img[18:62, 18:22] = [82, 54, 146]
    img[18:62, 58:62] = [82, 54, 146]
    img[18:22, 18:62] = [82, 54, 146]
    img[58:62, 18:62] = [82, 54, 146]
    alpha[18:62, 18:62] = np.maximum(alpha[18:62, 18:62], 0.44)

    fixed, meta = remove_graphic_border_residue(
        image=Image.fromarray(img),
        alpha=alpha,
        subject_type="sticker_logo",
        route_meta={"route_bg_rgb": bg},
    )

    assert meta["graphic_border_residue_cleanup"] == "applied"
    assert fixed[18:22, 30:50].mean() < 0.22
    assert fixed[30:50, 30:50].mean() == 1.0


def test_scrub_transparent_rgb_clears_hidden_background_color() -> None:
    rgba = np.zeros((12, 12, 4), dtype=np.uint8)
    rgba[..., :3] = [50, 85, 239]
    rgba[..., 3] = 0
    rgba[4:8, 4:8] = [240, 220, 90, 255]

    scrubbed = np.asarray(scrub_transparent_rgb(Image.fromarray(rgba, mode="RGBA")))

    assert scrubbed[0, 0, :4].tolist() == [0, 0, 0, 0]
    assert scrubbed[5, 5, :4].tolist() == [240, 220, 90, 255]


def test_portrait_lower_surface_removes_wide_table_without_hands() -> None:
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:] = [35, 35, 40]
    alpha = np.zeros((96, 96), dtype=np.float32)

    # Person torso/head.
    img[16:66, 36:60] = [230, 210, 180]
    img[40:78, 30:66] = [18, 18, 20]
    alpha[16:78, 30:66] = 1.0

    # Wide red table/prop connected to the bottom.
    img[66:96, :] = [160, 35, 30]
    alpha[66:96, :] = 1.0

    # Hands should survive.
    img[62:76, 14:28] = [225, 178, 145]
    img[62:76, 68:82] = [225, 178, 145]
    alpha[62:76, 14:28] = 1.0
    alpha[62:76, 68:82] = 1.0

    fixed, meta = remove_portrait_lower_surface(Image.fromarray(img), alpha, "portrait")

    assert meta["portrait_lower_surface_cleanup"] == "applied"
    assert fixed[84:94, 10:86].mean() == 0.0
    assert fixed[64:72, 16:26].mean() == 1.0
    assert fixed[22:60, 38:58].mean() == 1.0
