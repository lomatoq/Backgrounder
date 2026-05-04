from __future__ import annotations

import numpy as np
from PIL import Image

from backgrounder.stages.router import analyze_image_route


def test_clean_flat_anime_routes_to_chroma_without_cartoon_snap() -> None:
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:] = [50, 85, 239]
    img[28:70, 30:66] = [245, 210, 80]
    alpha = np.zeros((96, 96), dtype=np.float32)
    alpha[26:72, 28:68] = 1.0

    route = analyze_image_route(Image.fromarray(img), "anime", alpha=alpha)

    assert route.route_id == "flat_key_flat_cartoon"
    assert route.clean_border is True
    assert route.image_family == "flat_cartoon"
    assert route.keyable is True
    assert route.use_solid_background_cleanup is True
    assert route.use_cartoon_snap is False
    assert route.use_graphic_alpha_normalize is True
    assert route.allow_sdmatte_auto is False


def test_busy_anime_preserves_model_path() -> None:
    rng = np.random.default_rng(11)
    img = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    img[28:70, 30:66] = [245, 210, 80]

    route = analyze_image_route(Image.fromarray(img), "anime")

    assert route.route_id == "cartoon_busy"
    assert route.clean_border is False
    assert route.use_solid_background_cleanup is False
    assert route.use_cartoon_snap is False
    assert route.use_graphic_alpha_normalize is True


def test_transparent_and_fur_route_to_natural_sdmatte_candidates() -> None:
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:] = [210, 215, 220]
    img[18:80, 28:68] = [130, 90, 50]

    route = analyze_image_route(Image.fromarray(img), "animal_fur")

    assert route.route_id == "natural_fine_edge"
    assert route.material_hint == "fine_edge"
    assert route.use_solid_background_cleanup is False
    assert route.allow_sdmatte_auto is True
