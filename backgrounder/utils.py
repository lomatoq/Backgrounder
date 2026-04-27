from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import sobel


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str, fp16: bool) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") and fp16 else torch.float32


def pil_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def alpha_to_pil_mask(alpha: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8), mode="L")


def resize_alpha_to(alpha: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    pil = alpha_to_pil_mask(alpha).resize(target_wh, Image.LANCZOS)
    return np.array(pil).astype(np.float32) / 255.0


def compute_depth_edges(depth: np.ndarray) -> np.ndarray:
    """Return normalised Sobel-magnitude edge map from a monocular depth array."""
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    gx = sobel(d, axis=1)
    gy = sobel(d, axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag / (mag.max() + 1e-8)


def estimate_foreground(image_rgb: np.ndarray, alpha: np.ndarray, sigma: int = 20) -> np.ndarray:
    """
    Decontaminate foreground by removing background colour bleed at boundaries.

    Uses alpha-weighted Gaussian pooling: each boundary pixel's colour is replaced
    by a local weighted average that strongly favours definitely-foreground pixels,
    so background tinting is stripped out.
    """
    from scipy.ndimage import gaussian_filter

    fg = image_rgb.astype(np.float32)
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    w = a ** 2  # weight: 1 at fg=1, 0 at fg=0

    for c in range(fg.shape[2]):
        numerator = gaussian_filter(fg[:, :, c] * w, sigma=sigma)
        denominator = gaussian_filter(w, sigma=sigma) + 1e-8
        fg_est = numerator / denominator
        # Only replace colour for boundary pixels; definite fg keeps original.
        fg[:, :, c] = np.where(a > 0.95, fg[:, :, c], fg_est)

    return np.clip(fg, 0, 255).astype(np.uint8)


def compose_rgba(foreground: np.ndarray, alpha: np.ndarray) -> Image.Image:
    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([foreground, a8])
    return Image.fromarray(rgba, mode="RGBA")


@contextmanager
def timer(label: str, store: dict | None = None) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if store is not None:
        store[label] = round(elapsed * 1000, 1)
