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

    Two-pass approach:
    1. Estimate the background colour from definitely-background pixels.
    2. For boundary pixels, apply inverse-compositing to recover the true foreground
       colour: fg = (pixel - bg*(1-a)) / a.  This removes the background halo.
    Falls back to alpha-weighted Gaussian smoothing where inverse-compositing is
    numerically unstable (very low alpha).
    """
    from scipy.ndimage import gaussian_filter

    img = image_rgb.astype(np.float32)
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # 1. Background colour estimate: median of pixels where alpha < 0.05.
    bg_mask = a < 0.05
    if bg_mask.sum() > 16:
        bg_color = np.median(img[bg_mask], axis=0)  # shape (3,)
    else:
        # No clear background — fall back to Gaussian decontamination only.
        bg_color = None

    # 2. Gaussian-weighted foreground estimate (for low-alpha fallback).
    w = a ** 2
    fg_gauss = img.copy()
    for c in range(img.shape[2]):
        numerator = gaussian_filter(img[:, :, c] * w, sigma=sigma)
        denominator = gaussian_filter(w, sigma=sigma) + 1e-8
        fg_gauss[:, :, c] = numerator / denominator

    # 3. Inverse-compositing where alpha is high enough to be stable (a > 0.1).
    fg = img.copy()
    boundary = (a > 0.05) & (a < 0.95)
    if bg_color is not None:
        inv_a = np.where(a > 0.1, 1.0 / np.maximum(a, 0.1), 0.0)
        for c in range(img.shape[2]):
            decontam = (img[:, :, c] - bg_color[c] * (1.0 - a)) * inv_a
            # Blend: inverse-compositing for a>0.1, Gaussian for 0.05<a≤0.1.
            use_inv = boundary & (a > 0.1)
            use_gauss = boundary & (a <= 0.1)
            fg[:, :, c] = np.where(use_inv, decontam,
                          np.where(use_gauss, fg_gauss[:, :, c], img[:, :, c]))
    else:
        for c in range(img.shape[2]):
            fg[:, :, c] = np.where(boundary, fg_gauss[:, :, c], img[:, :, c])

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
