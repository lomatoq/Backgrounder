"""
No-reference failure-map analyzer (spec §2.1).

Five orthogonal channels, each tuned to catch one class of matting failure:

| Channel  | Signal                                            | Catches                |
|----------|---------------------------------------------------|------------------------|
| U_dis    | |α_BiRefNet − α_BEN2|                              | general uncertainty    |
| U_trans  | indicator α ∈ (ε, 1−ε)                             | soft edges (hair/glass)|
| U_tta    | variance of α over flip/scale augmentations       | model instability      |
| U_flat   | low local colour variance ∧ border-connected ∧    | flat / CG rim          |
|          | chroma-key score                                  | (blue/green screen)    |
| U_text   | Laplacian energy + stroke-width proxy             | text, thin lines, logo |

Output: a normalised stack ``U ∈ [0,1]^{H×W×5}`` plus discrete ``region_labels``
(argmax channel + connected components) marking the dominant failure per region.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_propagation,
    label,
    maximum_filter,
    uniform_filter,
)

# Channel order is fixed and used as the last axis of the stack.
FAILURE_CHANNELS = ("U_dis", "U_trans", "U_tta", "U_flat", "U_text")

# Label 0 is reserved for "no dominant failure" (base matte is trusted).
_LABEL_NONE = 0


@dataclass(frozen=True)
class FailureMap:
    """Per-pixel failure analysis."""

    stack: np.ndarray            # (H, W, 5) float32 in [0, 1], order = FAILURE_CHANNELS
    region_labels: np.ndarray    # (H, W) int32; per-pixel dominant-channel id (1..5), 0 = none
    dominant: np.ndarray         # (H, W) int32; argmax channel index (0..4) regardless of strength

    def channel(self, name: str) -> np.ndarray:
        return self.stack[..., FAILURE_CHANNELS.index(name)]

    def difficulty(self) -> np.ndarray:
        """Scalar per-pixel difficulty = max over channels (drop-in for trimaps)."""
        return self.stack.max(axis=-1)

    def metadata(self) -> dict:
        means = {f"failure_{c}_mean": round(float(self.stack[..., i].mean()), 4)
                 for i, c in enumerate(FAILURE_CHANNELS)}
        # Coverage of each dominant failure region as a fraction of the image.
        labelled = {
            f"failure_region_{c}": round(float((self.region_labels == i + 1).mean()), 4)
            for i, c in enumerate(FAILURE_CHANNELS)
        }
        return {**means, **labelled}


def compute_failure_map(
    image: Image.Image,
    alpha: np.ndarray,
    *,
    alpha_per_model: Optional[Sequence[np.ndarray]] = None,
    tta_variance: Optional[np.ndarray] = None,
    bg_rgb: Optional[Sequence[float]] = None,
    eps: float = 0.05,
    region_threshold: float = 0.30,
    min_region_area: int = 64,
) -> FailureMap:
    """
    Build the 5-channel failure map.

    Parameters
    ----------
    alpha :
        Fused base matte in [0, 1].
    alpha_per_model :
        Per-segmenter mattes (≥2) for the disagreement channel. When fewer than
        two are supplied, U_dis falls back to the transition band.
    tta_variance :
        Optional precomputed per-pixel α variance across TTA augmentations.
        When omitted, U_tta is zero (no instability evidence).
    bg_rgb :
        Known clean-plate colour (flat/CG route); sharpens the chroma-key term.
    """
    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = alpha.shape
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    u_dis = _channel_disagreement(alpha_per_model, a)
    u_trans = _channel_transition(a, eps)
    u_tta = _channel_tta(tta_variance, (h, w))
    u_flat = _channel_flat(img, a, bg_rgb)
    u_text = _channel_text(img)

    stack = np.stack([u_dis, u_trans, u_tta, u_flat, u_text], axis=-1).astype(np.float32)
    stack = np.clip(stack, 0.0, 1.0)

    dominant = np.argmax(stack, axis=-1).astype(np.int32)
    region_labels = _region_labels(stack, dominant, region_threshold, min_region_area)

    return FailureMap(stack=stack, region_labels=region_labels, dominant=dominant)


# ── channels ────────────────────────────────────────────────────────────────

def _channel_disagreement(
    alpha_per_model: Optional[Sequence[np.ndarray]],
    fused: np.ndarray,
) -> np.ndarray:
    if alpha_per_model is not None and len(alpha_per_model) >= 2:
        arr = np.stack([np.clip(a.astype(np.float32), 0.0, 1.0) for a in alpha_per_model], axis=0)
        # Max pairwise spread = (max − min) across models; robust to >2 models.
        return (arr.max(axis=0) - arr.min(axis=0)).astype(np.float32)
    # No second opinion: the soft band is the best available uncertainty proxy.
    return ((fused > 0.05) & (fused < 0.95)).astype(np.float32)


def _channel_transition(alpha: np.ndarray, eps: float) -> np.ndarray:
    """Soft transition mass, smoothed so thin bands form coherent regions."""
    band = ((alpha > eps) & (alpha < 1.0 - eps)).astype(np.float32)
    return np.clip(uniform_filter(band, size=5), 0.0, 1.0)


def _channel_tta(tta_variance: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if tta_variance is None:
        return np.zeros(shape, dtype=np.float32)
    v = np.asarray(tta_variance, dtype=np.float32)
    # Variance of a [0,1] signal is small; std (≤0.5) scaled to [0,1] reads better.
    return np.clip(np.sqrt(np.maximum(v, 0.0)) * 2.0, 0.0, 1.0)


def _channel_flat(
    img: np.ndarray,
    alpha: np.ndarray,
    bg_rgb: Optional[Sequence[float]],
) -> np.ndarray:
    """
    Flat / CG rim: pixels that are (a) in a locally flat colour neighbourhood,
    (b) connected to the image border, and (c) close to the clean-plate colour.
    """
    h, w = alpha.shape

    # (a) local colour flatness via per-channel local variance.
    local_var = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        ch = img[..., c]
        mean = uniform_filter(ch, size=7)
        sq = uniform_filter(ch * ch, size=7)
        local_var += np.maximum(sq - mean * mean, 0.0)
    flat = np.exp(-local_var / (3.0 * 12.0 ** 2)).astype(np.float32)  # 1 = perfectly flat

    # (c) clean-plate proximity (chroma-key score).
    border = _border_mask(h, w)
    plate = (np.asarray(bg_rgb, dtype=np.float32) if bg_rgb is not None
             else np.median(img[border], axis=0))
    dist = np.sqrt(((img - plate.reshape(1, 1, 3)) ** 2).sum(axis=-1))
    plate_like = np.clip(1.0 - dist / 60.0, 0.0, 1.0).astype(np.float32)

    # (b) connected to border through flat, plate-like pixels.
    bg_like = (flat > 0.6) & (plate_like > 0.5)
    seed = border & bg_like
    connected = binary_propagation(seed, mask=bg_like) if seed.any() else np.zeros_like(bg_like)

    score = flat * plate_like
    score = np.where(connected, score, score * 0.4)  # de-emphasise unconnected flat regions
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _channel_text(img: np.ndarray) -> np.ndarray:
    """
    Text / thin-stroke energy: high-frequency Laplacian response combined with a
    stroke-width proxy (thin, locally-isolated high-contrast structures).
    """
    luma = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722).astype(np.float32)

    # 4-neighbour Laplacian magnitude (separable-free, dependency-light).
    lap = np.zeros_like(luma)
    lap[1:-1, 1:-1] = (
        4.0 * luma[1:-1, 1:-1]
        - luma[:-2, 1:-1] - luma[2:, 1:-1]
        - luma[1:-1, :-2] - luma[1:-1, 2:]
    )
    energy = np.clip(np.abs(lap) / 80.0, 0.0, 1.0)

    # Stroke proxy: high-frequency energy that is *thin* — present at fine scale
    # but absent once blurred — discriminates strokes from textured regions.
    coarse = uniform_filter(energy, size=9)
    stroke = np.clip(energy - coarse, 0.0, 1.0)
    return np.clip(0.5 * energy + 0.5 * maximum_filter(stroke, size=3), 0.0, 1.0).astype(np.float32)


# ── regions ─────────────────────────────────────────────────────────────────

def _region_labels(
    stack: np.ndarray,
    dominant: np.ndarray,
    threshold: float,
    min_area: int,
) -> np.ndarray:
    """
    Per-pixel dominant-failure id (1..5), 0 where no channel exceeds threshold.

    Connected components smaller than ``min_area`` are dropped back to 0 so the
    router does not dispatch an expert for speckle.
    """
    strong = stack.max(axis=-1) >= threshold
    labels = np.where(strong, dominant + 1, _LABEL_NONE).astype(np.int32)

    out = np.zeros_like(labels)
    for ch in range(1, len(FAILURE_CHANNELS) + 1):
        mask = labels == ch
        if not mask.any():
            continue
        comp, n = label(mask)
        if n == 0:
            continue
        counts = np.bincount(comp.ravel())
        keep = np.where(counts >= min_area)[0]
        keep = keep[keep != 0]
        if keep.size:
            out[np.isin(comp, keep)] = ch
    return out


def _border_mask(h: int, w: int) -> np.ndarray:
    border_px = max(8, min(h, w) // 24)
    border = np.zeros((h, w), dtype=bool)
    border[:border_px, :] = True
    border[-border_px:, :] = True
    border[:, :border_px] = True
    border[:, -border_px:] = True
    return border
