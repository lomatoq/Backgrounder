"""
Level-1 decontamination — closed-form foreground unmixing (no training).

Given the composite I, the alpha matte α and a clean-plate estimate B̂, recover
the true foreground colour by inverting the matting equation:

    F(x) = (I(x) − (1 − α(x))·B̂(x)) / α(x),     for α(x) > τ

Where α → 0 the equation is ill-conditioned (dividing by ~0 amplifies noise and
the answer is unobservable anyway), so instead of trusting the inverse we
*propagate* F inward from the foreground boundary band-by-band (nearest reliable
F via a distance transform). A despill fallback additionally suppresses a
dominant background screen channel that leaks into antialiased edges where the
Euclidean clean-plate model is too coarse.

This replaces a CorridorKey-style chroma cleanup (CC BY-NC-SA, green-first) with
our own commercially-clean solver — see spec §2.3 Level 1 and §7 principle 7.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

__all__ = ["estimate_clean_plate", "unmix_foreground"]


def estimate_clean_plate(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    bg_rgb: Optional[Sequence[float]] = None,
    bg_threshold: float = 0.05,
    local: bool = False,
    sigma: float = 60.0,
) -> np.ndarray:
    """
    Estimate the background plate B̂ behind the foreground.

    Parameters
    ----------
    bg_rgb :
        Known clean-plate colour (e.g. router ``bg_rgb`` on a flat/CG plate). When
        given it short-circuits to a constant plate — the most reliable case.
    local :
        When True and ``bg_rgb`` is None, estimate a slowly-varying per-pixel plate
        by Gaussian-propagating definite-background pixels (handles gradients /
        vignettes). When False, use a single global median colour.

    Returns
    -------
    B̂ as float32 ``(H, W, 3)``.
    """
    img = image_rgb.astype(np.float32)
    h, w = alpha.shape

    if bg_rgb is not None:
        plate = np.asarray(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
        return np.broadcast_to(plate, (h, w, 3)).astype(np.float32)

    bg_mask = alpha < bg_threshold
    if bg_mask.sum() < 16:
        # Almost no certain background — fall back to a neutral mid grey so the
        # inverse stays bounded rather than exploding.
        return np.full((h, w, 3), 127.0, dtype=np.float32)

    if not local:
        color = np.median(img[bg_mask], axis=0).astype(np.float32)
        return np.broadcast_to(color.reshape(1, 1, 3), (h, w, 3)).astype(np.float32)

    # Per-region plate: blur the masked background and normalise by coverage so
    # foreground holes are filled by surrounding background colour.
    mask_f = bg_mask.astype(np.float32)
    plate = np.empty((h, w, 3), dtype=np.float32)
    den = gaussian_filter(mask_f, sigma=sigma) + 1e-6
    for c in range(3):
        num = gaussian_filter(img[..., c] * mask_f, sigma=sigma)
        plate[..., c] = num / den
    return plate


def unmix_foreground(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    bg_rgb: Optional[Sequence[float]] = None,
    plate: Optional[np.ndarray] = None,
    tau: float = 0.15,
    despill: bool = True,
    local_plate: bool = False,
) -> np.ndarray:
    """
    Recover the decontaminated foreground colour F (uint8 RGB).

    Pipeline:
      1. Build a clean-plate B̂ (provided ``plate``, constant ``bg_rgb``, or
         estimated from the image).
      2. Inverse-composite where α > τ to get F directly.
      3. Where α ≤ τ, propagate the nearest reliable F outward (band-by-band).
      4. Optional despill: clamp a dominant background screen channel at edges.

    The opaque interior (α ≈ 1) is left exactly as the original image — unmixing
    only acts on the partial-coverage band, so confident foreground colour is
    never disturbed.
    """
    img = image_rgb.astype(np.float32)
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    h, w = a.shape

    if plate is None:
        plate = estimate_clean_plate(img, a, bg_rgb=bg_rgb, local=local_plate)
    plate = plate.astype(np.float32)

    a3 = a[..., None]
    reliable = a > tau

    # 2. Inverse compositing on reliable pixels. Clamp the denominator so the
    #    band just above τ does not blow up.
    denom = np.maximum(a3, tau)
    fg_inv = (img - (1.0 - a3) * plate) / denom
    fg = np.where(reliable[..., None], fg_inv, img)

    # 3. Band-by-band propagation for α ≤ τ: each unobservable pixel inherits F
    #    from its nearest reliable neighbour (a distance-transform fill is the
    #    vectorised form of marching outward from the foreground boundary).
    unobservable = (~reliable) & (a > 1e-3)
    if unobservable.any() and reliable.any():
        _, (iy, ix) = distance_transform_edt(
            ~reliable, return_distances=True, return_indices=True
        )
        propagated = fg[iy, ix]
        fg = np.where(unobservable[..., None], propagated, fg)

    fg = np.clip(fg, 0.0, 255.0)

    # 4. Despill: where the background has a clearly dominant channel (blue/green
    #    screen), cap that channel of edge pixels at the max of the others so any
    #    residual screen tint that survived inverse compositing is removed.
    if despill:
        fg = _despill_edges(fg, a, plate, tau=tau)

    return np.clip(fg, 0.0, 255.0).astype(np.uint8)


def _despill_edges(
    fg: np.ndarray,
    alpha: np.ndarray,
    plate: np.ndarray,
    *,
    tau: float,
    dominance: float = 14.0,
) -> np.ndarray:
    """Suppress a dominant background screen channel on partial-alpha edges."""
    plate_med = np.median(plate.reshape(-1, 3), axis=0)
    key = int(np.argmax(plate_med))
    others = [c for c in range(3) if c != key]
    # Only meaningful when the plate genuinely leans on one channel (a screen),
    # not for neutral grey/white backgrounds where despill would tint the edge.
    if plate_med[key] - max(plate_med[others[0]], plate_med[others[1]]) < dominance:
        return fg

    edge = (alpha > 1e-3) & (alpha < 0.98)
    if not edge.any():
        return fg

    ceiling = np.maximum(fg[..., others[0]], fg[..., others[1]])
    spill = fg[..., key] > ceiling
    apply = edge & spill
    # Blend toward the ceiling proportionally to how transparent the pixel is —
    # fully opaque pixels keep their colour, the thinnest edge is fully despilled.
    strength = np.clip(1.0 - alpha, 0.0, 1.0)
    out = fg.copy()
    out[..., key] = np.where(
        apply,
        fg[..., key] * (1.0 - strength) + ceiling * strength,
        fg[..., key],
    )
    return out
