from __future__ import annotations

import numpy as np


def sharpen_alpha(alpha: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """
    Stage D' — Uncertainty-gated alpha contrast enhancement.

    Where the ensemble segmenters agree (low uncertainty) and the alpha is
    near a boundary, push values toward 0 or 1 using a sigmoid contrast
    curve.  Where models disagree (high uncertainty), leave alpha soft so
    natural semi-transparent details (hair, fur, smoke) are preserved.

    Parameters
    ----------
    alpha       : float32 H×W [0, 1]
    uncertainty : float32 H×W [0, 1]  — per-pixel std from ensemble, normalised
    """
    near_boundary = (alpha > 0.04) & (alpha < 0.96)
    if not near_boundary.any():
        return alpha

    confidence = np.clip(1.0 - uncertainty, 0.0, 1.0)

    # Sigmoid-based contrast curve: f(a, k) = σ(k * (2a − 1))
    # k=0 → f = 0.5 (flat);  k=8 → strong S-curve toward 0/1.
    k = confidence * 8.0
    alpha_sharp = 1.0 / (1.0 + np.exp(-k * (2.0 * alpha - 1.0)))

    # Blend: high-confidence pixels get the sharpened value;
    # uncertain pixels keep their original soft value.
    blended = confidence * alpha_sharp + (1.0 - confidence) * alpha

    result = alpha.copy()
    result[near_boundary] = blended[near_boundary].astype(np.float32)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
