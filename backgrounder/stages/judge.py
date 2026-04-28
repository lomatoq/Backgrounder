from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, sobel


@dataclass
class QualityReport:
    score: float                      # composite [0, 1]; higher is better
    edge_sharpness: float             # mean alpha gradient in uncertain band
    depth_edge_iou: float             # agreement between alpha & depth edges
    confidence_mean: float            # mean BEN2 confidence in uncertain band
    jaggedness: float                 # fraction of reversals along boundary (lower = smoother)

    def __str__(self) -> str:
        return (
            f"score={self.score:.3f}  "
            f"sharpness={self.edge_sharpness:.3f}  "
            f"depth_iou={self.depth_edge_iou:.3f}  "
            f"conf={self.confidence_mean:.3f}  "
            f"jagged={self.jaggedness:.3f}"
        )


def score_alpha(
    alpha: np.ndarray,
    depth_edges: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    w_sharpness: float = 0.20,
    w_depth_iou: float = 0.30,
    w_confidence: float = 0.35,
    w_smoothness: float = 0.15,
) -> QualityReport:
    """
    Composite quality score from four orthogonal signals.

    edge_sharpness  — mean Sobel magnitude in α∈(0.05, 0.95); rewards crisp boundaries.
                      Weight lowered vs v1: soft alpha at hair tips is correct, not bad.
    depth_edge_iou  — IoU of dilated alpha-edges vs depth-edges; catches halo/low-contrast.
    confidence_mean — mean BEN2 confidence in uncertain band; rewards model certainty.
                      Primary signal — high model confidence → reliable mask.
    smoothness      — 1 − jaggedness; rewards smooth, non-oscillating boundary.
    """
    uncertain = (alpha > 0.05) & (alpha < 0.95)

    # --- 1. Edge sharpness ---
    gx = np.abs(sobel(alpha, axis=1))
    gy = np.abs(sobel(alpha, axis=0))
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_sharpness = float(grad_mag[uncertain].mean()) if uncertain.any() else 0.0
    # Normalise: typical BiRefNet boundary gradient ≈ 0.15–0.40
    edge_sharpness_norm = min(edge_sharpness / 0.35, 1.0)

    # --- 2. Depth-edge IoU ---
    depth_edge_iou = 0.5  # neutral default when no depth available
    if depth_edges is not None:
        alpha_edges = grad_mag > 0.08
        depth_strong = depth_edges > 0.15
        struct = np.ones((5, 5), dtype=bool)
        alpha_dil = binary_dilation(alpha_edges, structure=struct)
        depth_dil = binary_dilation(depth_strong, structure=struct)
        intersection = (alpha_dil & depth_dil).sum()
        union = (alpha_dil | depth_dil).sum()
        depth_edge_iou = float(intersection / (union + 1e-8))

    # --- 3. BEN2 confidence in uncertain band ---
    confidence_mean = 0.5  # neutral default
    if confidence is not None and uncertain.any():
        confidence_mean = float(confidence[uncertain].mean())

    # --- 4. Smoothness (anti-jaggedness) ---
    boundary_pixels = grad_mag > 0.05
    jaggedness = 0.0
    if boundary_pixels.any():
        # Count sign-reversals in gradient direction along rows
        gx_boundary = gx[boundary_pixels]
        reversals = np.diff(np.sign(gx_boundary))
        jaggedness = float((reversals != 0).mean())
    smoothness_norm = max(0.0, 1.0 - jaggedness * 2)

    score = (
        w_sharpness * edge_sharpness_norm
        + w_depth_iou * depth_edge_iou
        + w_confidence * confidence_mean
        + w_smoothness * smoothness_norm
    )

    return QualityReport(
        score=float(np.clip(score, 0.0, 1.0)),
        edge_sharpness=edge_sharpness_norm,
        depth_edge_iou=depth_edge_iou,
        confidence_mean=confidence_mean,
        jaggedness=jaggedness,
    )
