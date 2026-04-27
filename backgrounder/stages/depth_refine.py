from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def closed_form_matting_refine(
    image: Image.Image,
    alpha: np.ndarray,
    trimap: np.ndarray,
    *,
    radius: int = 1,
    eps: float = 1e-7,
    constraint_weight: float = 100.0,
    prior_weight: float = 0.01,
    max_solve_pixels: int = 65_536,
) -> np.ndarray:
    """
    Levin et al. closed-form alpha matting refinement.

    The solve is performed on a bounded resolution for predictable latency, then
    composited back only into trimap unknown pixels. Definite fg/bg pixels stay
    anchored to the coarse matte.
    """
    h, w = alpha.shape
    if not np.any(trimap == 128):
        return alpha.astype(np.float32)

    scale = min(1.0, (max_solve_pixels / float(h * w)) ** 0.5)
    if scale < 1.0:
        solve_w = max(16, int(round(w * scale)))
        solve_h = max(16, int(round(h * scale)))
        rgb = image.convert("RGB").resize((solve_w, solve_h), Image.BILINEAR)
        alpha_s = np.asarray(
            Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8), mode="L")
            .resize((solve_w, solve_h), Image.BILINEAR)
        ).astype(np.float32) / 255.0
        trimap_s = np.asarray(
            Image.fromarray(trimap, mode="L").resize((solve_w, solve_h), Image.NEAREST)
        )
    else:
        rgb = image.convert("RGB")
        alpha_s = alpha.astype(np.float32)
        trimap_s = trimap
        solve_w, solve_h = w, h

    image_np = np.asarray(rgb).astype(np.float32) / 255.0
    refined_s = _solve_closed_form(
        image_np,
        alpha_s,
        trimap_s,
        radius=radius,
        eps=eps,
        constraint_weight=constraint_weight,
        prior_weight=prior_weight,
    )

    if (solve_w, solve_h) != (w, h):
        refined = np.asarray(
            Image.fromarray((refined_s * 255).astype(np.uint8), mode="L").resize((w, h), Image.LANCZOS)
        ).astype(np.float32) / 255.0
    else:
        refined = refined_s

    unknown = (trimap == 128).astype(np.float32)
    result = alpha * (1.0 - unknown) + refined * unknown
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _solve_closed_form(
    image_np: np.ndarray,
    alpha: np.ndarray,
    trimap: np.ndarray,
    *,
    radius: int,
    eps: float,
    constraint_weight: float,
    prior_weight: float,
) -> np.ndarray:
    h, w = alpha.shape
    n_pixels = h * w
    win_size = (2 * radius + 1) ** 2

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    flat_idx = np.arange(n_pixels, dtype=np.int32).reshape(h, w)

    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            win_inds = flat_idx[y - radius : y + radius + 1, x - radius : x + radius + 1].ravel()
            win_rgb = image_np[y - radius : y + radius + 1, x - radius : x + radius + 1].reshape(win_size, 3)
            mu = win_rgb.mean(axis=0, keepdims=True)
            centered = win_rgb - mu
            cov = centered.T @ centered / win_size
            regularized = cov + (eps / win_size) * np.eye(3, dtype=np.float32)
            inv = np.linalg.pinv(regularized)
            affinity = (1.0 + centered @ inv @ centered.T) / win_size
            lap = np.eye(win_size, dtype=np.float32) - affinity.astype(np.float32)

            grid_r, grid_c = np.meshgrid(win_inds, win_inds, indexing="ij")
            rows.append(grid_r.ravel())
            cols.append(grid_c.ravel())
            vals.append(lap.ravel())

    if not rows:
        return alpha.astype(np.float32)

    L = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_pixels, n_pixels),
    ).tocsr()

    trimap_f = trimap.ravel()
    known = (trimap_f == 0) | (trimap_f == 255)
    weights = np.full(n_pixels, prior_weight, dtype=np.float32)
    weights[known] = constraint_weight

    target = alpha.ravel().astype(np.float32)
    target[trimap_f == 0] = 0.0
    target[trimap_f == 255] = 1.0

    A = L + diags(weights, 0, shape=(n_pixels, n_pixels), dtype=np.float32)
    b = weights * target
    solved = spsolve(A, b)
    return np.clip(solved.reshape(h, w), 0.0, 1.0).astype(np.float32)


def uncertainty_gated_sharpen(
    alpha: np.ndarray,
    uncertainty: np.ndarray,
    *,
    threshold: float = 0.15,
    strength: float = 0.60,
) -> np.ndarray:
    """
    Push alpha toward binary only where the ensemble agrees.

    This removes soft-but-unnecessary edges on clean objects while leaving high
    disagreement regions, hair, smoke, and glass available for matting.
    """
    confident = np.clip((threshold - uncertainty) / max(threshold, 1e-6), 0.0, 1.0)
    sharpened = alpha * alpha * (3.0 - 2.0 * alpha)
    mix = confident * strength
    result = alpha * (1.0 - mix) + sharpened * mix
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def smooth_alpha_boundary(alpha: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Light Gaussian smoothing on the alpha boundary to remove jaggedness."""
    uncertain = (alpha > 0.05) & (alpha < 0.95)
    smoothed = gaussian_filter(alpha, sigma=sigma)
    return np.where(uncertain, smoothed, alpha).astype(np.float32)
