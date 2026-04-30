from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, sobel, label, binary_fill_holes

from backgrounder.stages.depth_refine import (
    smooth_alpha_boundary,
    closed_form_matting_refine,
)

if TYPE_CHECKING:
    from backgrounder.models.vitmatte import ViTMatteRefiner


def expert_refine(
    image: Image.Image,
    alpha: np.ndarray,
    trimap: np.ndarray,
    expert: str,
    depth_edges: Optional[np.ndarray],
    vitmatte: Optional["ViTMatteRefiner"] = None,
    use_closed_form: bool = True,
    closed_form_max_pixels: int = 65_536,
) -> np.ndarray:
    """
    Route to the appropriate Stage-D refiner based on subject type.

    expert = "vitmatte"   → ViTMatte (hair/fur) + Fourier Wiener sharpening
    expert = "depth_only" → L0 gradient minimisation (FFT, Xu 2011) — crisp edges
    expert = "color_key"  → background-color-distance extraction (text/logos)

    Pre-step (vitmatte + depth_only): closed-form matting (Levin 2006) initialises
    the unknown band with image-colour-consistent alpha values so the subsequent
    per-expert solver starts from a warm solution.
    """
    unknown = (trimap == 128).astype(np.float32)
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    if expert == "color_key":
        alpha_ck = _color_key_extract(image, alpha)
        alpha = 0.75 * alpha_ck + 0.25 * alpha
        # Fine-scale guided filter for text edges.
        alpha_gf = _guided_filter(image_np, alpha, r=2, eps=5e-5)
        alpha = np.where((alpha > 0.03) & (alpha < 0.97), alpha_gf, alpha)
        alpha = np.where(alpha > 0.88, 1.0, alpha)
        alpha = np.where(alpha < 0.12, 0.0, alpha)
        return np.clip(fill_alpha_holes(alpha), 0.0, 1.0).astype(np.float32)

    # ── Pre-step: closed-form matting (Levin 2006) in unknown band ──────────
    # Anchors boundary pixels to image-colour-consistent alpha values before
    # the per-expert refiner; provides a warm start that reduces residual error.
    if use_closed_form:
        alpha = closed_form_matting_refine(
            image, alpha, trimap, max_solve_pixels=closed_form_max_pixels
        )

    if expert == "vitmatte" and vitmatte is not None:
        alpha = vitmatte.refine(image, alpha, trimap)
        # Guided filter (r=4) smooths ViTMatte's residual jaggedness.
        alpha_gf = _guided_filter(image_np, alpha, r=4, eps=1e-3)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)
        # Fourier-domain Wiener deconvolution reverses the mild Gaussian blur
        # the guided filter introduces — recovers fine hair-strand sharpness.
        alpha = wiener_sharpen_alpha(alpha, sigma=1.0, snr=45.0)
        alpha = fill_alpha_holes(alpha)
        return smooth_alpha_boundary(alpha, sigma=0.5)

    elif expert == "vitmatte":
        # ViTMatte not loaded: gradient-weighted multi-scale guided filter.
        alpha_gf = multiscale_guided_filter(image_np, alpha)
        alpha = np.where(unknown > 0.5, alpha_gf, alpha)
        alpha = fill_alpha_holes(alpha)
        return smooth_alpha_boundary(alpha, sigma=0.5)

    else:
        # depth_only — products, vehicles, generic, complex.
        # L0 gradient minimisation (Xu et al. SIGGRAPH Asia 2011):
        # forces alpha piecewise-constant with sparse, sharp transitions.
        # Eliminates the soapy halo that the coarse guided filter produced.
        alpha_l0 = l0_smooth_alpha(alpha, lam=0.005)
        alpha = np.where(unknown > 0.5, alpha_l0, alpha)
        alpha = np.where((alpha > 0.92) & (unknown > 0.5), 1.0, alpha)
        alpha = np.where((alpha < 0.08) & (unknown > 0.5), 0.0, alpha)
        alpha = fill_alpha_holes(alpha)
        return smooth_alpha_boundary(alpha, sigma=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Fourier / FFT refiners
# ══════════════════════════════════════════════════════════════════════════════

def l0_smooth_alpha(
    alpha: np.ndarray,
    lam: float = 0.005,
    kappa: float = 2.0,
    max_iter: int = 20,
) -> np.ndarray:
    """
    L0 gradient minimisation — Xu et al. "Image Smoothing via L0 Gradient
    Minimization", SIGGRAPH Asia 2011.  Adapted for alpha mattes.

    Minimises:  E(S) = ||S − alpha||² + λ · #{p : |∂xS|² + |∂yS|² ≠ 0}

    The sparsity term forces alpha piecewise-constant: object body pixels
    snap to exactly 0 or 1; the ONLY non-zero gradients are at the object
    boundary — a single crisp transition.  This eliminates the wide soft
    gradient ("soap halo") created by over-smoothed guided filters.

    Solved via alternating minimisation + 2-D FFT in O(N log N) per iteration:

      (h, v) sub-problem — hard threshold: zero gradient where mag² < λ/β
      S       sub-problem — FFT linear solve:
          FS = (FT(alpha) + β · FT(div)) / (1 + β · MTF)

    beta starts at 2λ and doubles each iteration; ~16 iters reach β = 10⁵λ.
    Typical runtime: 5–20 ms for a 1 MP alpha matte.
    """
    S = alpha.astype(np.float64)
    H, W = S.shape

    # FFT of forward-difference kernels: Kx[0,0]=1, Kx[0,1]=−1
    # MTF = |FT(∂x)|² + |FT(∂y)|² — power transfer of the Laplacian.
    Kx = np.zeros((H, W), np.float64); Kx[0, 0] = 1.0; Kx[0, 1] = -1.0
    Ky = np.zeros((H, W), np.float64); Ky[0, 0] = 1.0; Ky[1, 0] = -1.0
    FKx = np.fft.fft2(Kx)
    FKy = np.fft.fft2(Ky)
    MTF = np.abs(FKx) ** 2 + np.abs(FKy) ** 2
    FTI = np.fft.fft2(S)   # FT of input — stays constant throughout

    beta = 2.0 * lam
    for _ in range(max_iter):
        # (h, v) sub-problem: forward-diff then hard threshold.
        dx = np.roll(S, -1, axis=1) - S
        dy = np.roll(S, -1, axis=0) - S
        sparse = (dx * dx + dy * dy) < (lam / beta)
        h = np.where(sparse, 0.0, dx)
        v = np.where(sparse, 0.0, dy)

        # S sub-problem: backward-divergence then FFT linear solve.
        # div(h,v)[i,j] = h[i,j] − h[i,j−1] + v[i,j] − v[i−1,j]
        div = (h - np.roll(h, 1, axis=1)) + (v - np.roll(v, 1, axis=0))
        S = np.real(np.fft.ifft2(
            (FTI + beta * np.fft.fft2(div)) / (1.0 + beta * MTF)
        ))

        beta *= kappa
        if beta > lam * 1e5:
            break

    return np.clip(S, 0.0, 1.0).astype(np.float32)


def wiener_sharpen_alpha(
    alpha: np.ndarray,
    sigma: float = 1.0,
    snr: float = 45.0,
) -> np.ndarray:
    """
    Fourier-domain Wiener deconvolution to reverse Gaussian blur in alpha.

    Models current alpha ≈ alpha_sharp * h(σ) where h is a Gaussian PSF.
    Inverts in the frequency domain using the Wiener filter:

        A_sharp(ξ) = A(ξ) · H(ξ) / (H(ξ)² + 1/SNR)

    where H(ξ) = exp(−2π²σ²|ξ|²) is the analytic Fourier transform of the
    2-D Gaussian (no PSF array required — evaluated directly at FFT freqs).

    H is real and non-negative for a Gaussian, so H* = H, giving:
        W(ξ) = H / (H² + ε)    where ε = 1/SNR controls regularisation.

    Applied only in the uncertain band (alpha ∈ 0.05, 0.95) — definite
    foreground/background pixels are not sharpened to avoid noise amplification.

    Typical use: post-ViTMatte guided_filter(r=4) introduces ~1-px Gaussian blur;
    Wiener with sigma=1.0 reverses it and recovers fine hair-strand sharpness.
    """
    H_arr, W_arr = alpha.shape
    fy = np.fft.fftfreq(H_arr)[:, np.newaxis]
    fx = np.fft.fftfreq(W_arr)[np.newaxis, :]
    # Analytic FT of 2-D Gaussian with spatial std σ: G(fx,fy)=exp(−2π²σ²(fx²+fy²))
    H_fft = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))

    A_fft = np.fft.fft2(alpha.astype(np.float64))
    # Wiener filter (H real → H* = H, no conjugate needed)
    W_filt = H_fft / (H_fft ** 2 + 1.0 / snr)
    A_sharp = np.real(np.fft.ifft2(A_fft * W_filt))

    # Blend sharpened result only in uncertain band
    uncertain = (alpha > 0.05) & (alpha < 0.95)
    result = alpha.copy().astype(np.float64)
    result[uncertain] = A_sharp[uncertain]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Alpha post-processing
# ══════════════════════════════════════════════════════════════════════════════

def fill_alpha_holes(
    alpha: np.ndarray,
    max_hole_ratio: float = 0.008,
) -> np.ndarray:
    """
    Fill small enclosed background pockets inside the foreground mask.

    Interior specular reflections, patterned clothing, and semi-transparent
    patches within an opaque object create "holes" — alpha≈0 pixels fully
    surrounded by alpha≈1 pixels.  These should be filled to 1.0.

    Only holes smaller than max_hole_ratio × total_pixels are filled to avoid
    touching intentional negative-space (rings, donuts, hollow objects).
    max_hole_ratio=0.008 ≈ 0.8% of image = a ~90×90 hole at 1 MP.
    """
    fg = alpha > 0.5
    filled = binary_fill_holes(fg)
    holes = filled & ~fg
    if not holes.any():
        return alpha

    labeled_arr, n_labels = label(holes)
    max_size = max_hole_ratio * alpha.size

    result = alpha.copy()
    for i in range(1, n_labels + 1):
        hole_mask = labeled_arr == i
        if hole_mask.sum() <= max_size:
            result[hole_mask] = 1.0
    return result.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Spatial guided filters (retained for ViTMatte-fallback + transparency_refine)
# ══════════════════════════════════════════════════════════════════════════════

def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    r: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    def box(x: np.ndarray) -> np.ndarray:
        return uniform_filter(x.astype(np.float64), size=2 * r + 1)

    I = guide.mean(axis=2)
    mean_I  = box(I)
    mean_p  = box(src)
    mean_Ip = box(I * src)
    mean_II = box(I * I)
    cov_Ip  = mean_Ip - mean_I * mean_p
    var_I   = mean_II - mean_I ** 2
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return np.clip(box(a) * I + box(b), 0.0, 1.0).astype(np.float32)


def multiscale_guided_filter(
    guide: np.ndarray,  # H×W×3 float32 [0,1]
    src: np.ndarray,    # H×W   float32 [0,1]
) -> np.ndarray:
    """
    Gradient-weighted multi-scale guided filter fusion.

    Three guided filters are merged per-pixel using the IMAGE gradient to
    select scale (not the alpha gradient — alpha noise would force fine-scale
    everywhere, defeating the purpose):

      fine   (r=2,  eps=1e-5) — dominates at real object boundaries
      medium (r=6,  eps=5e-4) — semi-transparent / soft-shadow regions
      coarse (r=8,  eps=3e-3) — smooth bodies, uniform backgrounds
                                 (radius reduced from 12→8 to limit soap halo)

    Blend weights (quadratic, sum to 1):
        w_fine   = g²        w_medium = 2g(1−g)        w_coarse = (1−g)²
    """
    fine   = _guided_filter(guide, src, r=2, eps=1e-5)
    medium = _guided_filter(guide, src, r=6, eps=5e-4)
    coarse = _guided_filter(guide, src, r=8, eps=3e-3)

    I = guide.mean(axis=2).astype(np.float64)
    gx = np.abs(sobel(I, axis=1))
    gy = np.abs(sobel(I, axis=0))
    g = np.sqrt(gx ** 2 + gy ** 2)
    g = (g / (g.max() + 1e-8)).astype(np.float32)

    w_fine   = g * g
    w_coarse = (1.0 - g) * (1.0 - g)
    w_medium = 2.0 * g * (1.0 - g)

    return np.clip(
        w_fine * fine + w_medium * medium + w_coarse * coarse,
        0.0, 1.0,
    ).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Color-key extractor (text / logos)
# ══════════════════════════════════════════════════════════════════════════════

def _color_key_extract(
    image: Image.Image,
    alpha_coarse: np.ndarray,
) -> np.ndarray:
    """
    Background-color-keying for text / logos on near-solid backgrounds.

    1. Background pixel set: border strip ∪ neural alpha < 0.05.
    2. Background colour = median of those pixels.
    3. Per-pixel Euclidean RGB distance from background.
    4. Soft threshold via 90th-pct distance of confirmed background pixels.
    5. Linear ramp [0,1] alpha.
    """
    img = np.array(image.convert("RGB")).astype(np.float32)
    h, w = img.shape[:2]
    border_px = max(8, min(h, w) // 20)

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_px, :] = True
    border_mask[-border_px:, :] = True
    border_mask[:, :border_px] = True
    border_mask[:, -border_px:] = True
    bg_mask = border_mask | (alpha_coarse < 0.05)

    bg_pixels = img[bg_mask] if bg_mask.any() else img.reshape(-1, 3)
    bg_color = np.median(bg_pixels, axis=0)

    dist = np.sqrt(((img - bg_color) ** 2).sum(axis=2))
    bg_dist = dist[bg_mask]
    bg_threshold = float(np.percentile(bg_dist, 90)) if len(bg_dist) > 10 else 20.0
    bg_threshold = max(bg_threshold, 12.0)

    spread = max(bg_threshold * 2.0, 20.0)
    alpha = np.clip((dist - bg_threshold) / (spread + 1e-6), 0.0, 1.0)
    return alpha.astype(np.float32)
