"""
Soft fusion + boundary feathering (spec §2.5).

Each expert refines its own region. Naively pasting region results back produces
visible seams at the region borders. Instead we build a soft weight per expert
(1 inside its region, falling off smoothly across the boundary via a Gaussian)
and take the weighted average. Where weights are confidence maps (uncertainty or
gating softmax), the better-trusted expert simply dominates locally.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from backgrounder.route.policy import EXPERTS, RoutePlan


def soft_fuse(
    alpha_by_expert: Mapping[str, np.ndarray],
    weight_by_expert: Mapping[str, np.ndarray],
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Weighted average of expert mattes: α* = Σ w_e·α_e / Σ w_e.

    Experts present in ``alpha_by_expert`` but missing a weight default to weight
    zero. The result is clipped to [0, 1].
    """
    if not alpha_by_expert:
        raise ValueError("soft_fuse requires at least one expert matte")

    shape = next(iter(alpha_by_expert.values())).shape
    num = np.zeros(shape, dtype=np.float32)
    den = np.zeros(shape, dtype=np.float32)
    for name, alpha in alpha_by_expert.items():
        w = weight_by_expert.get(name)
        if w is None:
            continue
        w = w.astype(np.float32)
        num += w * np.clip(alpha.astype(np.float32), 0.0, 1.0)
        den += w
    fused = num / np.maximum(den, eps)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def feather_weights(
    plan: RoutePlan,
    *,
    feather_sigma: float = 6.0,
    base_floor: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Build a feathered weight map per expert from a route assignment.

    Each expert's hard region mask is Gaussian-blurred so its influence decays
    smoothly past the region edge. ``base`` gets a constant floor everywhere so a
    region whose expert produced garbage can never fully erase the base matte and
    so pixels assigned to no expert are still covered.
    """
    assignment = plan.assignment
    shape = assignment.shape
    weights: dict[str, np.ndarray] = {}

    used = set(plan.experts_used) | {"base"}
    for name in used:
        if name == "base":
            weights["base"] = np.full(shape, base_floor, dtype=np.float32)
            continue
        mask = (assignment == EXPERTS.index(name)).astype(np.float32)
        if feather_sigma > 0:
            mask = gaussian_filter(mask, sigma=feather_sigma)
        weights[name] = mask.astype(np.float32)
    return weights


def fuse_plan(
    plan: RoutePlan,
    base_alpha: np.ndarray,
    expert_alphas: Mapping[str, np.ndarray],
    *,
    confidences: Optional[Mapping[str, np.ndarray]] = None,
    feather_sigma: float = 6.0,
    base_floor: float = 0.15,
) -> np.ndarray:
    """
    Fuse a base matte with per-expert refined mattes according to a RoutePlan.

    Parameters
    ----------
    expert_alphas :
        Refined matte per expert name (only experts that actually ran need be
        present). Experts in the plan but missing here fall back to base.
    confidences :
        Optional per-expert confidence map multiplied into the feathered region
        weight (e.g. 1 − uncertainty, or a gating softmax). When omitted the
        region weight alone is used.
    """
    weights = feather_weights(plan, feather_sigma=feather_sigma, base_floor=base_floor)

    alphas: dict[str, np.ndarray] = {"base": base_alpha}
    for name, w in list(weights.items()):
        if name == "base":
            continue
        if name not in expert_alphas:
            # Expert was planned but did not run: drop its weight, keep base.
            weights.pop(name, None)
            continue
        alphas[name] = expert_alphas[name]
        if confidences is not None and name in confidences:
            weights[name] = w * confidences[name].astype(np.float32)

    return soft_fuse(alphas, weights)
