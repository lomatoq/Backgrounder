from __future__ import annotations

import numpy as np

from backgrounder.analyze.failure_map import FAILURE_CHANNELS, FailureMap
from backgrounder.fuse import feather_weights, fuse_plan, soft_fuse
from backgrounder.route import EXPERTS, plan_routes


def _single_region_map(channel: str, strength: float, h: int = 40, w: int = 40) -> FailureMap:
    """Hand-built failure map: one square region dominated by `channel`."""
    ch = FAILURE_CHANNELS.index(channel)
    stack = np.zeros((h, w, 5), dtype=np.float32)
    stack[10:30, 10:30, ch] = strength
    labels = np.zeros((h, w), dtype=np.int32)
    labels[10:30, 10:30] = ch + 1
    dominant = np.argmax(stack, axis=-1).astype(np.int32)
    return FailureMap(stack=stack, region_labels=labels, dominant=dominant)


# ── policy ───────────────────────────────────────────────────────────────────

def test_lambda_mode_changes_expert_for_soft_edge() -> None:
    fm = _single_region_map("U_trans", strength=0.6)

    fast = plan_routes(fm, mode="fast")
    mx = plan_routes(fm, mode="max")

    # Fast: λ large → expensive fine-matte loses to base (no dispatch).
    assert "sdmatte" not in fast.experts_used and "zim" not in fast.experts_used
    # Max: λ→0 → spend on the fine-matte expert.
    assert "sdmatte" in mx.experts_used or "zim" in mx.experts_used


def test_flat_region_routed_to_unmix_even_in_fast_mode() -> None:
    fm = _single_region_map("U_flat", strength=0.7)
    fast = plan_routes(fm, mode="fast")
    # unmix is near-free and high-affinity → wins even under large λ.
    assert "unmix" in fast.experts_used
    assert fast.expert_mask("unmix")[20, 20]


def test_availability_falls_back_to_next_expert() -> None:
    fm = _single_region_map("U_trans", strength=0.9)
    # No CUDA experts: sdmatte/sam3 unavailable → zim should take the region.
    plan = plan_routes(fm, mode="max", available=["unmix", "crisp", "zim"])
    assert "sdmatte" not in plan.experts_used
    assert "zim" in plan.experts_used


def test_no_region_keeps_base() -> None:
    stack = np.zeros((20, 20, 5), dtype=np.float32)
    labels = np.zeros((20, 20), dtype=np.int32)
    fm = FailureMap(stack=stack, region_labels=labels, dominant=labels.copy())
    plan = plan_routes(fm, mode="max")
    assert plan.experts_used == ()
    assert (plan.assignment == 0).all()


# ── fusion ───────────────────────────────────────────────────────────────────

def test_soft_fuse_is_weighted_average() -> None:
    a = {"base": np.full((4, 4), 0.2, np.float32), "unmix": np.full((4, 4), 0.8, np.float32)}
    w = {"base": np.full((4, 4), 1.0, np.float32), "unmix": np.full((4, 4), 3.0, np.float32)}
    fused = soft_fuse(a, w)
    expected = (1 * 0.2 + 3 * 0.8) / 4
    assert np.allclose(fused, expected, atol=1e-5)


def test_feather_weights_have_base_floor_and_smooth_region() -> None:
    fm = _single_region_map("U_flat", strength=0.8)
    plan = plan_routes(fm, mode="smart")
    weights = feather_weights(plan, feather_sigma=4.0)
    assert "base" in weights and np.allclose(weights["base"], 0.15)
    unmix_w = weights["unmix"]
    # Strong inside the region, decaying (non-binary) at the boundary.
    assert unmix_w[20, 20] > 0.8
    assert 0.0 < unmix_w[9, 20] < 1.0  # just outside the region edge → feathered


def test_fuse_plan_falls_back_to_base_for_missing_expert() -> None:
    fm = _single_region_map("U_trans", strength=0.9)
    plan = plan_routes(fm, mode="max", available=["zim"])
    base = np.full((40, 40), 0.5, np.float32)
    # Plan dispatched zim but we provide no zim result → must fall back to base.
    fused = fuse_plan(plan, base, expert_alphas={}, feather_sigma=3.0)
    assert np.allclose(fused, 0.5, atol=1e-5)


def test_fuse_plan_blends_expert_into_region() -> None:
    fm = _single_region_map("U_flat", strength=0.9)
    plan = plan_routes(fm, mode="smart")
    base = np.full((40, 40), 0.2, np.float32)
    unmix_alpha = np.full((40, 40), 0.9, np.float32)
    fused = fuse_plan(plan, base, {"unmix": unmix_alpha}, feather_sigma=3.0, base_floor=0.15)
    # Region centre pulled strongly toward the expert; far corner stays near base.
    assert fused[20, 20] > 0.6
    assert fused[0, 0] < 0.3
