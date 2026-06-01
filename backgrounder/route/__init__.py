"""
Per-region router (spec §2.2).

Formalised as cost-sensitive dispatch / learning-to-defer on a region r:

    e*(r) = argmax_e [ ΔQ_e(features_r) − λ · cost_e ]

λ is literally the UI mode: Fast (large λ → only near-free experts win),
Smart Auto (balance), Max Quality (λ → 0 → spend on whatever helps). v1 uses
hand-tuned ΔQ/cost heuristics keyed off the failure map; v2/v3 swap in a learned
gating-net / value-net behind the same interface.
"""
from .policy import (
    EXPERTS,
    LAMBDA_BY_MODE,
    RoutePlan,
    plan_routes,
)

__all__ = [
    "EXPERTS",
    "LAMBDA_BY_MODE",
    "RoutePlan",
    "plan_routes",
]
