"""
Failure-map analysis (spec §2.1) — the scientific core of the per-region router.

A no-reference, per-pixel difficulty estimate built from orthogonal signals, each
catching one class of failure (disagreement, soft edges, model instability, flat
CG rim, text/strokes). Downstream the router dispatches each region to the expert
that fixes its dominant failure, instead of running every expert everywhere.
"""
from .failure_map import (
    FAILURE_CHANNELS,
    FailureMap,
    compute_failure_map,
)

__all__ = [
    "FAILURE_CHANNELS",
    "FailureMap",
    "compute_failure_map",
]
