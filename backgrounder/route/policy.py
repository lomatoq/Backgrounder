"""
v1 cost-sensitive per-region routing policy (spec §2.2).

The failure map tells us *what* is wrong where; the policy decides *which* expert
to spend on each region, trading expected quality gain against expert cost via the
mode dial λ. v1 is pure heuristic (no training) but already implements the moat:
different regions of one image go to different experts, then get fused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from backgrounder.analyze.failure_map import FAILURE_CHANNELS, FailureMap

# Expert roster (spec §3). "base" = keep the ensemble matte (always free).
EXPERTS = ("base", "unmix", "crisp", "zim", "sdmatte", "sam3")

# Relative cost (latency/VRAM), normalised so the heaviest = 1.0.
COST = {
    "base": 0.0,
    "unmix": 0.05,   # closed-form, CPU
    "crisp": 0.05,   # connectivity keyer, CPU
    "zim": 0.40,     # fine-matte network
    "sdmatte": 0.80, # diffusion-arch forward
    "sam3": 1.00,    # concept segmentation
}

# Expected quality gain ΔQ when a channel dominates a region, per candidate
# expert (the expert that actually fixes that failure class). Tuned by hand for
# v1; a learned predictor replaces this table in v2.
AFFINITY: dict[str, dict[str, float]] = {
    "U_dis":   {"sam3": 0.60, "zim": 0.30},          # disagreement → stronger seg
    "U_trans": {"sdmatte": 0.80, "zim": 0.70, "unmix": 0.20},  # soft edges → fine matte
    "U_tta":   {"sam3": 0.50, "sdmatte": 0.50},      # instability → robust model
    "U_flat":  {"unmix": 0.90, "crisp": 0.30},       # flat rim → decontam unmix
    "U_text":  {"crisp": 0.90, "unmix": 0.20},       # text/strokes → crisp key
}

# λ by UI mode. Fast keeps only near-free experts; Max spends freely.
LAMBDA_BY_MODE = {
    "fast": 1.0,
    "smart": 0.35,
    "max": 0.0,
}


@dataclass(frozen=True)
class RegionDecision:
    channel: str          # dominant failure channel for the region
    expert: str           # chosen expert
    strength: float       # mean failure strength in the region
    dq: float             # expected quality gain ΔQ
    cost: float           # expert cost
    score: float          # ΔQ − λ·cost (the maximised objective)
    area: int             # pixel count


@dataclass(frozen=True)
class RoutePlan:
    assignment: np.ndarray              # (H, W) int; index into EXPERTS per pixel
    decisions: tuple[RegionDecision, ...]
    lam: float
    mode: str

    @property
    def experts_used(self) -> tuple[str, ...]:
        ids = np.unique(self.assignment)
        return tuple(EXPERTS[i] for i in ids if EXPERTS[i] != "base")

    def expert_mask(self, expert: str) -> np.ndarray:
        return self.assignment == EXPERTS.index(expert)

    def metadata(self) -> dict:
        return {
            "route_mode": self.mode,
            "route_lambda": self.lam,
            "route_experts_used": list(self.experts_used),
            "route_decisions": [
                {
                    "channel": d.channel,
                    "expert": d.expert,
                    "strength": round(d.strength, 3),
                    "dq": round(d.dq, 3),
                    "score": round(d.score, 3),
                    "area": d.area,
                }
                for d in self.decisions
            ],
        }


def plan_routes(
    failure_map: FailureMap,
    *,
    mode: str = "smart",
    lam: Optional[float] = None,
    available: Optional[Iterable[str]] = None,
) -> RoutePlan:
    """
    Decide an expert per failure region.

    Parameters
    ----------
    mode :
        One of ``fast`` / ``smart`` / ``max`` (selects λ). Ignored if ``lam`` is
        given explicitly.
    available :
        Experts that can actually run (e.g. drop ``sdmatte``/``sam3`` without
        CUDA). ``base`` is always available; unavailable candidates are skipped so
        the policy degrades gracefully instead of dispatching a missing model.
    """
    if lam is None:
        lam = LAMBDA_BY_MODE.get(mode, LAMBDA_BY_MODE["smart"])
    avail = set(available) if available is not None else set(EXPERTS)
    avail.add("base")

    labels = failure_map.region_labels
    assignment = np.zeros(labels.shape, dtype=np.int32)  # default → base (0)
    decisions: list[RegionDecision] = []

    for ch_idx, channel in enumerate(FAILURE_CHANNELS):
        region = labels == ch_idx + 1
        area = int(region.sum())
        if area == 0:
            continue
        strength = float(failure_map.stack[..., ch_idx][region].mean())

        best_expert, best_dq, best_score = _best_expert(channel, strength, lam, avail)
        if best_expert == "base":
            continue  # base already assigned; no dispatch

        assignment[region] = EXPERTS.index(best_expert)
        decisions.append(
            RegionDecision(
                channel=channel,
                expert=best_expert,
                strength=strength,
                dq=best_dq,
                cost=COST[best_expert],
                score=best_score,
                area=area,
            )
        )

    return RoutePlan(
        assignment=assignment,
        decisions=tuple(decisions),
        lam=float(lam),
        mode=mode,
    )


def _best_expert(
    channel: str,
    strength: float,
    lam: float,
    avail: set[str],
) -> tuple[str, float, float]:
    """argmax_e [ΔQ_e − λ·cost_e] over candidate experts for this channel."""
    # Baseline: keeping the base matte is a zero-gain, zero-cost option.
    best_expert, best_dq, best_score = "base", 0.0, 0.0
    for expert, affinity in AFFINITY.get(channel, {}).items():
        if expert not in avail:
            continue
        dq = affinity * strength
        score = dq - lam * COST[expert]
        if score > best_score:
            best_expert, best_dq, best_score = expert, dq, score
    return best_expert, best_dq, best_score
