"""
Soft fusion of per-region expert mattes (spec §2.5).

    α*(x) = Σ_e w_e(x)·α_e(x) / Σ_e w_e(x)

with the per-expert weights feathered across region boundaries so the seams
between experts are invisible (no hard switching, spec §7 principle 4).
"""
from .blend import feather_weights, fuse_plan, soft_fuse

__all__ = ["feather_weights", "fuse_plan", "soft_fuse"]
