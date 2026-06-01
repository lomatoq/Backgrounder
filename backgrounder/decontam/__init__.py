"""
Decontamination core — the project moat (see BACKGROUNDER_MASTER_SPEC §2.3).

Every output pixel with alpha < 1 must carry the recovered *foreground* colour F,
not the observed composite I. The compositing equation is

    I(x) = α(x)·F(x) + (1 − α(x))·B(x)

so emitting I leaves the background "bleeding" into semi-transparent pixels
(the blue/green rim everyone else ships). Level 1 here recovers F in closed form.
"""
from .unmix import estimate_clean_plate, unmix_foreground
from .trimap import adaptive_trimap

__all__ = [
    "estimate_clean_plate",
    "unmix_foreground",
    "adaptive_trimap",
]
