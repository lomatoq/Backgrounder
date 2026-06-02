# Backgrounder — Implementation Plan (seg-first / SAMA-aligned)

> Companion to `BACKGROUNDER_MASTER_SPEC.md`. Grounded in the SAMA paper
> (arXiv:2601.12147, "Segment and Matte Anything in a Unified Model", AAAI 2026).
> Decision: **SAM 3.1 is the primary silhouette ("seg-first")**; we rebuild the
> pipeline into the spec §1 scheme and retire the conflicting whole-image
> heuristic cascade that caused regressions.

## Why (root cause of the thrash)

The shipped pipeline ran a whole-image **heuristic cascade** (closed_form on
everything, uncertainty_sharpen, normalize_graphic, solid_bg cleanup, 3× visual
re-keys, SAM3 bolted on as a late refiner with fragile accept gates). Each stage
makes a *global* assumption and they fight: every fix shifts the failure
elsewhere (dark suit → sweater → letters → oven). Diagnostics proved this:
SAM 3.1 already produces a clean silhouette, but downstream re-keys (e.g.
`remove_checkerboard_background` sampling a light wall == a light sweater) punch
the body back into speckle.

## Target = SAMA, realised in two stages

SAMA = **frozen SAM + MVLE + Local-Adapter + dual (seg+matte) heads**, one
forward → clean silhouette **and** fine alpha. It is literally "seg-first that
also mattes", which dissolves our problem (no fighting refiners; prompts can
pick detached parts). **Its weights are not released** → real SAMA needs
training. So:

- **Stage 1 (now, no training):** approximate SAMA with a clean *seg → matte*
  cascade (the MatAny pattern SAMA cites): SAM 3.1 silhouette → adaptive trimap →
  matting expert for edges → decontam. Re-architect to spec §1; retire the
  fighting heuristics.
- **Stage 2 (later, trained):** build MVLE + Local-Adapter + dual-head on a SAM
  backbone (fork SEMat, not from scratch — spec §5 Phase 4), train on
  DIS-5K/ThinObject-5K (seg) + AIM/AIM-500 (matte; Adobe AIM by request), drop in
  as the unified opaque expert. Gated on compute + dataset licensing.

## Spec §1 pipeline → modules

```
0 Classify subject            backgrounder/classifier.py            (exists)
1 BASE PASS  BiRefNet+BEN2     stages/ensemble.py                    (exists)  → α0, α0' (for U_dis + SAM3 box prompts + fallback)
2 FAILURE-MAP (5 channels)     analyze/failure_map.py                (exists)  U_dis/U_trans/U_tta/U_flat/U_text + region_labels
3 ROUTER  e*(r)=argmax ΔQ−λc   route/policy.py                       (exists)  λ = Fast/Smart/Max; per-region dispatch
4 EXPERTS (per region/class)
    OPAQUE OBJECT  → SAM 3.1 silhouette → matting edge-refine        models/sam3.py + stages/expert_refine.py
    FLAT / CG      → unmix-solver                                    decontam/unmix.py
    TEXT / LOGO    → crisp keyer (+ RGBA passthrough)                stages/solid_background.py + pipeline passthrough
    FINE / HAIR    → ViTMatte / SDMatte / (FBA later)                models/vitmatte.py, models/sdmatte.py
    TRANSPARENT    → soft matting (NO SAM3)                          stages/transparency.py
5 DECONTAM CORE  F=(I−(1−α)B̂)/α → FBA (later)                       decontam/unmix.py, (decontam/fba.py later)
6 SOFT FUSION   α*=Σ w·α/Σ w + feather                              fuse/blend.py
```

## Stage 1 — concrete steps (each verified on `eval/test_images` before moving on)

- **1a. SAM 3.1 = primary silhouette for opaque classes.** For
  {portrait, animal_fur, plant_thin, product*, vehicle, anime, flat_cartoon,
  complex_multi, generic}: run SAM 3.1 as the authoritative coarse alpha (box
  prompts from the base ensemble), not an occasional score-gated refiner. Base
  ensemble still runs (cheap) for the failure map + prompts + non-CUDA fallback.
  Keep the strict gate: never let SAM erase confident base foreground (detached
  parts).
- **1b. Matting refines only the edge band.** Build an adaptive trimap from the
  SAM silhouette (`decontam/trimap.py`), run the matting expert
  (ViTMatte/SDMatte) on the unknown band only → recover hair/fur; interior stays
  solid. Then decontam (`decontam/unmix.py`, emit F not I).
- **1c. Retire conflicting heuristics on the opaque path.** uncertainty_sharpen,
  normalize_graphic, the three visual_qa re-keys and solid_bg cleanup no longer
  run once a confident silhouette exists; their valid behaviour stays only on the
  routes that actually need them (text/graphic/flat).
- **Routes preserved:** text/logo → crisp + RGBA passthrough; transparent/glass →
  soft matting, never SAM3; flat/CG keyable → unmix.

## Non-negotiables (spec §7)

1. Emit **F**, not the composite I. 2. Route per region. 3. Adaptive trimap.
4. Soft fusion, no hard seam. 5. Crisp text, soft hair. 6. Eval before optimise
(`eval/run_eval.py` + `eval/test_images/`). 7. No CorridorKey.

## Verification protocol

After every step: `python eval/run_eval.py --mode max` and inspect
`eval/results/max/<name>/` (rgba + per-stage alpha dumps). No "done" claim until
the cutouts are visually confirmed on the test set.
