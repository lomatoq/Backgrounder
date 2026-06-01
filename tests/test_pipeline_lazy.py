from __future__ import annotations

from backgrounder.config import Device, PipelineConfig
from backgrounder.pipeline import BackgroundRemovalPipeline, _ProgressReporter


def _cpu_pipeline(**overrides) -> BackgroundRemovalPipeline:
    cfg = PipelineConfig(device=Device.CPU, **overrides)
    p = BackgroundRemovalPipeline(cfg)
    p._device = "cpu"  # resolve_device would do this; force for the test
    return p


def test_ensure_returns_none_when_expert_disabled() -> None:
    p = _cpu_pipeline(use_sdmatte=False, use_sam3=False, use_sam2=False, use_vitmatte=False)
    assert p._ensure_sdmatte() is None
    assert p._ensure_sam2() is None
    assert p._ensure_sam3() is None
    assert p._ensure_vitmatte() is None
    assert p._ensure_owlv2() is None


def test_sdmatte_and_sam3_need_cuda() -> None:
    # Enabled but on CPU → must not load (returns None), and SAM3 records why.
    p = _cpu_pipeline(use_sdmatte=True, use_sam3=True)
    assert p._ensure_sdmatte() is None
    assert p._ensure_sam3() is None
    assert p._sam3_load_error and "CUDA" in p._sam3_load_error


def test_available_experts_reflects_config_and_device() -> None:
    cpu = _cpu_pipeline(use_sdmatte=True, use_sam3=True)
    # No CUDA → heavy experts are not offered to the router.
    assert set(cpu._available_experts()) == {"base", "unmix", "crisp"}


def test_release_expert_is_noop_without_lazy() -> None:
    p = _cpu_pipeline(lazy_load_experts=False)
    p._sdmatte = object()  # pretend a model is resident
    p._release_expert("_sdmatte")
    assert p._sdmatte is not None  # not unloaded when lazy mode is off


def test_progress_reporter_is_monotonic_and_tolerant() -> None:
    seen = []
    rep = _ProgressReporter(lambda frac, desc="": seen.append((frac, desc)))
    rep(0.5, "half")
    rep(0.2, "back")   # lower fraction must not regress
    rep(1.0, "done")
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs)
    assert fracs[0] == 0.5 and fracs[-1] == 1.0

    # A callback that raises must never propagate.
    bad = _ProgressReporter(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    bad(0.3, "x")  # should not raise


def test_progress_reporter_none_is_safe() -> None:
    rep = _ProgressReporter(None)
    rep(0.4, "noop")  # no callback → silently ignored
