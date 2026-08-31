"""
tests/test_hardware.py — hardware detection and Config adaptation.
Detection is read-only and safe to run anywhere; this is exactly what
you'd run on your real machine to see how MANA classifies it.
"""
from __future__ import annotations

from mana.hardware import HardwareProfile, apply_hardware_profile, detect_hardware


def test_detect_hardware_returns_sane_profile():
    profile = detect_hardware()
    assert profile.cpu_count >= 1
    assert profile.tier in {"low", "medium", "high"}
    assert profile.total_ram_gb >= 0.0


def test_apply_hardware_profile_never_raises_evolution_workers_above_preset(isolated_config):
    """The specific invariant this module claims: hardware adaptation only
    narrows explicit settings, never widens them."""
    isolated_config.evolution_workers = 1  # user (or CLI) explicitly asked for 1
    high_end = HardwareProfile(cpu_count=64, total_ram_gb=256.0, has_cuda=True,
                                gpu_name="fake-gpu", platform="test", tier="high")
    apply_hardware_profile(isolated_config, high_end)
    assert isolated_config.evolution_workers == 1, "must never increase an explicit setting"


def test_apply_hardware_profile_narrows_low_tier(isolated_config):
    before_population = isolated_config.strategy_population
    low_end = HardwareProfile(cpu_count=1, total_ram_gb=2.0, has_cuda=False,
                               gpu_name="", platform="test", tier="low")
    changes = apply_hardware_profile(isolated_config, low_end)
    assert isolated_config.strategy_population <= before_population
    assert isolated_config.use_embeddings is False
    assert "strategy_population" in changes or before_population <= 4


def test_cuda_gpu_is_decisive_even_when_ram_is_unreadable(monkeypatch):
    """Regression test for a bug found on a real Windows machine (GTX 1070,
    8 cores, no psutil installed): RAM read as 0.0, and the RAM-unknown
    branch fired BEFORE the CUDA check, classifying an obviously capable
    machine as 'medium'. Missing evidence about one signal must not
    discard the signals that were successfully read."""
    import mana.hardware as hw
    monkeypatch.setattr(hw, "_detect_ram_gb", lambda: (0.0, "unknown"))
    monkeypatch.setattr(hw, "_detect_gpu", lambda: (True, "NVIDIA GeForce GTX 1070", "torch"))
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 8)
    profile = hw.detect_hardware()
    assert profile.has_cuda is True
    assert profile.tier == "high", "a detected CUDA GPU must outweigh an unreadable RAM figure"


def test_unreadable_ram_without_gpu_still_falls_back_to_medium(monkeypatch):
    import mana.hardware as hw
    monkeypatch.setattr(hw, "_detect_ram_gb", lambda: (0.0, "unknown"))
    monkeypatch.setattr(hw, "_detect_gpu", lambda: (False, "", "none"))
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 4)
    assert hw.detect_hardware().tier == "medium"


def test_ram_detection_reports_its_source():
    """The reporter's log showed detected_via.ram == 'unknown', which is
    what surfaced the missing Windows fallback in the first place -- this
    field is diagnostic, so assert it stays populated."""
    from mana.hardware import _detect_ram_gb
    _gb, source = _detect_ram_gb()
    assert source in {"psutil", "/proc/meminfo", "GlobalMemoryStatusEx", "unknown"}
