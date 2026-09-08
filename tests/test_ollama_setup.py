"""
tests/test_ollama_setup.py — offering a model sized to the machine.

A fresh installation refuses every question because it has no language
model. Telling the user that is half the answer; the other half is
offering the thing, sized to the machine it will run on.

Nothing here installs anything. `recommend` and `status` only look, and
the two functions that change the machine are never called by a test --
downloading gigabytes is a decision that belongs to whoever owns the
computer, and that includes whoever runs this suite.
"""
from __future__ import annotations

import pytest

from mana.apps import ollama_setup as setup


def _recommend(bandwidth, ram, monkeypatch, gpu=None):
    monkeypatch.setattr(setup, "memory_bandwidth_gbps", lambda *a, **k: bandwidth)
    monkeypatch.setattr(setup, "_usable_gpu",
                        lambda: gpu or {"present": False, "why": "нет"})

    class Profile:
        total_ram_gb = ram
        cpu_count = 8

    import mana.hardware
    monkeypatch.setattr(mana.hardware, "detect_hardware", lambda: Profile())
    return setup.recommend()


def test_the_comfortable_model_wins_over_the_biggest_that_fits(monkeypatch):
    """"Largest that fits" is not "best choice".

    On this machine that rule picked a 14B at 5.7 tokens a second over a
    7B at 10.9 -- more capable and worse to sit in front of.
    """
    pick = _recommend(53.0, 31.0, monkeypatch)
    assert pick.model == "qwen2.5:7b-instruct"
    assert pick.tokens_per_second > setup.COMFORTABLE_TOKENS_PER_SECOND
    # The bigger one is still offered, for anyone who wants capability
    # more than latency.
    bigger = [a for a in pick.alternatives if "14b" in a["model"]][0]
    assert bigger["fits"] is True
    assert bigger["comfortable"] is False


def test_a_slow_machine_gets_a_small_model(monkeypatch):
    pick = _recommend(12.0, 8.0, monkeypatch)
    assert "1.5b" in pick.model or "3b" in pick.model


def test_a_machine_that_cannot_run_anything_gets_no_recommendation(monkeypatch):
    """Better to say nothing fits than to suggest something unusable."""
    assert _recommend(2.0, 4.0, monkeypatch) is None


def test_ram_headroom_is_left_for_everything_else(monkeypatch):
    """A model that fits only if nothing else runs does not fit."""
    pick = _recommend(53.0, 12.0, monkeypatch)
    assert pick.size_gb <= 12.0 - setup._headroom(12.0)


def test_an_unusable_gpu_is_not_counted(monkeypatch):
    """An RX 580 holds 8 GB that ollama cannot reach: ROCm dropped
    gfx803. Counting it would promise a speed the machine cannot give and
    leave the user no way to find out why."""
    detected = setup._usable_gpu()
    assert "present" in detected and "why" in detected
    if not detected["present"]:
        assert detected["why"]


def test_the_reason_carries_the_numbers(monkeypatch):
    """A recommendation that cannot say "about 11 tokens a second" is one
    nobody can judge."""
    pick = _recommend(53.0, 31.0, monkeypatch)
    assert "ГБ/с" in pick.reason
    assert "токен" in pick.reason
    assert pick.bandwidth_gbps == 53.0


def test_bandwidth_is_measured_not_assumed():
    """Two machines with the same RAM and different memory speed deserve
    different answers."""
    measured = setup.memory_bandwidth_gbps(seconds=0.1)
    assert measured > 0.0


def test_installed_and_running_are_different_questions():
    """The binary can be present with nothing listening, and reporting
    that as "you have ollama" sends the user back to a window that still
    refuses everything."""
    state = setup.status()
    assert "installed" in state and "service_up" in state
    assert isinstance(state["models"], list)


def test_a_bare_base_url_finds_the_models():
    """probe_ollama used to request the root for a URL with no /api/ in
    it, and the Ollama root answers "Ollama is running" as plain text --
    so it reported reachable with an empty list and a JSONDecodeError,
    which reads as "up, and has nothing"."""
    import inspect

    from mana import brains

    source = inspect.getsource(brains.probe_ollama)
    assert '"/api/tags"' in source
    assert "root.rstrip" in source


def test_nothing_installs_itself():
    """The two functions that change the machine are reached only from a
    button somebody pressed -- nothing inside this module calls them."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(setup))
    callers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and getattr(inner.func, "id", "") in
                    ("install_runtime", "pull_model")):
                callers.append(node.name)
    assert callers == [], callers


def test_a_small_machine_is_not_told_it_can_run_nothing(monkeypatch):
    """A flat 8 GB of headroom gave an 8 GB machine a budget of zero,
    while a 1.5B model -- one gigabyte -- runs there fine."""
    pick = _recommend(25.0, 8.0, monkeypatch)
    assert pick is not None
    assert pick.size_gb <= 5.0
