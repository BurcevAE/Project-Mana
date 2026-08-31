"""
tests/test_verification_correction.py — P0 #2: a failed verification must
CORRECT the answer, not merely lower confidence.

Motivated by a real observed failure: asked "сколько будет 17 умножить на
23", MANA answered 401 and shipped it. Two separate defects caused that,
and both are covered here:
  1. word-form arithmetic ("умножить на") was unparseable, so the verifier
     returned kind:"none" and there was nothing to check against;
  2. even when the verifier DID refute an answer, the loop only recorded
     verified=False and returned the wrong answer anyway.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from mana.config import Config
from mana.pipeline import PipelineSpec
from mana.verifier import LocalVerifier


# --- 1. word-form arithmetic is now extractable -------------------------

@pytest.mark.parametrize("task,answer,expected_value,should_verify", [
    ("сколько будет 17 умножить на 23", "391", 391.0, True),
    ("сколько будет 17 умножить на 23", "ответ будет 401", 391.0, False),
    ("Сколько будет 144 разделить на 12?", "12", 12.0, True),
    ("Сколько будет 20 плюс 22?", "42", 42.0, True),
    ("Сколько будет 30 минус 8?", "22", 22.0, True),
    ("Сколько будет 5 в квадрате?", "25", 25.0, True),
    ("what is 6 multiplied by 7", "42", 42.0, True),
    ("Сколько будет 17 * 23?", "391", 391.0, True),  # symbolic form must keep working
])
def test_word_form_arithmetic_is_verifiable(task, answer, expected_value, should_verify):
    cfg = Config(); cfg.local_exec_enabled = True
    result = LocalVerifier(cfg).verify(task, answer, "math")
    assert result["kind"] == "arithmetic", f"should be verifiable: {task!r}"
    assert result["value"] == expected_value
    assert result["verified"] is should_verify


def test_non_arithmetic_text_is_not_falsely_matched():
    """Guard against the normalization over-reaching: ordinary prose must
    still produce kind:'none' rather than a bogus expression."""
    cfg = Config(); cfg.local_exec_enabled = True
    v = LocalVerifier(cfg)
    for task in ["Расскажи про Git", "Почему полезно разделять память и веса?",
                 "Объясни, что такое рекурсия"]:
        assert v.verify(task, "какой-то ответ", "general")["kind"] == "none", task


def test_normalization_leaves_unknown_phrasing_alone():
    norm = LocalVerifier._normalize_word_arithmetic("расскажи про историю Python")
    assert "Python" in norm or "python" in norm


# --- 2. a refuted answer gets corrected, not just noted ------------------

def _agent_answering(monkeypatch, wrong_answer: str):
    from mana import ManaAgent
    def fake_answer_core(self, task, spec, save_memory=True, context_tag=""):
        return {"task": task, "answer": wrong_answer, "latency": 0.01, "trace": {},
                "pipeline": asdict(spec), "critic_score": 0.0, "critic_trace": {},
                "llm_ok": True, "passes_used": 1, "timeout_count": 0,
                "fallback": False, "llm_latency": 0.01}
    monkeypatch.setattr(ManaAgent, "_answer_core", fake_answer_core)


def _adaptive_spec(agent):
    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.route_mode = "auto"
    spec.architecture = "adaptive"
    spec.compute_budget = 4
    return spec.normalize(agent.config)


def test_refuted_arithmetic_answer_is_corrected(isolated_agent_exec_enabled, monkeypatch):
    """The exact real-world failure: a wrong arithmetic answer must not
    reach the user unchanged."""
    agent = isolated_agent_exec_enabled
    _agent_answering(monkeypatch, "ответ будет 401")
    result = agent.answer("сколько будет 17 умножить на 23", spec=_adaptive_spec(agent),
                           save_memory=False, context_tag="TEST")
    assert "391" in result["answer"], f"wrong answer was not corrected: {result['answer']!r}"
    assert "401" not in result["answer"]
    verification = result["verification"]
    assert verification["corrected"] is True
    assert verification["corrected_from"] == "ответ будет 401"


def test_correct_answer_is_left_untouched(isolated_agent_exec_enabled, monkeypatch):
    """Correction must not fire when the answer already matches -- otherwise
    it would rewrite good answers into bare numbers."""
    agent = isolated_agent_exec_enabled
    _agent_answering(monkeypatch, "Получается 391, если перемножить.")
    result = agent.answer("сколько будет 17 умножить на 23", spec=_adaptive_spec(agent),
                           save_memory=False, context_tag="TEST")
    assert result["answer"] == "Получается 391, если перемножить."
    assert result["verification"].get("corrected") is not True


def test_no_correction_without_independent_ground_truth(isolated_agent_exec_enabled, monkeypatch):
    """Correction must never fire on a model's opinion -- only on a truth
    the verifier derived itself. A non-arithmetic task has none, so the
    answer must pass through untouched even though it is unverified."""
    agent = isolated_agent_exec_enabled
    _agent_answering(monkeypatch, "Git — это система контроля версий.")
    result = agent.answer("Расскажи про Git", spec=_adaptive_spec(agent),
                           save_memory=False, context_tag="TEST")
    assert result["answer"] == "Git — это система контроля версий."
    assert (result.get("verification") or {}).get("corrected") is not True


def test_correction_helper_returns_none_when_nothing_to_do(isolated_agent_exec_enabled):
    agent = isolated_agent_exec_enabled
    spec = _adaptive_spec(agent)
    current = {"answer": "391"}
    # already verified -> no action
    assert agent._correct_refuted_answer("t", current, {"kind": "arithmetic", "ok": True,
                                                         "value": 391.0, "verified": True}, spec, "T") is None
    # no ground truth -> no action
    assert agent._correct_refuted_answer("t", current, {"kind": "none", "verified": False}, spec, "T") is None
    # evaluation failed -> no action
    assert agent._correct_refuted_answer("t", current, {"kind": "arithmetic", "ok": False,
                                                         "verified": False}, spec, "T") is None
