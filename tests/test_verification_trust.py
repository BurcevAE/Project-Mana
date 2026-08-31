"""
tests/test_verification_trust.py — P0 #2: LLM-generated tests must not be
treated as independent proof of correctness.

The defect: verification was a single boolean, so "the model wrote code,
wrote its own tests, and passed them" produced exactly the same 0.95
quality credit and 1.0 confidence signal as "we evaluated the arithmetic
in the task ourselves". A model that writes a lenient test for its own
buggy code could therefore buy full trust with it -- and the routing and
experience learners then trained on that as ground truth.

The fix grades verification by WHO AUTHORED THE ORACLE, not by whether
the check passed.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from mana.agent_parts.execution import ExecutionMixin
from mana.pipeline import PipelineSpec


# --- trust classification ------------------------------------------------

@pytest.mark.parametrize("verification,expected", [
    ({"kind": "arithmetic", "verified": True}, ExecutionMixin.TRUST_INDEPENDENTLY_VERIFIED),
    ({"kind": "code", "mode": "generated_tests", "verified": True, "ok": True},
     ExecutionMixin.TRUST_MODEL_TESTED),
    ({"kind": "code", "mode": "generated_tests", "verified": False}, ExecutionMixin.TRUST_UNVERIFIED),
    ({"kind": "none", "verified": False}, ExecutionMixin.TRUST_UNVERIFIED),
    (None, ExecutionMixin.TRUST_UNVERIFIED),
    ({}, ExecutionMixin.TRUST_UNVERIFIED),
])
def test_trust_level_classification(verification, expected):
    assert ExecutionMixin.verification_trust_level(verification) == expected


def test_self_authored_tests_rank_strictly_below_independent_verification():
    """The core claim of this fix: passing your own tests is worth less
    than being checked against an oracle you did not write."""
    caps = ExecutionMixin.TRUST_QUALITY_CAP
    assert caps[ExecutionMixin.TRUST_MODEL_TESTED] < caps[ExecutionMixin.TRUST_INDEPENDENTLY_VERIFIED]
    assert caps[ExecutionMixin.TRUST_UNVERIFIED] < caps[ExecutionMixin.TRUST_MODEL_TESTED]


def test_model_tested_cannot_reach_the_old_095_credit():
    """Regression guard: the specific number the old code granted."""
    assert ExecutionMixin.TRUST_QUALITY_CAP[ExecutionMixin.TRUST_MODEL_TESTED] < 0.95


def test_model_tested_credit_stays_below_typical_confidence_threshold():
    """MODEL_TESTED must not, on its own, push an answer over the default
    confidence threshold -- otherwise self-certification would still end
    computation early. Note the threshold lives on PipelineSpec (the
    evolvable genome), not on Config."""
    from mana.pipeline import PipelineSpec
    assert (ExecutionMixin.TRUST_QUALITY_CAP[ExecutionMixin.TRUST_MODEL_TESTED]
            <= PipelineSpec().confidence_threshold + 0.01)


# --- the confidence signal is graded, not binary -------------------------

def _confidence_for(agent, verification):
    result = {"answer": "какой-то ответ", "verification": verification,
              "llm_ok": True, "critic_score": 0.0, "trace": {}, "latency": 0.1}
    spec = PipelineSpec(**asdict(agent.pipeline)).normalize(agent.config)
    return agent._evaluate_confidence_v41("Напиши функцию сложения", result, "local", spec, None)


def test_confidence_signal_grades_model_tested_below_independent(isolated_agent):
    independent = _confidence_for(isolated_agent, {"kind": "arithmetic", "verified": True})
    model_tested = _confidence_for(isolated_agent, {"kind": "code", "mode": "generated_tests",
                                                     "verified": True, "ok": True})
    assert independent["signals"]["verification"] == 1.0
    assert model_tested["signals"]["verification"] < 1.0
    assert model_tested["confidence"] < independent["confidence"], (
        "self-authored tests must yield strictly lower confidence than an independent check")


def test_unverified_signal_is_lowest(isolated_agent):
    unverified = _confidence_for(isolated_agent, {"kind": "none", "verified": False})
    model_tested = _confidence_for(isolated_agent, {"kind": "code", "mode": "generated_tests",
                                                     "verified": True, "ok": True})
    assert unverified["signals"]["verification"] < model_tested["signals"]["verification"]


# --- the level is surfaced, not hidden -----------------------------------

def test_trust_level_is_reported_in_the_adaptive_trace(isolated_agent_exec_enabled, monkeypatch):
    """Whoever reads a result must be able to tell WHICH kind of
    verification backed it, not just that 'verification happened'."""
    from mana import ManaAgent
    def fake_answer_core(self, task, spec, save_memory=True, context_tag=""):
        return {"task": task, "answer": "391", "latency": 0.01, "trace": {},
                "pipeline": asdict(spec), "critic_score": 0.0, "critic_trace": {},
                "llm_ok": True, "passes_used": 1, "timeout_count": 0,
                "fallback": False, "llm_latency": 0.01}
    monkeypatch.setattr(ManaAgent, "_answer_core", fake_answer_core)

    agent = isolated_agent_exec_enabled
    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.route_mode = "auto"; spec.architecture = "adaptive"; spec.compute_budget = 4
    spec = spec.normalize(agent.config)
    result = agent.answer("сколько будет 17 умножить на 23", spec=spec,
                           save_memory=False, context_tag="TEST")
    assert result["adaptive"]["verification_trust"] == ExecutionMixin.TRUST_INDEPENDENTLY_VERIFIED
    assert result["verification_trust"] == ExecutionMixin.TRUST_INDEPENDENTLY_VERIFIED


def test_routing_learner_is_not_trained_on_inflated_self_certified_quality(
        isolated_agent_exec_enabled, monkeypatch):
    """The actual harm pathway: proxy_quality feeds _record_route_outcome,
    which trains the router. If self-authored tests grant 0.95 there, the
    router learns from the model's own say-so as if it were ground truth.
    Captures what quality value would be recorded for a MODEL_TESTED run."""
    from mana import ManaAgent
    recorded = {}

    def fake_answer_core(self, task, spec, save_memory=True, context_tag=""):
        return {"task": task, "answer": "def add(a,b): return a+b", "latency": 0.01,
                "trace": {}, "pipeline": asdict(spec), "critic_score": 0.0,
                "critic_trace": {}, "llm_ok": True, "passes_used": 1,
                "timeout_count": 0, "fallback": False, "llm_latency": 0.01}

    def fake_autonomous_execute(self, task, current, spec, context_tag):
        # Simulate: the model wrote code AND its own tests, and they passed.
        return {"kind": "code", "mode": "generated_tests", "ok": True, "verified": True}

    def spy_record(self, task, route, quality, execution_success, web_ok, latency):
        recorded["quality"] = quality

    monkeypatch.setattr(ManaAgent, "_answer_core", fake_answer_core)
    monkeypatch.setattr(ManaAgent, "_autonomous_execute", fake_autonomous_execute)
    monkeypatch.setattr(ManaAgent, "_record_route_outcome", spy_record)

    agent = isolated_agent_exec_enabled
    agent._benchmark_learning = False
    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.route_mode = "auto"; spec.architecture = "minimal"
    spec.graph_nodes = ("LLM", "EXECUTE", "EVALUATE")
    spec.compute_budget = 4
    spec = spec.normalize(agent.config)

    agent.answer("Напиши функцию сложения на Python", spec=spec,
                 save_memory=False, context_tag="TEST")

    assert "quality" in recorded, "routing outcome was never recorded"
    assert recorded["quality"] <= ExecutionMixin.TRUST_QUALITY_CAP[ExecutionMixin.TRUST_MODEL_TESTED], (
        f"router trained on inflated quality {recorded['quality']} from self-authored tests")
    assert recorded["quality"] < 0.95


def test_trust_level_is_present_on_every_solve_task_result(isolated_agent):
    """The interactive loop now displays this to the user, so it must
    always exist -- including on runs where nothing was verified at all."""
    result = isolated_agent.solve_task("Расскажи про Git")
    assert result.get("verification_trust") == ExecutionMixin.TRUST_UNVERIFIED


def test_trust_level_is_independent_for_verifiable_arithmetic(isolated_agent_exec_enabled):
    result = isolated_agent_exec_enabled.solve_task("Сколько будет 17 умножить на 23?")
    assert result.get("verification_trust") == ExecutionMixin.TRUST_INDEPENDENTLY_VERIFIED
