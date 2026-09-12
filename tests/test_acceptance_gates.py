"""
Phase 4: one set of gates for everything that changes the agent.

Two loops really changed MANA's answers -- evolution of PipelineSpec and
patches to its own source -- and both judged their own candidates. These
tests hold them to core/gates: the old checks may still refuse, but only
the gate may accept, and an adoption leaves a committed transaction.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict

import pytest

from mana import code_evolution
from mana.agent_parts import evolution as evolution_mod
from mana.core import transaction
from mana.core.gates import Evidence, PairedOutcome
from mana.pipeline import PipelineSpec


# --------------------------------------------------------------------------
# the source says so
# --------------------------------------------------------------------------

def test_neither_loop_can_write_accepted_on_its_own():
    evolution_src = inspect.getsource(evolution_mod.EvolutionMixin.self_improve)
    assert "accepted = bool(old_ok and gate_verdict.accepted)" in evolution_src
    assert "accepted = True" not in evolution_src
    decide_src = inspect.getsource(code_evolution.decide)
    assert '"accepted": True' not in decide_src
    assert "verdict.accepted" in decide_src


# --------------------------------------------------------------------------
# code patches
# --------------------------------------------------------------------------

def _evaluation(total, base, cand):
    return {"ok": True, "target_id": "t",
            "baseline": {"passed": sorted(base), "failed": [], "total": total},
            "candidate": {"passed": sorted(cand), "failed": [], "total": total}}


def test_a_patch_on_many_cases_is_still_not_accepted_without_a_hidden_set():
    decision = code_evolution.decide(_evaluation(40, range(10), range(30)))
    assert decision["reason"] == "strict_improvement"
    assert decision["accepted"] is False
    assert "hidden_confirms" in decision["gate"]["failed_gates"]


def test_a_regressing_patch_is_rejected_by_the_gate():
    decision = code_evolution.decide(_evaluation(40, range(20), range(1, 30)))
    assert decision["reason"] == "regression"
    assert decision["accepted"] is False
    assert "counterexamples" in decision["gate"]["failed_gates"]


# --------------------------------------------------------------------------
# PipelineSpec evolution
# --------------------------------------------------------------------------

def _metrics(agent, quality):
    rows = [{"id": f"t{i}", "category": "math", "rep": 1, "score": quality,
             "latency": 1.0, "answer": "x", "llm_ok": True, "timeouts": 0,
             "fallback": False, "web": 0, "web_attempted": False,
             "adaptive_steps": 1, "adaptive_confidence": 0.8} for i in range(6)]
    return agent._metrics_from_rows(rows)


def _strong_evidence():
    outcomes = [PairedOutcome(f"d{i}", "arithmetic", i < 5, i < 35) for i in range(40)]
    return Evidence(paired_dev=outcomes, baseline_hidden=0.5, candidate_hidden=0.8,
                    counterexamples_sought=5, counterexamples_found=0)


def _weak_evidence():
    outcomes = [PairedOutcome(f"d{i}", "arithmetic", False, True) for i in range(8)]
    return Evidence(paired_dev=outcomes, counterexamples_sought=0)


@pytest.fixture
def staged(isolated_agent, monkeypatch):
    """A cycle in which the old checks pass: the candidate scores higher on
    the substring suite, with nothing slower or less reliable. What the
    gate sees is set by each test."""
    agent = isolated_agent
    agent.config.routing_gate_enabled = False
    champion = PipelineSpec(**asdict(agent.pipeline)).normalize(agent.config)
    candidate = PipelineSpec(**asdict(champion))
    candidate.temperature = min(1.0, champion.temperature + 0.2)
    candidate = candidate.normalize(agent.config)

    def control(spec=None):
        better = spec is not None and spec.temperature == candidate.temperature
        q = 0.8 if better else 0.5
        m = _metrics(agent, q)
        return {"train": m, "generalization": m, "holdout": m,
                "pipeline": asdict(spec or agent.pipeline), "timestamp": 0.0}

    monkeypatch.setattr(agent, "run_control_benchmark", control)
    monkeypatch.setattr(agent, "evolve_pipeline",
                        lambda budget=None: (candidate, 90.0, {}, control(candidate)["train"]))
    monkeypatch.setattr(agent, "_compare_to_champion",
                        lambda fit, metrics: (True, {"failed_gates": []}))
    monkeypatch.setattr(agent, "adaptive_benchmark",
                        lambda holdout=True, spec=None: {"route_accuracy": 1.0,
                                                         "execution_accuracy": 1.0,
                                                         "quality": 1.0})
    before, after = control(champion), control(candidate)
    ok, _ = agent._strict_acceptance(before, after, 90.0, 10.0)
    agent.best_pipeline_fitness = 10.0
    agent.best_metrics = {"train": before["train"], "generalization": before["generalization"],
                          "holdout": before["holdout"], "fitness": 10.0}
    assert ok, "precondition: the old checks alone would have accepted this"
    return agent, champion, candidate


def _last_transaction():
    rows = transaction.history(1)          # newest first, as dicts
    return rows[0] if rows else None


def test_what_the_old_checks_accepted_the_gate_can_refuse(staged, monkeypatch):
    agent, champion, candidate = staged
    monkeypatch.setattr(agent, "_gate_evidence", lambda a, b: _weak_evidence())
    agent.self_improve()
    assert agent.pipeline.temperature == champion.temperature      # not adopted
    txn = _last_transaction()
    assert txn is not None and txn["last_state"] == transaction.ROLLED_BACK


def test_the_gate_accepts_and_the_adoption_is_committed(staged, monkeypatch):
    agent, champion, candidate = staged
    monkeypatch.setattr(agent, "_gate_evidence", lambda a, b: _strong_evidence())
    agent.self_improve()
    assert agent.pipeline.temperature == candidate.temperature     # adopted
    txn = _last_transaction()
    assert txn is not None and txn["last_state"] == transaction.COMMITTED
    assert txn["claim_id"] == f"pipeline:{candidate.key()}"


def test_the_gate_evidence_uses_graded_tasks_and_the_hidden_set(isolated_agent):
    """Offline, both arms answer from the same fallback: the evidence is
    real, and it says nothing is better."""
    champion = isolated_agent.pipeline
    evidence = isolated_agent._gate_evidence(champion, champion)
    assert len(evidence.paired_dev) >= 30
    assert {o.domain for o in evidence.paired_dev} == {"arithmetic", "sequence", "code"}
    assert evidence.counterexamples_found == 0
