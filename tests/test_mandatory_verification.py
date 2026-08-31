"""
tests/test_mandatory_verification.py — P0 issue #1: confidence must never
be allowed to terminate computation before a mandatory verification stage
has actually run.

These are INTEGRATION tests on the real _adaptive_answer_v41 loop (not
unit tests of _next_graph_action), because the bug was precisely that
three *separate* early-exit paths in that loop could each skip EXECUTE:
the value-of-computation check, the EVALUATE branch, and the post-action
confidence check. Testing the action-picker alone would have missed all
three.
"""
from __future__ import annotations

from dataclasses import asdict

from mana.pipeline import PipelineSpec


def _arithmetic_spec(agent):
    spec = PipelineSpec(**asdict(agent.pipeline))
    # NOTE: route_mode must be "auto" -- ExecutionMixin.answer() only takes
    # the adaptive-compute path for "auto"; forced routes go through the
    # deterministic _answer_routed path, which has no adaptive loop at all
    # (and therefore no "adaptive" key in its result).
    spec.route_mode = "auto"
    spec.architecture = "adaptive"
    spec.compute_budget = 4
    return spec.normalize(agent.config)


def _force_max_confidence(agent, monkeypatch):
    """Make every confidence evaluation return 1.0, i.e. the LLM is
    maximally sure of itself. Before the fix this alone was enough to
    stop the loop at the first opportunity."""
    def _always_certain(self, task, current, route, spec, previous=None):
        return {"confidence": 1.0, "raw_confidence": 1.0, "signals": {"forced": True}}
    monkeypatch.setattr(type(agent), "_evaluate_confidence_v41", _always_certain)


def test_execute_still_runs_when_confidence_is_maximal(isolated_agent_exec_enabled, monkeypatch):
    """The core regression: arithmetic task + EXECUTE in the graph +
    confidence pinned at 1.0 => EXECUTE must STILL run."""
    agent = isolated_agent_exec_enabled
    _force_max_confidence(agent, monkeypatch)
    spec = _arithmetic_spec(agent)

    result = agent.answer("Сколько будет 17 * 23?", spec=spec, save_memory=False, context_tag="TEST")
    adaptive = result["adaptive"]

    assert "EXECUTE" in adaptive["graph"], "precondition: the graph must contain EXECUTE"
    assert "EXECUTE" in adaptive["required_nodes"], "arithmetic tasks must mark EXECUTE as required"
    assert adaptive["required_nodes_satisfied"] is True, (
        f"loop stopped with mandatory nodes pending (stop_reason={adaptive['stop_reason']})")
    executed = [a["node"] for a in adaptive["attempts"]]
    assert "EXECUTE" in executed, f"EXECUTE never ran; nodes executed were {executed}"


def test_confidence_cannot_be_the_stop_reason_while_verification_pending(isolated_agent_exec_enabled, monkeypatch):
    agent = isolated_agent_exec_enabled
    _force_max_confidence(agent, monkeypatch)
    spec = _arithmetic_spec(agent)
    result = agent.answer("Сколько будет 19 * 17?", spec=spec, save_memory=False, context_tag="TEST")
    adaptive = result["adaptive"]
    if adaptive["stop_reason"] in {"confidence_threshold", "value_of_computation"}:
        assert adaptive["required_nodes_satisfied"] is True, (
            f"stopped early on {adaptive['stop_reason']} with required nodes still pending: "
            f"{set(adaptive['required_nodes']) }")


def test_verification_actually_reached_for_arithmetic(isolated_agent_exec_enabled, monkeypatch):
    agent = isolated_agent_exec_enabled
    _force_max_confidence(agent, monkeypatch)
    spec = _arithmetic_spec(agent)
    result = agent.answer("Сколько будет 17 * 23?", spec=spec, save_memory=False, context_tag="TEST")
    assert result["adaptive"]["verification_kind"] == "arithmetic"


def test_execute_not_required_when_sandbox_unavailable(isolated_agent, monkeypatch):
    """Guard against the opposite failure: requiring a node the machine
    cannot run would make the loop burn its whole budget instead of
    answering. isolated_agent has local_exec_enabled=False."""
    agent = isolated_agent
    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.route_mode = "local"
    spec.architecture = "minimal"
    spec.graph_nodes = ("LLM", "EXECUTE", "EVALUATE")
    spec = spec.normalize(agent.config)
    required = agent._required_nodes_for_task("Напиши функцию сортировки на Python", spec,
                                               spec.graph_nodes, "local")
    assert "EXECUTE" not in required, "must not require a stage the sandbox can't perform"


def test_no_required_nodes_when_execute_absent_from_graph(isolated_agent_exec_enabled):
    agent = isolated_agent_exec_enabled
    spec = PipelineSpec(graph_nodes=("LLM", "EVALUATE")).normalize(agent.config)
    required = agent._required_nodes_for_task("Сколько будет 2+2?", spec, spec.graph_nodes, "local")
    assert required == set(), "cannot require a node the graph doesn't contain"


def test_non_verifiable_task_has_no_mandatory_execute(isolated_agent_exec_enabled):
    agent = isolated_agent_exec_enabled
    spec = PipelineSpec(graph_nodes=("LLM", "EXECUTE", "EVALUATE")).normalize(agent.config)
    required = agent._required_nodes_for_task("Расскажи о своих впечатлениях от прогулки", spec,
                                               spec.graph_nodes, "local")
    assert "EXECUTE" not in required
