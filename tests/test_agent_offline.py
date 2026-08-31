"""
tests/test_agent_offline.py — end-to-end agent behavior with LLM and web
both disabled (the fallback/local path). No network, no LLM required --
safe to run anywhere, including CI.
"""
from __future__ import annotations

import pytest


def test_solve_task_arithmetic_fallback(isolated_agent):
    result = isolated_agent.solve_task("Сколько будет 17 * 23? Ответь кратко.")
    assert "391" in result["answer"]
    assert result["fallback"] is True  # no LLM available -> the deterministic fallback answered


def test_solve_task_writes_to_graph_memory(isolated_agent):
    isolated_agent.solve_task("Расскажи про Git")
    status = isolated_agent.graph_memory_status()
    assert status["turn_nodes"] >= 1


def test_build_context_surfaces_graph_memory_for_related_later_question(isolated_agent):
    isolated_agent.graph_memory.record_turn(
        isolated_agent.session_id, "Расскажи про Git",
        "Git — распределённая система контроля версий с полной историей коммитов.")
    for u, a in [("Погода?", "Солнечно."), ("Ужин?", "Паста."), ("Матч?", "Победа гостей."), ("Книга?", "Роман.")]:
        isolated_agent.graph_memory.record_turn(isolated_agent.session_id, u, a)

    context, trace = isolated_agent._build_context("Как в Git устроены ветки?", isolated_agent.pipeline)
    assert trace["graph_memory"] > 0
    assert "[GRAPH MEMORY]" in context
    assert "распределённая система контроля версий" in context


def test_adaptive_answer_loop_completes_with_verification(isolated_agent_exec_enabled):
    from dataclasses import asdict
    from mana.pipeline import PipelineSpec
    spec = PipelineSpec(**asdict(isolated_agent_exec_enabled.pipeline))
    spec.route_mode = "auto"
    spec.compute_budget = 3
    spec = spec.normalize(isolated_agent_exec_enabled.config)

    result = isolated_agent_exec_enabled.answer(
        "Сколько будет 17 * 23? Ответь кратко.", spec=spec, save_memory=False, context_tag="TEST")
    assert "391" in result["answer"]
    assert result["adaptive"]["verification_used"] is True
    assert result["adaptive"]["verification_kind"] == "arithmetic"


def test_verify_cli_tool_via_registry(isolated_agent):
    result = isolated_agent.tools.call("verify_arithmetic", expression="9*9")
    assert result.output["value"] == 81.0


@pytest.mark.slow
def test_offline_self_improve_cycle_does_not_crash(isolated_agent):
    isolated_agent.config.strategy_generations = 1
    isolated_agent.config.strategy_population = 2
    isolated_agent.config.evolution_workers = 1
    report = isolated_agent.self_improve()
    assert "decision" in report or "cycle" in report


@pytest.mark.slow
def test_routing_benchmark_does_not_crash(isolated_agent):
    result = isolated_agent.routing_benchmark()
    assert isinstance(result, dict)


def test_hardware_status_and_tools_status_shapes(isolated_agent):
    hw = isolated_agent.hardware_status()
    assert "profile" in hw and "adapted" in hw and "auto_adapt_enabled" in hw
    tools = isolated_agent.tools_status()
    assert len(tools["tools"]) == 9
