"""
Phase 2: one plan per turn, and every PipelineSpec field has one owner.

The planner calls the functions the answer path used to call on its own,
so the first tests hold it to that: the route and graph it plans are the
ones the answer path then executes. The rest hold the genome to its new
boundary -- evolution moves the execution profile and nothing else -- and
make sure state written by an older version still loads.
"""
from __future__ import annotations

import pickle
from dataclasses import asdict, fields

import pytest

from mana.config import RandomManager
from mana.pipeline import (CAPABILITY, CONFIG, EVOLVABLE, EXECUTION, FIELD_ROLES,
                           ROUTING, PipelineFactory, PipelineSpec)

ANSWERED = [
    "Сколько будет 17 * 23? Ответь кратко.",
    "Какая сейчас последняя версия Python? Ответь кратко.",
    "Сравни актуальные версии Python и Java и объясни, какую выбрать.",
    "Расскажи про Git",
]


@pytest.mark.parametrize("task", ANSWERED)
def test_the_answer_path_carries_out_the_plan(isolated_agent, task):
    plan = isolated_agent.plan_task(task)
    assert plan.kind == "answer"
    result = isolated_agent.solve_task(task)
    assert result["plan"]["kind"] == "answer"
    assert result["plan"]["route"] == plan.route
    assert result["route"] == plan.route
    if plan.path == "adaptive":
        assert result["adaptive"]["graph"] == list(plan.graph)
    executed = result.get("trace", {}).get("brain_strategy")
    if executed:
        assert executed == plan.brain_strategy


def test_the_planned_route_is_the_one_the_router_would_choose(isolated_agent):
    spec = isolated_agent.pipeline
    for task in ANSWERED:
        assert isolated_agent.plan_task(task).route == \
            isolated_agent._effective_route(task, spec)


def test_an_instruction_to_a_program_is_planned_as_a_tool_call(isolated_agent,
                                                               monkeypatch):
    from mana.apps import intent as app_intent

    plan = isolated_agent.plan_task("Открой Notepad++")
    assert plan.kind == "app_action"
    assert plan.capability.startswith("tool:")

    called = []
    monkeypatch.setattr(app_intent, "perform",
                        lambda found, registry, goal="": called.append(found.tool)
                        or {"ok": False, "output": None, "error": "заглушка"})
    result = isolated_agent.solve_task("Открой Notepad++")
    assert called == [plan.capability.split(":", 1)[1]]
    assert result["plan"]["kind"] == "app_action"
    assert result["trace"]["performed"] is False


def test_a_request_to_remember_is_planned_as_memory(isolated_agent):
    plan = isolated_agent.plan_task("Запомни, что сервер 1С называется srv-01")
    assert plan.kind == "remember"
    result = isolated_agent.solve_task("Запомни, что сервер 1С называется srv-01")
    assert result["plan"]["kind"] == "remember"


def test_planning_changes_nothing(isolated_agent):
    before = isolated_agent.classify_route(ANSWERED[1])
    for task in ANSWERED:
        isolated_agent.plan_task(task)
    assert isolated_agent.classify_route(ANSWERED[1]) == before
    assert isolated_agent.history == []


# --------------------------------------------------------------------------
# the genome and its owners
# --------------------------------------------------------------------------

def test_every_field_has_exactly_one_owner():
    names = {f.name for f in fields(PipelineSpec)}
    assert set(FIELD_ROLES) == names
    assert set(FIELD_ROLES.values()) == {CAPABILITY, ROUTING, EXECUTION, CONFIG}
    assert "verification_mode" not in names and "web_provider" not in names


def _parent(cfg):
    spec = PipelineSpec(route_mode="web", brain_policy="fastest", use_web=False,
                        use_critic=False, memory_top_k=5, max_context_chars=4000,
                        decompose_mode="auto", architecture="deep", cost_budget=7.0)
    return spec.normalize(cfg)


def test_evolution_moves_only_the_execution_profile(isolated_config):
    rm = RandomManager(7)
    parent = _parent(isolated_config)
    kept = {name: value for name, value in asdict(parent).items()
            if name not in EVOLVABLE}
    moved = set()
    for _ in range(300):
        child = PipelineFactory.mutate(parent, rm, isolated_config, rate=0.9,
                                       max_changes=2)
        other = PipelineFactory.random(rm, isolated_config)
        mixed = PipelineFactory.crossover(parent, other, rm, isolated_config)
        fresh = PipelineFactory.random(rm, isolated_config, parent=parent)
        for candidate in (child, mixed, fresh):
            row = asdict(candidate)
            assert {name: row[name] for name in kept} == kept
            moved |= {name for name in EVOLVABLE if row[name] != asdict(parent)[name]}
    assert len(moved) >= 5              # the profile itself does evolve


def test_a_state_saved_with_retired_fields_still_loads(isolated_config):
    from mana import ManaAgent

    first = ManaAgent(isolated_config)
    first.pipeline.temperature = 0.33
    first._save_state()
    first.persistent_memory.close()
    first.experience.close()

    path = isolated_config.state_file
    with open(path, "rb") as handle:
        state = pickle.load(handle)
    state["pipeline"].update(verification_mode="always", web_provider="ddgs",
                             from_a_later_version=1)
    with open(path, "wb") as handle:
        pickle.dump(state, handle)

    again = ManaAgent(isolated_config)
    try:
        assert again.pipeline.temperature == pytest.approx(0.33)
    finally:
        again.persistent_memory.close()
        again.experience.close()


def test_old_experience_rows_are_read_not_dropped():
    old = dict(asdict(PipelineSpec()), verification_mode="never", web_provider="ddgs")
    assert PipelineSpec.from_dict(old).temperature == PipelineSpec().temperature
