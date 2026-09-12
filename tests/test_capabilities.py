"""
Phase 1: one place that says what the agent can do right now.

The registry is read-only. These tests hold it to the sources it reads:
a tool is listed with exactly the availability the tool reports, a brain
with exactly the readiness the pool reports, and nothing measured is
reported as zero.
"""
from __future__ import annotations


def _by_kind(rows, kind):
    return {row["name"]: row for row in rows if row["kind"] == kind}


def test_every_tool_is_listed_with_its_own_availability(isolated_agent):
    tools = _by_kind(isolated_agent.capabilities_status(), "tool")
    registered = isolated_agent.tools.list_tools()
    assert registered, "the agent has no tools at all"
    for row in registered:
        assert row["name"] in tools
        assert tools[row["name"]]["available"] == row["available"]


def test_every_brain_is_listed_with_its_readiness(isolated_agent):
    pool = isolated_agent.llm.pool
    brains = _by_kind(isolated_agent.capabilities_status(), "brain")
    assert set(brains) == set(pool.brains)
    for brain_id in pool.brains:
        assert brains[brain_id]["available"] == pool.ready(brain_id)


def test_a_brain_nobody_called_is_unmeasured_not_zero(isolated_agent):
    pool = isolated_agent.llm.pool
    for row in _by_kind(isolated_agent.capabilities_status(), "brain").values():
        if pool.health[row["name"]].calls == 0:
            assert row["measured"]["success_rate"] is None


def test_every_capability_has_a_stable_id(isolated_agent):
    rows = isolated_agent.capabilities_status()
    ids = [row["capability_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert all(":" in capability_id for capability_id in ids)


def test_asking_what_it_can_do_changes_nothing(isolated_agent):
    task = "Сколько будет 17 * 23? Ответь кратко."
    before = isolated_agent.solve_task(task)
    isolated_agent.capabilities_status()
    after = isolated_agent.solve_task(task)
    assert before["answer"] == after["answer"] == "391"
