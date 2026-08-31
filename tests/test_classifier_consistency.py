"""
tests/test_classifier_consistency.py — two independent keyword lists decide
whether a task is "programming", and they must agree.

Found on real hardware: asked "напиши функцию, которая проверяет, является
ли число простым", MANA reported "не проверено" even with the sandbox
enabled. Root cause was NOT the verification logic -- it was a
disagreement between two classifiers:

  * _task_category()          (routing.py)   -> decides the compute GRAPH
  * _is_code_verifiable_task() (execution.py) -> decides the verification POLICY

_is_code_verifiable_task knew "функци"/"алгоритм" were programming;
_task_category did not. So the policy said "code_tests" while EXECUTE was
never placed in the graph, and the generated-tests path could not run at
all. The lists live in two files (and _task_category is a code_evolution
patch target that must stay a pure self-free function, so it cannot import
a shared constant) -- this test is what keeps them from drifting apart
again.
"""
from __future__ import annotations

import pytest

PROGRAMMING_REQUESTS = [
    "напиши функцию, которая проверяет, является ли число простым",
    "напиши код для сортировки списка",
    "реализуй функцию быстрой сортировки",
    "напиши алгоритм поиска в ширину",
    "исправь код этой программы",
    "напиши python скрипт для чтения файла",
]

NON_PROGRAMMING_REQUESTS = [
    "какая сегодня погода",
    "сколько будет 17 умножить на 23",
    "расскажи о своих впечатлениях",
]


def test_known_issue_tool_name_in_a_history_question_is_misclassified(isolated_agent_exec_enabled):
    """KNOWN, PRE-EXISTING defect -- documented, not fixed here.

    "расскажи про историю Git" is classified as `programming` purely
    because the substring "git" appears, even though it is a history
    question with no code in it. This predates the keyword broadening in
    this change (the original list already contained "git"), and the same
    trap applies to "python", "sql" and "1с" inside ordinary prose.

    The consequence is over-eager verification rather than a wrong answer:
    MANA may schedule EXECUTE and try to generate tests for an answer that
    contains no code. In the observed run it still correctly reported "не
    проверено", so the impact is wasted work, not a false trust claim.

    Fixing it properly means intent detection ("напиши"/"реализуй" vs
    "расскажи про"), not more substrings -- a separate change with its own
    benchmark check. This test pins the CURRENT behaviour so the fix is
    deliberate and visible when it happens, instead of the issue silently
    persisting.
    """
    agent = isolated_agent_exec_enabled
    assert agent._task_category("расскажи про историю Git") == "programming", (
        "if this now returns something else, the known issue was fixed -- "
        "update this test to assert the corrected behaviour")


@pytest.mark.parametrize("task", PROGRAMMING_REQUESTS)
def test_graph_builder_and_verification_policy_agree_on_programming(isolated_agent_exec_enabled, task):
    """Whenever the verifier believes a task is code-verifiable, the
    classifier that builds the graph must also call it programming --
    otherwise EXECUTE never gets scheduled and the check cannot run."""
    agent = isolated_agent_exec_enabled
    answer = "```python\ndef f(n):\n    return n\n```"
    if agent._is_code_verifiable_task(task, answer):
        assert agent._task_category(task) == "programming", (
            f"{task!r}: verifier says code-verifiable but _task_category says "
            f"{agent._task_category(task)!r} -- EXECUTE will never be scheduled")


@pytest.mark.parametrize("task", PROGRAMMING_REQUESTS)
def test_programming_task_gets_execute_in_its_graph(isolated_agent_exec_enabled, task):
    """The end-to-end consequence: a code request must produce a graph that
    actually contains the verification stage."""
    from dataclasses import asdict
    from mana.pipeline import PipelineSpec
    agent = isolated_agent_exec_enabled
    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.route_mode = "auto"
    spec.architecture = "adaptive"
    spec = spec.normalize(agent.config)
    route = agent._effective_route(task, spec)
    graph = agent._graph_for_task(task, spec, route)
    assert "EXECUTE" in graph, f"{task!r} -> graph {graph} has no verification stage"


@pytest.mark.parametrize("task", NON_PROGRAMMING_REQUESTS)
def test_non_programming_requests_are_not_misclassified(isolated_agent_exec_enabled, task):
    """Guard the other direction: broadening the keyword list must not turn
    ordinary questions into programming tasks."""
    assert isolated_agent_exec_enabled._task_category(task) != "programming", task


def test_benchmark_category_accuracy_not_regressed(isolated_agent_exec_enabled):
    """The keyword list is also graded by BenchmarkSuite's own labels (it is
    a code_evolution target). Broadening it must not lose accuracy there."""
    from mana.pipeline import BenchmarkSuite
    agent = isolated_agent_exec_enabled
    tasks = (BenchmarkSuite.train_tasks() + BenchmarkSuite.generalization_tasks()
             + BenchmarkSuite.holdout_tasks())
    hits = sum(1 for t in tasks if agent._task_category(t.query) == t.category)
    assert hits >= 15, f"category accuracy regressed to {hits}/{len(tasks)}"
