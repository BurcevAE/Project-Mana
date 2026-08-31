"""
tests/test_code_evolution.py — the gated self-improvement gate's decision
logic (evaluate_candidate/decide). Deliberately does NOT call apply_patch
or rollback here: those mutate real files under mana/agent_parts/ (the
WHITELIST's file_path is relative to the installed package, not
redirectable via Config). That end-to-end path is covered separately and
manually in scripts/manual_code_evolution_apply_test.py, which always
rolls back at the end. Requires local_exec_enabled (real sandbox), but no
LLM and no network.
"""
from __future__ import annotations

from mana import code_evolution as ce


def test_list_targets_matches_whitelist():
    targets = ce.list_targets()
    ids = {t["target_id"] for t in targets}
    assert ids == set(ce.WHITELIST.keys())
    for t in targets:
        assert t["test_cases"] > 0


def test_baseline_score_shape():
    base = ce.baseline_score(ce.WHITELIST["local_fallback"])
    assert set(base.keys()) == {"passed", "failed", "total"}
    assert base["total"] == len(ce.WHITELIST["local_fallback"].test_cases)
    assert len(base["passed"]) + len(base["failed"]) == base["total"]


def test_genuine_improvement_is_accepted(isolated_agent_exec_enabled):
    """Uses a test-local CodeTarget (registered under a throwaway id, never
    touching the real WHITELIST entries) so this test is robust regardless
    of whatever state the real local_fallback/task_category targets
    happen to be in after previous fixes. Targets a category
    (_task_category has no branch that ever returns "safety") and asserts
    its own precondition, so if a future change happens to add that
    branch, this test fails loudly on the precondition instead of
    silently proving something different than intended."""
    verifier = isolated_agent_exec_enabled.verifier
    real_target = ce.WHITELIST["task_category"]
    local_target = ce.CodeTarget(
        target_id="task_category_local_test",
        file_path=real_target.file_path, class_name=real_target.class_name,
        function_name=real_target.function_name, is_staticmethod=False,
        param_names=["task"], signature_hint=real_target.signature_hint,
        description="test-local target, not part of the real whitelist",
        test_cases=[ce.CodeTestCase("вопрос про безопасность хеширования паролей", ["safety"])],
    )
    base = ce.baseline_score(local_target)
    assert base["failed"] == [0], "precondition: _task_category must not already classify this as 'safety'"

    candidate = '''
def _task_category(task: str) -> str:
    t = task.lower()
    if "безопасност" in t: return "safety"
    if any(x in t for x in ["python", "код", "программ", "sql", "1с", "git", "тест"]): return "programming"
    if any(x in t for x in ["сколько", "вычисл", "математ", "процент", "арифмет", "умнож", "раздели"]): return "math"
    if any(x in t for x in ["почему", "сравни", "логик", "объясни"]): return "reasoning"
    if any(x in t for x in ["сегодня", "сейчас", "актуаль", "последн", "новост", "цена", "текущ"]): return "current"
    return "general"
'''.strip()
    ce.WHITELIST["task_category_local_test"] = local_target
    try:
        evaluation = ce.evaluate_candidate("task_category_local_test", candidate, verifier)
        decision = ce.decide(evaluation)
    finally:
        del ce.WHITELIST["task_category_local_test"]
    assert decision["accepted"] is True
    assert decision["reason"] == "strict_improvement"


def test_noop_candidate_is_rejected_as_no_improvement(isolated_agent_exec_enabled):
    verifier = isolated_agent_exec_enabled.verifier
    target = ce.WHITELIST["task_category"]
    # Reconstruct the CURRENT implementation verbatim as the "candidate" --
    # identical behavior must never count as an accepted improvement,
    # regardless of what that implementation currently is.
    import inspect
    import textwrap
    from mana.agent_parts.routing import RoutingMixin
    src = textwrap.dedent(inspect.getsource(RoutingMixin._task_category))
    src = src.replace("def _task_category(self, task: str)", "def _task_category(task: str)")
    evaluation = ce.evaluate_candidate("task_category", src, verifier)
    decision = ce.decide(evaluation)
    assert decision["accepted"] is False
    assert decision["reason"] == "no_improvement"


def test_regressing_candidate_is_rejected(isolated_agent_exec_enabled):
    verifier = isolated_agent_exec_enabled.verifier
    candidate = 'def _task_category(task: str) -> str:\n    return "general"'
    evaluation = ce.evaluate_candidate("task_category", candidate, verifier)
    decision = ce.decide(evaluation)
    assert decision["accepted"] is False
    assert decision["reason"] == "regression"
    assert len(decision["regressed_cases"]) > 0


def test_self_referencing_candidate_rejected_at_static_validation(isolated_agent_exec_enabled):
    verifier = isolated_agent_exec_enabled.verifier
    candidate = 'def _task_category(task: str) -> str:\n    return self.foo(task)'
    evaluation = ce.evaluate_candidate("task_category", candidate, verifier)
    assert evaluation["ok"] is False
    assert "self" in evaluation["reason"]


def test_wrong_function_name_rejected(isolated_agent_exec_enabled):
    verifier = isolated_agent_exec_enabled.verifier
    candidate = 'def totally_different(task: str) -> str:\n    return "x"'
    evaluation = ce.evaluate_candidate("task_category", candidate, verifier)
    assert evaluation["ok"] is False


def test_unknown_target_id_rejected():
    evaluation = ce.evaluate_candidate("not_a_real_target", "def x(): pass", verifier=None)
    assert evaluation["ok"] is False
    assert "unknown target_id" in evaluation["reason"]
