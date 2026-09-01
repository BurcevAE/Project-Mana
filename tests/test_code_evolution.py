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
    """Uses a test-local CodeTarget under a throwaway id, so it never
    depends on what state the real whitelist entries happen to be in.

    The single test case is one the current implementation demonstrably
    fails, and the test asserts that precondition first -- if a future
    change makes it pass, this fails loudly on the precondition rather
    than quietly proving something else.
    """
    verifier = isolated_agent_exec_enabled.verifier
    real_target = ce.WHITELIST["local_fallback"]
    local_target = ce.CodeTarget(
        target_id="local_fallback_local_test",
        file_path=real_target.file_path, class_name=real_target.class_name,
        function_name=real_target.function_name, is_staticmethod=True,
        param_names=["task"], signature_hint=real_target.signature_hint,
        description="test-local target, not part of the real whitelist",
        test_cases=[ce.CodeTestCase("Что такое рекурсия простыми словами?", ["рекурс"])],
    )
    base = ce.baseline_score(local_target)
    assert base["failed"] == [0], (
        "precondition: _local_fallback must not already answer this")

    candidate = '''
def _local_fallback(task: str) -> str:
    t = task.lower()
    if "рекурс" in t:
        return "Рекурсия — это когда функция вызывает саму себя."
    return "Недостаточно данных для надёжного ответа без внешней модели."
'''.strip()
    ce.WHITELIST["local_fallback_local_test"] = local_target
    try:
        evaluation = ce.evaluate_candidate("local_fallback_local_test", candidate, verifier)
        decision = ce.decide(evaluation)
    finally:
        del ce.WHITELIST["local_fallback_local_test"]
    assert decision["accepted"] is True, decision
    assert decision["reason"] == "strict_improvement"


def test_noop_candidate_is_rejected_as_no_improvement(isolated_agent_exec_enabled):
    """Identical behaviour must never count as an accepted improvement.

    Uses the CURRENT implementation of the remaining whitelist target,
    reconstructed as a free function. (This test previously used
    `task_category`, which left the whitelist in 5.9.1 when it stopped
    being a pure function -- see the note in mana/code_evolution.py.)
    """
    import inspect
    import textwrap

    from mana.agent_parts.execution import ExecutionMixin
    verifier = isolated_agent_exec_enabled.verifier
    src = textwrap.dedent(inspect.getsource(ExecutionMixin._local_fallback))
    src = src.replace("@staticmethod\n", "")
    evaluation = ce.evaluate_candidate("local_fallback", src, verifier)
    decision = ce.decide(evaluation)
    assert decision["accepted"] is False
    assert decision["reason"] == "no_improvement"


def test_impure_functions_are_not_whitelisted():
    """`_task_category` now calls a module-level helper, so a candidate
    for it could not run standalone in the sandbox. A target that breaks
    the purity contract must leave the whitelist rather than have the
    contract loosened."""
    assert "task_category" not in ce.WHITELIST


def test_regressing_candidate_is_rejected(isolated_agent_exec_enabled):
    verifier = isolated_agent_exec_enabled.verifier
    candidate = 'def _local_fallback(task: str) -> str:\n    return "не знаю"'
    evaluation = ce.evaluate_candidate("local_fallback", candidate, verifier)
    decision = ce.decide(evaluation)
    assert decision["accepted"] is False
    assert decision["reason"] == "regression"
    assert len(decision["regressed_cases"]) > 0


def test_self_referencing_candidate_rejected_at_static_validation(isolated_agent_exec_enabled):
    verifier = isolated_agent_exec_enabled.verifier
    candidate = 'def _local_fallback(task: str) -> str:\n    return self.foo(task)'
    evaluation = ce.evaluate_candidate("local_fallback", candidate, verifier)
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
