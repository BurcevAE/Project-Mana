"""
tests/test_core_measurement.py — the measurement layer and its boundary.

These tests protect the property everything else rests on: that MANA
cannot improve itself by editing, reading or wearing out the thing that
declares it improved. Each one names the specific way that could happen.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from mana.core import CORE_ROOT, is_immutable_path, oracle, splits, tasks


@pytest.fixture(autouse=True)
def _fresh_budget():
    splits._reset_for_tests()
    yield
    splits._reset_for_tests()


# ---------------------------------------------------------------------------
# the boundary
# ---------------------------------------------------------------------------

def test_core_is_protected_by_directory_not_by_a_list_of_names():
    """The name list had to be edited by hand for every new boundary
    module -- protection that erodes. A file added to core is protected
    the moment it exists."""
    assert is_immutable_path(CORE_ROOT / "a_module_added_tomorrow.py")
    assert is_immutable_path(CORE_ROOT / "nested" / "deeper.py")
    assert not is_immutable_path(CORE_ROOT.parent / "pipeline.py")


def test_traversal_cannot_disguise_a_core_path_as_an_outside_one():
    disguised = CORE_ROOT / ".." / "core" / "oracle.py"
    assert is_immutable_path(disguised)


def test_a_whitelist_target_inside_core_is_rejected_at_declaration():
    """Not only when a patch is applied: the mistake must surface at
    import time, not at the moment a patch is about to be written."""
    from mana.code_evolution import CodeTarget
    with pytest.raises(ValueError, match="immutable boundary"):
        CodeTarget(target_id="x", file_path="core/oracle.py", class_name="C",
                   function_name="f", is_staticmethod=True, param_names=[],
                   signature_hint="def f():", description="")


def test_apply_patch_refuses_a_core_file_even_if_one_got_through():
    from mana import code_evolution
    reason = code_evolution._is_forbidden_target(code_evolution.PACKAGE_ROOT / "core" / "splits.py")
    assert "immutable boundary" in reason


# ---------------------------------------------------------------------------
# generated tasks carry computed truth
# ---------------------------------------------------------------------------

def test_every_domain_generates_and_is_self_consistent():
    for domain in tasks.DOMAINS:
        produced = tasks.generate(domain, 12, seed=7)
        assert len(produced) == 12
        assert all(t.domain == domain for t in produced)
        assert all(t.checker in tasks.CHECKERS for t in produced)
        assert all(t.answer is not None or t.checker == "code_tests" for t in produced)


def test_generation_is_deterministic_for_a_seed():
    """A hidden split that changed between runs would be unusable as a
    gate -- and would have to be stored somewhere an agent could read."""
    a = tasks.generate("arithmetic", 8, seed=42)
    b = tasks.generate("arithmetic", 8, seed=42)
    c = tasks.generate("arithmetic", 8, seed=43)
    assert [t.prompt for t in a] == [t.prompt for t in b]
    assert [t.prompt for t in a] != [t.prompt for t in c]


def test_arithmetic_answers_are_exact_not_floating():
    """The verifier used to compute in float and report 3e+18 as correct.
    Generated ground truth must not repeat that."""
    for t in tasks.generate("arithmetic", 30, seed=11):
        assert isinstance(t.answer, int)
        assert eval(t.metadata["expression"], {"__builtins__": {}}, {}) == t.answer


def test_logic_puzzles_are_consistent_by_construction():
    """Constraints are derived from a drawn order, so an instance cannot be
    unsolvable or ambiguous -- an ambiguous task silently punishes a
    correct answer."""
    for t in tasks.generate("logic", 20, seed=3):
        order = t.metadata["order"]
        assert t.answer in order
        assert len(set(order)) == len(order)


def test_public_view_never_carries_the_answer():
    for domain in tasks.DOMAINS:
        for t in tasks.generate(domain, 5, seed=5):
            public = t.public()
            assert "answer" not in public
            assert "metadata" not in public, "metadata leaks the rule and the expected order"
            assert public["prompt"] == t.prompt


def test_difficulty_sweeps_the_range_rather_than_clustering():
    produced = tasks.generate("sequence", 10, seed=9, difficulty_range=(0.1, 0.9))
    assert produced[0].difficulty < 0.3
    assert produced[-1].difficulty > 0.7


# ---------------------------------------------------------------------------
# grading is exact and never consults a model
# ---------------------------------------------------------------------------

def test_correct_number_in_the_requested_format_passes():
    t = tasks.generate("arithmetic", 1, seed=1)[0]
    assert oracle.grade(t, str(t.answer)).correct


def test_the_same_number_written_as_a_float_still_passes():
    t = tasks.generate("arithmetic", 1, seed=1)[0]
    assert oracle.grade(t, f"{t.answer}.0").correct


def test_prose_around_the_number_is_a_format_failure_not_a_pass():
    """The prompt asked for one number without explanation. A lenient
    extractor would let a model emit several candidates and have the
    grader pick the flattering one."""
    t = tasks.generate("arithmetic", 1, seed=1)[0]
    g = oracle.grade(t, f"Думаю, получается {t.answer}, но可能 и {t.answer + 1}")
    assert g.correct is False
    assert g.reason == "format"


def test_a_wrong_number_is_wrong_not_a_format_problem():
    t = tasks.generate("arithmetic", 1, seed=1)[0]
    g = oracle.grade(t, str(t.answer + 1))
    assert g.correct is False and g.reason == "wrong"


def test_text_answers_require_the_single_word_that_was_asked_for():
    t = tasks.generate("logic", 1, seed=2)[0]
    assert oracle.grade(t, t.answer).correct
    assert oracle.grade(t, f"На этой позиции стоит {t.answer}").reason == "format"


def test_sequence_answers_compare_in_order():
    t = tasks.Task("x", "text_ops", "p", ["дом", "лес", "сад"], "sequence", 0.5)
    assert oracle.grade(t, "дом, лес, сад").correct
    assert not oracle.grade(t, "лес, дом, сад").correct


def test_a_grader_error_is_reported_rather_than_raised():
    """A grader that can crash takes down the experiment that called it."""
    broken = tasks.Task("x", "arithmetic", "p", object(), "number", 0.5)
    g = oracle.grade(broken, "5")
    assert g.correct is False


def test_ungradable_is_counted_apart_from_wrong():
    """Without a sandbox, code tasks cannot be graded. Counting them as
    failures would make a missing dependency look like a capability
    deficit -- a measurement error disguised as a finding."""
    code_tasks = tasks.generate("code", 3, seed=4)
    summary = oracle.score(code_tasks, ["def f(): pass"] * 3, verifier=None)
    assert summary["ungradable"] == 3
    assert summary["graded"] == 0
    assert summary["accuracy"] == 0.0


def test_code_tasks_are_graded_by_running_them(isolated_config):
    from mana.verifier import LocalVerifier
    isolated_config.local_exec_enabled = True
    verifier = LocalVerifier(isolated_config)
    task = next(t for t in tasks.generate("code", 20, seed=6)
                if t.metadata["function"] == "sum_even")
    good = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n"
    bad = "def sum_even(numbers):\n    return sum(numbers)\n"
    assert oracle.grade(task, good, verifier).correct is True
    assert oracle.grade(task, bad, verifier).correct is False


# ---------------------------------------------------------------------------
# the hidden holdout cannot be read
# ---------------------------------------------------------------------------

def test_there_is_no_way_to_obtain_the_hidden_tasks():
    """The old holdout was a function returning the tasks, and
    code_evolution read it as part of its regression oracle. The whole
    design here is that no such function exists."""
    exported = [name for name in dir(splits) if not name.startswith("_")]
    assert "hidden_tasks" not in exported
    assert "transfer_tasks" not in exported
    for name in exported:
        attr = getattr(splits, name)
        if callable(attr) and "hidden" in name:
            assert name in {"hidden_score"}, f"{name} may expose the hidden set"


def test_hidden_score_returns_numbers_and_no_task_content():
    seen = []

    def answer_fn(public_task):
        seen.append(public_task)
        return "0"

    result = splits.hidden_score(answer_fn, per_domain=2)
    payload = result.as_dict()
    assert set(payload) >= {"accuracy", "graded", "by_domain", "evaluations_left"}
    assert all("answer" not in t for t in seen), "the answer must never reach the caller"
    assert not any("prompt" in str(v) for v in payload.values() if isinstance(v, str))


def test_hidden_and_transfer_are_different_task_populations():
    """Transfer must not be a paraphrase of hidden, or passing it would
    mean nothing."""
    hidden_domains, transfer_domains = [], []
    splits.hidden_score(lambda t: hidden_domains.append(t["domain"]) or "0", per_domain=2)
    splits.transfer_score(lambda t: transfer_domains.append(t["domain"]) or "0", per_domain=2)
    assert set(hidden_domains).isdisjoint(set(transfer_domains))


def test_the_hidden_budget_is_enforced_by_raising_not_by_degrading():
    """A gate that quietly stops measuring is worse than one that stops
    working: the run continues and the numbers keep being believed."""
    for _ in range(3):
        splits.hidden_score(lambda t: "0", per_domain=1, budget=3)
    with pytest.raises(splits.HiddenBudgetExceeded):
        splits.hidden_score(lambda t: "0", per_domain=1, budget=3)


def test_every_hidden_evaluation_is_recorded_for_audit():
    """How many times the holdout was consulted before something was
    accepted matters as much as the score."""
    splits.hidden_score(lambda t: "0", per_domain=1, label="baseline")
    splits.transfer_score(lambda t: "0", per_domain=1, label="candidate")
    log = splits.audit_log()
    assert [e["label"] for e in log] == ["baseline", "candidate"]
    assert all("accuracy" in e for e in log)


def test_a_caller_that_crashes_on_one_task_does_not_lose_the_evaluation():
    calls = {"n": 0}

    def flaky(public_task):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("brain died")
        return "0"

    result = splits.hidden_score(flaky, per_domain=2)
    assert result.graded > 0


def test_the_measurement_is_large_enough_to_measure_with():
    """The set this replaces had 21 tasks total, and acceptance gates
    computed z-scores over it."""
    assert len(splits.train_tasks()) >= 100
    assert len(splits.dev_tasks()) >= 50


def test_a_refused_answer_is_not_free_on_the_hidden_set(isolated_config):
    """`accuracy` excludes ungradable answers so a missing sandbox does
    not read as a capability deficit -- right for diagnosis, but it makes
    refusing free. `strict_accuracy` counts everything attempted, which
    is what a comparison between two systems needs."""
    from mana.core import splits

    def refuses_half(public):
        return "42" if public["domain"] == "arithmetic" else ""

    result = splits.hidden_score(refuses_half, per_domain=3, label="test-strict")
    assert result.attempted > result.graded, "the refusals must be counted somewhere"
    assert result.strict_accuracy <= result.accuracy
    assert set(result.strict_by_domain) == set(result.by_domain)


def test_strict_and_lenient_agree_when_nothing_was_refused(isolated_config):
    from mana.core import splits
    result = splits.hidden_score(lambda public: "0", per_domain=2,
                                 label="test-nothing-refused")
    if result.ungradable == 0:
        assert result.strict_accuracy == result.accuracy
