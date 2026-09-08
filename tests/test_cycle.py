"""The whole loop, and the two things it must never do.

    реальный опыт → ошибка → исследование → изменение
                  → повторная ситуация → улучшенный результат

It must not claim a proven improvement it has not measured, and it must
not walk in a circle: a candidate already measured and rejected under
conditions that still hold is not the next experiment. Both were found by
running it over the real record before they were tested here.
"""
from __future__ import annotations

import pytest

from mana.cognition import cycle
from mana.cognition.findings import Finding, Ledger, measurement_of
from mana.core.gates import ACCEPTED, NOT_EVALUATED, REJECTED
from mana.journal import Episode
from mana.policy import Policy

TARGETS = ["UT11-ER", "Информационная база"]


def _episodes():
    """A record with a real failure in it: an action asked for, described
    instead of performed."""
    said = ("Чтобы запустить конфигуратор информационной базы, найдите "
            "ярлык 1С на рабочем столе и выберите режим Конфигуратор.")
    return [
        Episode("a", "s", 1.0,
                "я хочу поработать с 1С запусти конфигуратор "
                "информационной базы", said),
        Episode("b", "s", 2.0, "расскажи про налоги",
                "Развёрнутый содержательный ответ про налоги, по существу."),
        Episode("c", "s", 3.0, "спасибо", "Пожалуйста, обращайтесь."),
        Episode("d", "s", 4.0, "сколько будет два плюс два", "Четыре."),
    ]


# --------------------------------------------------------------------------
# the six stages
# --------------------------------------------------------------------------

def test_the_loop_walks_every_stage_on_a_record_with_a_failure(tmp_path, monkeypatch):
    from mana.apps import intent

    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    narrow = Policy.of(intent_verb_anywhere=False, intent_stem_match=False,
                       intent_verb_forms="imperative")
    ran = cycle.run(_episodes(), current=narrow,
                    ledger=Ledger(tmp_path / "f.jsonl"), targets=TARGETS)

    assert [s.name for s in ran.stages] == list(cycle.STAGES)
    assert all(s.reached for s in ran.stages), ran.describe()
    assert ran.verdict in (ACCEPTED, REJECTED, NOT_EVALUATED)


def test_an_empty_record_stops_at_the_first_stage(tmp_path):
    ran = cycle.run([], ledger=Ledger(tmp_path / "f.jsonl"), targets=[])
    assert ran.stage(cycle.EXPERIENCE).reached is False
    assert ran.proven is False
    assert "цикл начинается с работы" in ran.missing


def test_a_clean_record_stops_at_the_failure_stage(tmp_path):
    clean = [Episode("a", "s", 1.0, "сколько будет два плюс два", "Четыре."),
             Episode("b", "s", 2.0, "а три плюс три", "Шесть.")]
    ran = cycle.run(clean, ledger=Ledger(tmp_path / "f.jsonl"), targets=[])
    assert ran.stage(cycle.FAILURE).reached is False
    assert "исправный конец" in ran.missing


def test_the_result_is_written_to_the_ledger(tmp_path, monkeypatch):
    from mana.apps import intent

    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    ledger = Ledger(tmp_path / "f.jsonl")
    narrow = Policy.of(intent_verb_anywhere=False, intent_stem_match=False,
                       intent_verb_forms="imperative")
    ran = cycle.run(_episodes(), current=narrow, ledger=ledger, targets=TARGETS)

    assert ran.finding_id
    assert [f.finding_id for f in ledger.findings()] == [ran.finding_id]


def test_nothing_is_written_when_the_caller_says_not_to(tmp_path, monkeypatch):
    from mana.apps import intent

    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    ledger = Ledger(tmp_path / "f.jsonl")
    cycle.run(_episodes(), ledger=ledger, targets=TARGETS, record=False)
    assert ledger.findings() == []


# --------------------------------------------------------------------------
# it must not walk in a circle
# --------------------------------------------------------------------------

def test_a_settled_candidate_is_not_the_next_experiment(tmp_path, monkeypatch):
    """Measured on the real record: the loop picked the least bad of the
    candidates it had already rejected and re-ran it, pass after pass.
    `rank` sinks a settled candidate rather than hiding it, which is right
    for a reader and wrong for a loop."""
    from mana.apps import intent

    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    ledger = Ledger(tmp_path / "f.jsonl")
    narrow = Policy.of(intent_verb_anywhere=False, intent_stem_match=False,
                       intent_verb_forms="imperative")

    first = cycle.run(_episodes(), current=narrow, ledger=ledger,
                      targets=TARGETS)
    assert first.stage(cycle.INVESTIGATION).reached
    chosen = first.stage(cycle.INVESTIGATION).detail["chosen"]

    seen = {tuple(sorted(chosen.items()))}
    for _ in range(6):
        again = cycle.run(_episodes(), current=narrow, ledger=ledger,
                          targets=TARGETS)
        stage = again.stage(cycle.INVESTIGATION)
        if not stage.reached:
            assert "уже проведены" in again.missing
            break
        picked = tuple(sorted(stage.detail["chosen"].items()))
        assert picked not in seen, "the loop re-ran a settled candidate"
        seen.add(picked)
    else:
        pytest.fail("the loop never ran out of candidates and never stalled")


def test_the_stall_says_what_would_unstick_it(tmp_path, monkeypatch):
    from mana.apps import intent

    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    ledger = Ledger(tmp_path / "f.jsonl")
    narrow = Policy.of(intent_verb_anywhere=False, intent_stem_match=False,
                       intent_verb_forms="imperative")
    for _ in range(8):
        ran = cycle.run(_episodes(), current=narrow, ledger=ledger,
                        targets=TARGETS)
        if not ran.stage(cycle.INVESTIGATION).reached:
            break
    assert ("новый опыт" in ran.missing
            or "не оцениваются всухую" in ran.missing)


# --------------------------------------------------------------------------
# it must not claim what it has not measured
# --------------------------------------------------------------------------

def test_a_small_sample_is_never_a_proven_cycle(tmp_path, monkeypatch):
    from mana.apps import intent

    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    narrow = Policy.of(intent_verb_anywhere=False, intent_stem_match=False,
                       intent_verb_forms="imperative")
    ran = cycle.run(_episodes(), current=narrow,
                    ledger=Ledger(tmp_path / "f.jsonl"), targets=TARGETS)
    result = ran.stage(cycle.RESULT)
    assert "sample_size" in result.detail["verdict"]["failed_gates"]
    assert ran.proven is False
    assert "нужно ещё" in ran.missing


def test_the_verdict_comes_from_the_gates_and_not_from_here():
    import inspect

    source = inspect.getsource(cycle)
    # No second acceptance rule. Everything that decides lives in
    # core/gates.py, reached through failure_domain.judge_change.
    assert "judge_change" in source
    for smuggled in ("MIN_ABSOLUTE_MARGIN", "mcnemar", "ALPHA"):
        assert smuggled not in source
