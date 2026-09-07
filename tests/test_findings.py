"""What was tried, and not trying it twice.

The load-bearing test is `test_the_same_experiment_is_recognised_later`.
Everything else here supports it: a ledger exists to answer one question
before work starts -- have we already done this -- and a ledger nobody
consults is a diary.
"""
from __future__ import annotations

import json
import time

import pytest

from mana.core.gates import ACCEPTED, REJECTED, NOT_EVALUATED
from mana.cognition.findings import Finding, Ledger, VERDICTS

QUESTION = "Может ли обученная оценка играть лучше подсчёта материала?"
APPROACH = {"domain": "chess", "model": "ridge", "features": 12}


def finding(verdict: str = REJECTED, **kwargs) -> Finding:
    row = {"question": QUESTION, "approach": APPROACH, "verdict": verdict,
           "measurement": {"games": 240, "score": 0.4708},
           "conditions": {"corpus_games": 1000, "depth": 2}}
    row.update(kwargs)
    return Finding(**row)


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def test_the_same_experiment_has_the_same_id():
    """An assigned id would make every repetition look like a new result,
    which is the failure this exists to prevent."""
    assert finding().finding_id == finding().finding_id


def test_a_different_approach_is_a_different_experiment():
    other = finding(approach={"domain": "chess", "model": "gradient_boosting"})
    assert other.finding_id != finding().finding_id


def test_the_question_is_matched_regardless_of_case_and_spacing():
    spaced = finding(question="  " + QUESTION.upper() + "  ")
    assert spaced.finding_id == finding().finding_id


def test_a_finding_must_answer_a_question():
    with pytest.raises(ValueError):
        finding(question="   ")


def test_only_the_three_verdicts_are_allowed():
    """The same three as the acceptance gates, on purpose."""
    assert set(VERDICTS) == {ACCEPTED, REJECTED, NOT_EVALUATED}
    with pytest.raises(ValueError):
        finding(verdict="СКОРЕЕ ДА")


@pytest.mark.parametrize("verdict", VERDICTS)
def test_a_negative_result_is_a_first_class_record(verdict):
    """"We tried this and it did not work" saves more time than the
    positive case, which is usually already visible in the behaviour."""
    assert finding(verdict).verdict == verdict


# --------------------------------------------------------------------------
# the lookup that saves the time
# --------------------------------------------------------------------------

def test_the_same_experiment_is_recognised_later(tmp_path):
    """The whole point, in the user's words: "если захочу вернуться,
    начну сначала и потрачу время на повторение"."""
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding())

    found = ledger.already_tried(QUESTION, APPROACH,
                                 {"corpus_games": 1000, "depth": 2})
    assert found is not None
    assert found["finding"]["verdict"] == REJECTED
    assert found["staleness"]["stale"] is False


def test_something_never_tried_comes_back_empty(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding())
    assert ledger.already_tried("совсем другой вопрос", APPROACH) is None


def test_changed_conditions_are_reported_not_hidden(tmp_path):
    """A finding is a prior, not a prohibition. A ledger that silently
    suppressed a re-run would hide the one case where re-running was the
    right call."""
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding())

    found = ledger.already_tried(QUESTION, APPROACH,
                                 {"corpus_games": 20000, "depth": 4})
    assert found is not None
    assert found["staleness"]["stale"] is True
    changed = found["staleness"]["changed"]
    assert changed["corpus_games"] == {"was": 1000, "now": 20000}
    assert changed["depth"] == {"was": 2, "now": 4}


def test_nothing_here_refuses_to_run_anything():
    """It reports and decides nothing. If this module ever gains a way to
    block an experiment, that has to be a deliberate change."""
    import inspect

    from mana.cognition import findings as module

    source = inspect.getsource(module)
    for blocking in ("raise Refuse", "return False, ", "skip_experiment",
                     "do_not_run"):
        assert blocking not in source


def test_other_approaches_to_the_same_question_are_offered(tmp_path):
    """"This exact thing, no -- but here are other ways somebody went at
    it" is usually more useful than a bare miss."""
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding())
    ledger.record(finding(approach={"domain": "chess", "model": "linear"},
                          verdict=NOT_EVALUATED))

    assert ledger.already_tried(QUESTION, {"domain": "chess",
                                           "model": "boosting"}) is None
    assert len(ledger.about(QUESTION)) == 2


# --------------------------------------------------------------------------
# the ledger on disk
# --------------------------------------------------------------------------

def test_a_finding_survives_a_round_trip(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    original = finding()
    ledger.record(original)
    read_back = ledger.findings()
    assert len(read_back) == 1
    assert read_back[0].as_dict() == original.as_dict()


def test_a_later_record_about_the_same_experiment_wins(tmp_path):
    """Appended rather than rewritten: the history of what was believed
    and when is worth having, and only `latest` has to care."""
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding(REJECTED, created=1000.0))
    ledger.record(finding(ACCEPTED, created=2000.0,
                          measurement={"games": 900, "score": 0.56}))

    assert len(ledger.findings()) == 2
    latest = ledger.latest()
    assert len(latest) == 1
    assert latest[0].verdict == ACCEPTED
    assert ledger.already_tried(QUESTION, APPROACH)["finding"]["verdict"] == ACCEPTED


def test_a_damaged_line_costs_one_finding_not_the_ledger(tmp_path):
    path = tmp_path / "findings.jsonl"
    ledger = Ledger(path)
    ledger.record(finding())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ not json\n")
    ledger.record(finding(approach={"model": "other"}))
    assert len(ledger.findings()) == 2


def test_an_unwritable_ledger_does_not_raise(tmp_path):
    """A record about work must not be the reason the work fails."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    ledger = Ledger(blocker / "sub" / "findings.jsonl")
    assert ledger.record(finding()) is False
    assert ledger.findings() == []


def test_stats_count_experiments_and_records_apart(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding(REJECTED, created=1.0))
    ledger.record(finding(ACCEPTED, created=2.0))
    ledger.record(finding(REJECTED, approach={"model": "other"}, created=3.0))

    stats = ledger.stats()
    assert stats["records"] == 3
    assert stats["experiments"] == 2
    assert stats["by_verdict"] == {ACCEPTED: 1, REJECTED: 1}


def test_the_chess_result_is_actually_on_disk():
    """Not a synthetic record: the real experiment from this session must
    be findable, or it went nowhere after all."""
    ledger = Ledger()
    about = [f for f in ledger.latest()
             if f.approach.get("domain") == "chess"]
    if not about:
        pytest.skip("шахматный эксперимент на этой машине не проводился")
    result = about[0]
    assert result.verdict == REJECTED
    assert result.measurement["games"] >= 240
    assert result.conditions["corpus_games"] >= 1000
