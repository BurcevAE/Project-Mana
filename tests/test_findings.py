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
    # `trials` rather than `games`: the canonical key a failure class can
    # be derived from, counting independent units.
    assert result.measurement["trials"] >= 240
    assert result.conditions["corpus_games"] >= 1000


# --------------------------------------------------------------------------
# the failure class is derived, and a guess is kept apart from it
#
# `"type": "NO_GENERALIZATION"` written onto the chess result would have
# been an interpretation recorded as a fact. Reasoning built on labels a
# system assigns itself is reasoning built on nothing.
# --------------------------------------------------------------------------

from mana.cognition.findings import (  # noqa: E402
    BETTER, COSTS_MORE_THAN_IT_GAINS, NOT_BETTER, NOT_MEASURED, UNCLASSIFIED,
    WORSE, COST_TOLERANCE, classify, measurement_of)


@pytest.mark.parametrize("verdict,measurement,expected", [
    (REJECTED, measurement_of(240, [0.376, 0.533], 0.5, 6.2), NOT_BETTER),
    (REJECTED, measurement_of(12, [0.2, 0.9], 0.5), NOT_MEASURED),
    (REJECTED, measurement_of(200, [0.30, 0.44], 0.5), WORSE),
    (ACCEPTED, measurement_of(200, [0.55, 0.62], 0.5, 6.2),
     COSTS_MORE_THAN_IT_GAINS),
    (ACCEPTED, measurement_of(200, [0.55, 0.62], 0.5, 1.1), BETTER),
    (ACCEPTED, measurement_of(200, [0.55, 0.62], 0.5), BETTER),
    (NOT_EVALUATED, measurement_of(500, [0.55, 0.62], 0.5), NOT_MEASURED),
])
def test_the_class_follows_from_the_numbers(verdict, measurement, expected):
    assert classify(verdict, measurement).failure == expected


def test_every_class_names_the_rule_that_produced_it():
    """So a reader can disagree with the rule instead of with the label."""
    for measurement in (measurement_of(240, [0.376, 0.533], 0.5),
                        measurement_of(5, [0.1, 0.9], 0.5),
                        {"score": 0.47}):
        assert classify(REJECTED, measurement).rule.strip()


def test_a_measurement_that_cannot_be_classified_says_so(tmp_path):
    """"We cannot classify this" is a fact. Inventing a class for it is
    the failure this whole design avoids."""
    found = classify(REJECTED, {"score": 0.4708, "games": 240})
    assert found.failure == UNCLASSIFIED
    assert "trials" in found.rule


def test_a_malformed_measurement_is_unclassified_not_a_crash():
    assert classify(REJECTED, {"trials": "много", "interval": [0.1, 0.2],
                               "null": 0.5}).failure == UNCLASSIFIED
    assert classify(REJECTED, {"trials": 40, "interval": "широкий",
                               "null": 0.5}).failure == UNCLASSIFIED


def test_the_class_is_derived_on_read_not_stored(tmp_path):
    """A stored label can be edited, or drift out of agreement with the
    numbers printed beside it."""
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding(REJECTED,
                          measurement=measurement_of(240, [0.376, 0.533], 0.5)))

    # Hand-edit the file to claim a class its numbers do not support.
    path = tmp_path / "findings.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["failure"] = {"failure": BETTER, "rule": "потому что я так решил"}
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")

    assert ledger.findings()[0].failure.failure == NOT_BETTER


def test_the_chess_result_classifies_as_not_better():
    ledger = Ledger()
    chess = [f for f in ledger.latest() if f.approach.get("domain") == "chess"]
    if not chess:
        pytest.skip("шахматный эксперимент на этой машине не проводился")
    assert chess[0].failure.failure == NOT_BETTER


# --------------------------------------------------------------------------
# guesses
# --------------------------------------------------------------------------

def test_a_guess_is_stored_apart_from_the_derived_class():
    """A guess may seed the next experiment; it may never be a premise in
    a conclusion."""
    guessed = finding(suspected=("признаки грубые", "метки от базового игрока"))
    assert guessed.suspected
    assert guessed.failure.failure in (NOT_BETTER, NOT_MEASURED, UNCLASSIFIED)
    assert "признаки грубые" not in json.dumps(guessed.failure.as_dict(),
                                               ensure_ascii=False)


def test_a_guess_does_not_change_what_experiment_this_is():
    """Otherwise adding a hunch would make an old result invisible."""
    assert finding().finding_id == finding(
        suspected=("что-то", "ещё что-то")).finding_id


def test_nothing_branches_on_a_guess():
    """The rule this design turns on. If `classify` ever reads
    `suspected`, a self-assigned label becomes a premise."""
    import inspect

    from mana.cognition import findings as module

    assert "suspected" not in inspect.getsource(module.classify)
    assert "suspected" not in inspect.getsource(module.Ledger.already_tried)


def test_guesses_survive_a_round_trip(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding(suspected=("догадка раз", "догадка два")))
    assert ledger.findings()[0].suspected == ("догадка раз", "догадка два")


def test_an_old_record_without_guesses_still_loads(tmp_path):
    """Records written before the field existed must not become
    unreadable -- losing the ledger would be worse than the gap it was
    built to close."""
    path = tmp_path / "findings.jsonl"
    path.write_text(json.dumps({
        "question": QUESTION, "approach": APPROACH, "verdict": REJECTED,
        "measurement": {"score": 0.47}, "conditions": {}, "created": 1.0,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    read_back = Ledger(path).findings()
    assert len(read_back) == 1
    assert read_back[0].suspected == ()
    assert read_back[0].failure.failure == UNCLASSIFIED


# --------------------------------------------------------------------------
# conditions belong to identity
#
# Found by running the series on the real ledger: corpus=1000 and
# corpus=5000 collapsed to one id, `latest()` kept only the newer, and the
# reader compared the new point against a stale record while reporting a
# series of two that was really one.
# --------------------------------------------------------------------------

def test_the_same_approach_under_different_conditions_is_a_different_finding():
    """A series is exactly "same approach, different conditions". With
    conditions outside the identity those are the points that vanish."""
    at_1000 = finding(conditions={"corpus_games": 1000, "depth": 2})
    at_5000 = finding(conditions={"corpus_games": 5000, "depth": 2})
    assert at_1000.approach_id == at_5000.approach_id
    assert at_1000.finding_id != at_5000.finding_id


def test_both_points_of_a_series_survive_deduplication(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding(conditions={"corpus_games": 1000}, created=1.0))
    ledger.record(finding(conditions={"corpus_games": 5000}, created=2.0))
    assert len(ledger.latest()) == 2


def test_a_rerun_under_identical_conditions_still_supersedes(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding(REJECTED, conditions={"corpus_games": 1000},
                          created=1.0))
    ledger.record(finding(ACCEPTED, conditions={"corpus_games": 1000},
                          created=2.0))
    latest = ledger.latest()
    assert len(latest) == 1 and latest[0].verdict == ACCEPTED


def test_an_exact_condition_match_says_so(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding(conditions={"corpus_games": 1000, "depth": 2}))
    found = ledger.already_tried(QUESTION, APPROACH,
                                 {"corpus_games": 1000, "depth": 2})
    assert found["match"] == "exact"


def test_the_same_approach_elsewhere_is_reported_as_weaker(tmp_path):
    """"We have run this approach, somewhere else in the condition space"
    is a different and weaker thing than "we have run precisely this"."""
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding(conditions={"corpus_games": 1000, "depth": 2}))

    found = ledger.already_tried(QUESTION, APPROACH,
                                 {"corpus_games": 20000, "depth": 2})
    assert found is not None
    assert found["match"] == "same_approach_other_conditions"
    assert found["staleness"]["changed"]["corpus_games"] == {"was": 1000,
                                                             "now": 20000}


def test_a_different_approach_is_still_not_found(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.record(finding())
    assert ledger.already_tried(QUESTION, {"model": "boosting"}, {}) is None
