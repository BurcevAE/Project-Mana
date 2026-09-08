"""What we learned, as opposed to what we tried -- and what changes because of it.

The criterion these are written against: a finding must change the space
of next actions, not only sink a candidate in a list. So the tests that
matter compare two plans built from the same violations and the same
policy, differing only in what the ledger holds.
"""
from __future__ import annotations

import pytest

from mana import policy as policy_mod
from mana.cognition import candidates, lessons, observed
from mana.cognition.findings import Finding, Ledger, measurement_of
from mana.cognition.invariants import MECHANICAL, Violation
from mana.core.gates import ACCEPTED, NOT_EVALUATED, REJECTED


REPEATS = "repeats_earlier_answer"
STATE = "acted_but_state_disagrees"


def _violation(invariant: str = REPEATS, episode: str = "e1") -> Violation:
    return Violation(invariant=invariant, kind=MECHANICAL,
                     episode_id=episode, reason="повтор")


def _measured(approach, verdict=REJECTED, question=None) -> Finding:
    """A finding a number stands behind."""
    return Finding(question=question or candidates.question_for(REPEATS),
                   approach=approach, verdict=verdict,
                   measurement=measurement_of(trials=40, interval=(-0.2, -0.05),
                                              null=0.0))


def _adopted(approach) -> Finding:
    """A finding adopted under stated uncertainty. Tried, not learned."""
    return Finding(question=candidates.question_for(REPEATS),
                   approach=approach, verdict=NOT_EVALUATED,
                   measurement={}, note="принято при названной неопределённости")


# --------------------------------------------------------------------------
# tried is not learned
# --------------------------------------------------------------------------

def test_an_adoption_under_uncertainty_is_tried_and_not_learned(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(_adopted({"echo_lookback": 3}))

    lesson = lessons.read(ledger)[REPEATS]
    assert len(lesson.tried) == 1
    assert lesson.learned == ()
    assert lesson.unmeasured[0].verdict == NOT_EVALUATED


def test_a_measured_rejection_is_both(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(_measured({"echo_lookback": 3}))

    lesson = lessons.read(ledger)[REPEATS]
    assert len(lesson.tried) == 1 and len(lesson.learned) == 1
    assert lesson.learned[0].measured is True
    assert lesson.learned[0].helped is False


def test_what_was_measured_narrows_where_the_information_is(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    before = lessons.read(ledger)
    ledger.record(_measured({"echo_lookback": 3}))
    after = lessons.read(ledger)[REPEATS]

    assert REPEATS not in before  # an empty ledger teaches nothing
    assert ("echo_lookback", 3) in after.measured_settings
    assert ("echo_lookback", 3) not in after.open_settings
    # The other options on the same axis are still open.
    assert ("echo_lookback", 12) in after.open_settings


def test_an_unmeasured_attempt_narrows_nothing(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(_adopted({"echo_lookback": 3}))
    lesson = lessons.read(ledger)[REPEATS]
    assert lesson.measured_settings == ()
    assert ("echo_lookback", 3) in lesson.open_settings


# --------------------------------------------------------------------------
# the count comes from real work, not from the window
# --------------------------------------------------------------------------

def test_the_observed_count_is_read_from_what_live_work_recorded(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(Finding(
        question=observed.question_for(REPEATS),
        approach=dict(observed.OBSERVED_APPROACH), verdict=NOT_EVALUATED,
        measurement={"observed_failures": 14, "last_episode": "e9",
                     "last_request": "Мана, привет!"}))

    lesson = lessons.read(ledger)[REPEATS]
    assert lesson.observed == 14
    assert lesson.last_request == "Мана, привет!"
    # An observation is not an attempt.
    assert lesson.tried == ()


def test_a_failure_the_ledger_knows_about_is_addressed_on_a_clean_window(tmp_path):
    # The strategy change, plainly: no violation in this window at all,
    # and candidates are still proposed, because the record says this
    # keeps happening.
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(Finding(
        question=observed.question_for(REPEATS),
        approach=dict(observed.OBSERVED_APPROACH), verdict=NOT_EVALUATED,
        measurement={"observed_failures": 5}))
    learned = lessons.read(ledger)

    assert candidates.propose([]) == []
    with_record = candidates.propose([], lessons=learned)
    assert with_record
    assert {c.addresses for c in with_record} == {REPEATS}


def test_the_record_outranks_the_window_when_they_disagree(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(Finding(
        question=observed.question_for("actionable_request_not_acted_on"),
        approach=dict(observed.OBSERVED_APPROACH), verdict=NOT_EVALUATED,
        measurement={"observed_failures": 20}))
    learned = lessons.read(ledger)

    violations = [_violation(REPEATS, "e1")]
    without = candidates.propose(violations)
    within = candidates.propose(violations, lessons=learned)

    assert without[0].addresses == REPEATS
    # Seen twenty times in real work against one in this window.
    assert within[0].addresses == "actionable_request_not_acted_on"


# --------------------------------------------------------------------------
# the space changes, not just the order
# --------------------------------------------------------------------------

def test_an_exhausted_axis_stops_being_proposed(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    for knob in policy_mod.knobs_for(REPEATS):
        for option in knob.options:
            if option != knob.default:
                ledger.record(_measured({knob.name: option}))
    learned = lessons.read(ledger)

    assert learned[REPEATS].exhausted is True
    made = candidates.plan([_violation(REPEATS)], lessons=learned)
    assert made.candidates == ()
    assert [r["strategy"] for r in made.refusals] == [lessons.EXHAUSTED]
    # And without the ledger, the same violation still produces candidates.
    assert candidates.plan([_violation(REPEATS)]).candidates


def test_one_measurement_that_helped_keeps_the_axis_open(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    for knob in policy_mod.knobs_for(REPEATS):
        for option in knob.options:
            if option != knob.default:
                ledger.record(_measured({knob.name: option}))
    ledger.record(Finding(
        question=candidates.question_for(REPEATS),
        approach={"echo_lookback": 12}, verdict=ACCEPTED,
        measurement=measurement_of(trials=40, interval=(0.05, 0.2), null=0.0)))

    lesson = lessons.read(ledger)[REPEATS]
    assert lesson.exhausted is False


def test_adoptions_under_uncertainty_never_exhaust_an_axis(tmp_path):
    # Strict on purpose. Loosening this until it fires would be the same
    # mistake as calling an unobserved launch a successful one.
    ledger = Ledger(tmp_path / "f.jsonl")
    for knob in policy_mod.knobs_for(REPEATS):
        for option in knob.options:
            if option != knob.default:
                ledger.record(_adopted({knob.name: option}))
    assert lessons.read(ledger)[REPEATS].exhausted is False


# --------------------------------------------------------------------------
# what nothing can be proposed for is said out loud
# --------------------------------------------------------------------------

def test_a_failure_no_knob_aims_at_is_reported_rather_than_dropped():
    made = candidates.plan([_violation(STATE)])
    assert made.candidates == ()
    assert len(made.refusals) == 1
    assert made.refusals[0]["addresses"] == STATE
    assert made.refusals[0]["strategy"] == lessons.NO_KNOB
    assert "кодом" in made.refusals[0]["why"]


def test_the_refusal_carries_how_often_it_happened(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(Finding(
        question=observed.question_for(STATE),
        approach=dict(observed.OBSERVED_APPROACH), verdict=NOT_EVALUATED,
        measurement={"observed_failures": 7}))
    made = candidates.plan([], lessons=lessons.read(ledger))
    assert made.refusals[0]["observed_failures"] == 7


def test_a_clean_record_proposes_and_refuses_nothing():
    made = candidates.plan([])
    assert made.candidates == () and made.refusals == ()


# --------------------------------------------------------------------------
# the strategy, named
# --------------------------------------------------------------------------

def test_the_strategy_says_which_space_to_search(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(Finding(
        question=observed.question_for(REPEATS),
        approach=dict(observed.OBSERVED_APPROACH), verdict=NOT_EVALUATED,
        measurement={"observed_failures": 2}))
    assert lessons.read(ledger)[REPEATS].strategy()[0] == lessons.KNOB_SEARCH

    empty = lessons.Lesson(invariant=STATE, observed=3)
    assert empty.strategy()[0] == lessons.NO_KNOB


def test_rank_carries_tried_and_learned_apart(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(_adopted({"echo_lookback": 3}))
    ledger.record(Finding(
        question=observed.question_for(REPEATS),
        approach=dict(observed.OBSERVED_APPROACH), verdict=NOT_EVALUATED,
        measurement={"observed_failures": 4}))
    learned = lessons.read(ledger)

    rows = candidates.rank([_violation(REPEATS)], [], ledger=ledger,
                           lessons=learned)
    assert rows
    seen = rows[0]["lesson"]
    assert seen["observed"] == 4 and seen["tried"] == 1 and seen["learned"] == 0


def test_a_question_from_another_domain_is_not_read_as_a_lesson(tmp_path):
    # The chess findings share the ledger. A lesson is about an invariant,
    # and a finding that is not about one must not be counted as an
    # attempt at it.
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(Finding(question="Обученная оценка лучше подсчёта материала?",
                          approach={"domain": "chess"}, verdict=NOT_EVALUATED))
    assert lessons.read(ledger) == {}


# --------------------------------------------------------------------------
# the window reads what this produces
# --------------------------------------------------------------------------

def test_the_tab_payload_carries_both_halves(tmp_path, monkeypatch):
    """The seam to the window. `proposals` changed shape here, and the
    renderer that read it as an array would have shown an empty panel
    with nothing failing anywhere."""
    from mana_desktop.session import AgentSession

    monkeypatch.chdir(tmp_path)      # an empty journal and an empty ledger
    session = object.__new__(AgentSession)
    data = session.self_knowledge(limit=5)

    assert "lessons" in data
    assert isinstance(data["proposals"], dict)
    assert set(data["proposals"]) >= {"candidates", "refusals"}


# --------------------------------------------------------------------------
# the ranking, before and after the record learned something
# --------------------------------------------------------------------------

def _echo_rows(ledger, learned=None):
    return candidates.rank([_violation(REPEATS)], [], ledger=ledger,
                           lessons=learned)


def test_the_same_candidates_are_ranked_differently_after_a_measurement(tmp_path):
    """The criterion: same set, different order, because of the record."""
    ledger = Ledger(tmp_path / "f.jsonl")
    before = _echo_rows(ledger, lessons.read(ledger))

    # Measured under exactly the conditions that hold now: one observed
    # failure, which is what this violation amounts to.
    ledger.record(Finding(
        question=candidates.question_for(REPEATS),
        approach={"echo_lookback": 3}, verdict=REJECTED,
        conditions={"observed_failures": 1},
        measurement=measurement_of(trials=40, interval=(-0.2, -0.05), null=0.0)))
    after = _echo_rows(ledger, lessons.read(ledger))

    def ids(rows):
        return [r["candidate_id"] for r in rows]

    assert set(ids(before)) == set(ids(after)), "the set must not change"
    assert ids(before) != ids(after), "the order must"

    measured = next(r for r in after if r["changes"] == {"echo_lookback": 3})
    assert measured["information"] == candidates.ALREADY_MEASURED
    assert ids(after)[-1] == measured["candidate_id"]


def test_without_a_record_every_candidate_is_equally_informative(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    rows = _echo_rows(ledger, lessons.read(ledger))
    assert rows
    assert {r["information"] for r in rows} == {candidates.NEVER_MEASURED}


def test_a_measurement_under_conditions_that_moved_is_a_weaker_prior(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    stale = Finding(question=candidates.question_for(REPEATS),
                    approach={"echo_lookback": 3}, verdict=REJECTED,
                    conditions={"observed_failures": 99},  # not 1, as now
                    measurement=measurement_of(trials=40, interval=(-0.2, -0.05),
                                               null=0.0))
    ledger.record(stale)
    rows = _echo_rows(ledger, lessons.read(ledger))
    row = next(r for r in rows if r["changes"] == {"echo_lookback": 3})
    # Measured, but not here: worth more than a settled result and less
    # than an axis nobody has touched.
    assert row["information"] == candidates.CONDITIONS_MOVED
    assert (candidates.ALREADY_MEASURED < row["information"]
            < candidates.NEVER_MEASURED)


def test_an_unmeasured_attempt_does_not_lower_the_information(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(_adopted({"echo_lookback": 3}))
    rows = _echo_rows(ledger, lessons.read(ledger))
    row = next(r for r in rows if r["changes"] == {"echo_lookback": 3})
    assert row["information"] == candidates.NEVER_MEASURED


def test_the_value_leaves_the_dry_bound_out(tmp_path):
    # It exists for some candidates and not others; averaging it in would
    # make "could not be evaluated" mean "no gain".
    scored = candidates.Scored(row={}, information=1.0)
    assert scored.value == 1.0
    priced = candidates.Scored(row={}, information=1.0,
                               estimated_calls=int(candidates.COST_SCALE))
    assert priced.value < scored.value


def test_the_dry_bound_still_decides_between_equally_informative_ones(monkeypatch):
    """The old ordering survives where the record says nothing."""
    from mana.apps import intent
    from mana.cognition import failure_domain as fd
    from mana.journal import Episode
    from mana.policy import Policy

    targets = ["UT11-ER", "Информационная база"]
    monkeypatch.setattr(intent, "_known_bases", lambda: targets)
    episodes = [
        Episode("a", "s", 1.0,
                "я хочу поработать с 1С запусти конфигуратор информационной базы",
                "Чтобы запустить, найдите ярлык 1С на рабочем столе."),
        Episode("b", "s", 2.0, "расскажи про налоги",
                "Развёрнутый содержательный ответ про налоги.")]
    from mana.cognition.invariants import scan

    found = scan(episodes, targets)
    narrow = Policy.of(intent_verb_anywhere=False, intent_stem_match=False,
                       intent_verb_forms="imperative")
    rows = candidates.rank(found, fd.situations_from(episodes, targets),
                           current=narrow)
    evaluable = [r for r in rows if r["dry"].get("dry_evaluable")]
    assert evaluable
    best = evaluable[0]["dry"]
    assert best["candidate_pass_rate"] > best["baseline_pass_rate"]


# --------------------------------------------------------------------------
# chosen by the same selector as everything else
# --------------------------------------------------------------------------

def test_choosing_goes_through_the_shared_selector(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    picked = candidates.choose([_violation(REPEATS)], [], budget=100,
                               ledger=ledger, lessons=lessons.read(ledger))
    assert picked is not None
    assert picked["value"] >= candidates.MIN_EXPERIMENT_VALUE


def test_the_least_informative_candidate_is_not_the_one_chosen(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    ledger.record(Finding(
        question=candidates.question_for(REPEATS),
        approach={"echo_lookback": 3}, verdict=REJECTED,
        conditions={"observed_failures": 1},
        measurement=measurement_of(trials=40, interval=(-0.2, -0.05), null=0.0)))
    picked = candidates.choose([_violation(REPEATS)], [], budget=100,
                               ledger=ledger, lessons=lessons.read(ledger))
    assert picked["changes"] != {"echo_lookback": 3}


def test_nothing_to_measure_is_a_real_answer():
    assert candidates.choose([], [], budget=100) is None


def test_next_answers_without_a_journal(tmp_path, monkeypatch, capsys):
    """The reader is the visible half of this; a crash in it would be
    found by a person typing the flag, which is the loop this project is
    trying to get out of."""
    from mana.cli import _next_policy_experiment

    # Both, because either one alone can be defeated: a working directory
    # is ignored when MANA_DATA_DIR is set, and the variable is ignored
    # by anything that resolves against the CWD. Found by running the
    # suite with the variable set for a chess experiment, where this test
    # then read the real journal.
    monkeypatch.setenv("MANA_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert _next_policy_experiment() == 0
    assert capsys.readouterr().out == ""    # nothing to say, said quietly
