"""Turning an isolated flip into a law candidate, and the ceiling on it.

The load-bearing test is `test_proposed_is_the_ceiling_without_a_hidden_set`.
A law that rose to SUPPORTED on evidence that never met a holdout would
make the four statuses decorative, and the chess experiments had no
holdout. `test_nothing_invents_a_hidden_confirmation` guards the same line
from the other side.
"""
from __future__ import annotations

import pytest

from mana.core.gates import ACCEPTED, REJECTED
from mana.cognition import lawgiver, series
from mana.cognition.findings import (Finding, Ledger, measurement_of,
                                     BETTER, COSTS_MORE_THAN_IT_GAINS,
                                     NOT_BETTER, WORSE)
from mana.cognition.laws import (Condition, LawBook, PROPOSED, REFUTED,
                                 SUPPORTED, MIN_TRIALS_FOR_SUPPORT)

QUESTION = "Может ли обученная оценка играть лучше материала?"
APPROACH = {"domain": "chess", "model": "ridge"}


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "findings.jsonl")


def add(ledger, *, interval, created, verdict=REJECTED, cost_ratio=None,
        approach=None, **conditions):
    ledger.record(Finding(
        question=QUESTION, approach=approach or APPROACH, verdict=verdict,
        measurement=measurement_of(240, interval, 0.5, cost_ratio=cost_ratio),
        conditions=conditions, created=created))


def read(ledger):
    return series.read(QUESTION, ledger)


def flipped(ledger):
    """One isolated flip on feature_set, NOT_BETTER -> COSTS_MORE."""
    add(ledger, feature_set="base12", corpus_games=5000,
        interval=[0.4495, 0.6144], created=1.0)
    add(ledger, feature_set="extended", corpus_games=5000,
        interval=[0.6155, 0.7615], created=2.0, verdict=ACCEPTED,
        cost_ratio=7.41)
    return read(ledger)


# --------------------------------------------------------------------------
# the ceiling
# --------------------------------------------------------------------------

def test_proposed_is_the_ceiling_without_a_hidden_set(ledger):
    """240 trials, and still PROPOSED. A law that rose higher on evidence
    that never met a holdout would make the four statuses decorative."""
    book = LawBook()
    laws = lawgiver.propose(flipped(ledger), book, ledger)
    assert len(laws) == 1
    assert laws[0].status == PROPOSED
    assert laws[0].evidence.trials >= MIN_TRIALS_FOR_SUPPORT
    assert laws[0].evidence.hidden_confirmations == 0


def test_nothing_invents_a_hidden_confirmation():
    import inspect

    source = inspect.getsource(lawgiver)
    # Named in prose is fine and is the point -- what must not appear is
    # it being passed.
    assert "hidden_confirmed=" not in source


def test_the_report_says_what_proposed_means():
    """A summary of counts alone would let PROPOSED read as "almost a
    law". It is one supported experiment and nothing more."""
    reported = lawgiver.report(LawBook())
    assert "один поддержанный эксперимент" in reported["note"]
    assert "скрытая выборка" in reported["note"] or "скрытой выборки" in reported["note"]


# --------------------------------------------------------------------------
# only isolated flips
# --------------------------------------------------------------------------

def test_an_isolated_flip_becomes_a_candidate(ledger):
    found = lawgiver.candidates(flipped(ledger), ledger)
    assert len(found) == 1
    assert found[0].intervention == ("feature_set: base12 -> extended",)
    assert found[0].discovered_in == "chess"
    assert found[0].trials == 240


def test_a_confounded_flip_becomes_nothing(ledger):
    """A claim nobody can attribute is not a law."""
    add(ledger, feature_set="base12", corpus_games=1000,
        interval=[0.376, 0.533], created=1.0)
    add(ledger, feature_set="extended", corpus_games=5000,
        interval=[0.6155, 0.7615], created=2.0, verdict=ACCEPTED)

    read_series = read(ledger)
    assert read_series.confounded_flips
    assert lawgiver.candidates(read_series, ledger) == []


def test_a_flat_axis_becomes_nothing(ledger):
    """"Nothing changed" is a fact the series already holds. Recorded as a
    law it would be scored a null effect and refuted on arrival, which
    would put a falsehood in the book."""
    add(ledger, feature_set="base12", corpus_games=1000,
        interval=[0.376, 0.533], created=1.0)
    add(ledger, feature_set="base12", corpus_games=5000,
        interval=[0.4495, 0.6144], created=2.0)

    read_series = read(ledger)
    assert read_series.flips == ()
    assert lawgiver.candidates(read_series, ledger) == []


def test_a_flip_between_two_kinds_of_failure_is_not_an_intervention(ledger):
    """WORSE to NOT_BETTER is real and is not "this helps". Phrasing it as
    one would put a direction into the book the numbers do not carry."""
    add(ledger, feature_set="base12", corpus_games=5000,
        interval=[0.30, 0.44], created=1.0)
    add(ledger, feature_set="extended", corpus_games=5000,
        interval=[0.45, 0.55], created=2.0)

    read_series = read(ledger)
    assert {c.from_class for c in read_series.flips} == {WORSE}
    assert lawgiver.candidates(read_series, ledger) == []


def test_an_empty_series_proposes_nothing():
    assert lawgiver.candidates(series.Series(question="ничего")) == []
    assert lawgiver.propose(series.Series(question="ничего"), LawBook()) == []


# --------------------------------------------------------------------------
# one claim, evidence accumulated
# --------------------------------------------------------------------------

def test_the_same_claim_reached_twice_strengthens_one_law(ledger):
    """Measured on the real ledger: the chess flip is isolated twice, once
    at each eval_version. Two laws would split 480 trials into two lots of
    240 and leave both short of what a status needs."""
    for index, version in enumerate(("1.0", "1.1")):
        add(ledger, feature_set="base12", corpus_games=5000,
            eval_version=version, interval=[0.4495, 0.6144],
            created=1.0 + 2 * index)
        add(ledger, feature_set="extended", corpus_games=5000,
            eval_version=version, interval=[0.6155, 0.7615],
            created=2.0 + 2 * index, verdict=ACCEPTED, cost_ratio=7.41)

    book = LawBook()
    lawgiver.propose(read(ledger), book, ledger)
    assert len(book.all()) == 1
    assert book.all()[0].evidence.trials == 480
    assert len(book.all()[0].evidence.experiments) == 2


def test_proposing_twice_does_not_double_the_evidence(ledger):
    book = LawBook()
    read_series = flipped(ledger)
    lawgiver.propose(read_series, book, ledger)
    lawgiver.propose(read_series, book, ledger)
    assert len(book.all()) == 1
    assert book.all()[0].evidence.trials == 240


# --------------------------------------------------------------------------
# what the law may not claim is written down
# --------------------------------------------------------------------------

def test_the_cost_caveat_becomes_an_exception(ledger):
    """"Better per unit of search" is established; "better on a clock" is
    not. A law whose limits are not written down is a law that will be
    applied outside them."""
    law = lawgiver.propose(flipped(ledger), LawBook(), ledger)[0]
    assert any("7.41" in note and "времени" in note for note in law.exceptions)


def test_the_missing_holdout_becomes_an_exception(ledger):
    law = lawgiver.propose(flipped(ledger), LawBook(), ledger)[0]
    assert any("скрытой выборке" in note for note in law.exceptions)


def test_an_exception_is_not_a_refutation(ledger):
    """A law that holds except under a stated condition is more useful
    than one deleted for being imperfect."""
    law = lawgiver.propose(flipped(ledger), LawBook(), ledger)[0]
    assert law.exceptions
    assert law.status == PROPOSED


def test_the_held_conditions_travel_as_scope(ledger):
    candidate = lawgiver.candidates(flipped(ledger), ledger)[0]
    assert candidate.scope["corpus_games"] == 5000
    assert "feature_set" not in candidate.scope     # that is the intervention


# --------------------------------------------------------------------------
# the book demotes as readily as it promotes
# --------------------------------------------------------------------------

def test_contradicting_evidence_demotes_a_law(ledger):
    """A system that can only promote its own laws accumulates folklore.

    `positive_share` counts experiments, not weight, so one contrary
    result out of two is exactly 0.5 and leaves the law standing -- that
    is ambiguity, not refutation. A majority going the other way refutes.
    """
    book = LawBook()
    law = lawgiver.propose(flipped(ledger), book, ledger)[0]
    assert law.status == PROPOSED

    law.record_evidence(effect=-0.4, trials=300, experiment_id="later")
    assert law.status == PROPOSED, "одно опровержение из двух — это 50/50"

    law.record_evidence(effect=-0.3, trials=300, experiment_id="later-still")
    assert law.status == REFUTED


def test_a_counterexample_refutes(ledger):
    law = lawgiver.propose(flipped(ledger), LawBook(), ledger)[0]
    law.record_evidence(counterexamples_sought=10, counterexamples_found=1,
                        experiment_id="hunt")
    assert law.status == REFUTED


def test_a_refuted_law_is_kept_and_not_acted_on(ledger):
    """"We tried this and it did not hold" is one of the more valuable
    things a research loop can know."""
    book = LawBook()
    law = lawgiver.propose(flipped(ledger), book, ledger)[0]
    law.record_evidence(effect=-0.4, trials=300, experiment_id="later")
    law.record_evidence(effect=-0.3, trials=300, experiment_id="later-still")
    assert law.status == REFUTED

    assert law in book.all()
    assert book.applicable("chess", 0.5) == []


# --------------------------------------------------------------------------
# persistence and the live book
# --------------------------------------------------------------------------

def test_a_book_survives_a_round_trip(tmp_path, ledger):
    book = LawBook()
    lawgiver.propose(flipped(ledger), book, ledger)
    path = tmp_path / "laws.json"
    assert lawgiver.save_book(book, path)

    restored = lawgiver.load_book(path)
    assert len(restored.all()) == 1
    assert restored.all()[0].status == PROPOSED
    assert restored.all()[0].evidence.trials == 240


def test_an_unwritable_book_does_not_raise(tmp_path, ledger):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    book = LawBook()
    lawgiver.propose(flipped(ledger), book, ledger)
    assert lawgiver.save_book(book, blocker / "sub" / "laws.json") is False


def test_a_missing_book_loads_empty(tmp_path):
    assert lawgiver.load_book(tmp_path / "nothing.json").all() == []


def test_the_real_series_yields_a_proposed_law():
    """On this machine the chess flip is isolated. Whatever it proposes,
    it may not exceed PROPOSED without a holdout."""
    from mana.cognition.findings import Ledger as RealLedger

    question = ("Может ли MANA обучить оценку позиции, "
                "которая играет лучше подсчёта материала?")
    read_series = series.read(question, RealLedger())
    if not read_series.isolated_flips:
        pytest.skip("на этой машине изолированных переворотов нет")

    book = LawBook()
    laws = lawgiver.propose(read_series, book, RealLedger())
    assert laws
    for law in laws:
        assert law.status == PROPOSED
        assert law.exceptions


# --------------------------------------------------------------------------
# the intervention format is a contract, owned here
# --------------------------------------------------------------------------

def test_an_intervention_names_its_axis_readably(ledger):
    """`probes.py` reads this. The format lives here because this module
    writes it -- the alternative is the reader guessing at prose."""
    law = lawgiver.propose(flipped(ledger), LawBook(), ledger)[0]
    assert lawgiver.axes_of(law) == ["feature_set"]


@pytest.mark.parametrize("text,expected", [
    ("feature_set: base12 -> extended", "feature_set"),
    ("corpus_games: 1000 -> 5000", "corpus_games"),
    ("VERIFY before ANSWER", ""),          # a law written by hand
    ("", ""),
    ("no separator here", ""),
])
def test_axis_of_returns_empty_rather_than_guessing(text, expected):
    """A law imported from elsewhere names its intervention however it
    likes, and inventing an axis for it would be guessing."""
    assert lawgiver.axis_of(text) == expected
