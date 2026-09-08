"""Reading a run of findings, and refusing to explain it.

Two tests carry the weight. `test_it_never_says_why` guards the line
between measurement and explanation: the reader may say the class changed
when a condition differed, and may not say the condition caused it.
`test_a_flip_with_two_changed_conditions_is_not_attributable` guards the
thing that makes that line real -- without it a series of experiments
becomes a place to find the answer you came with.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mana.core.gates import ACCEPTED, REJECTED, NOT_EVALUATED
from mana.cognition import series
from mana.cognition.findings import (Finding, Ledger, measurement_of,
                                     BETTER, NOT_BETTER, UNCLASSIFIED)

QUESTION = "Может ли обученная оценка играть лучше материала?"


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "findings.jsonl")


def add(ledger, *, corpus, depth, interval, verdict=REJECTED,
        measurement=None, created=None, **conditions):
    row = {"corpus_games": corpus, "depth": depth}
    row.update(conditions)
    finding = Finding(
        question=QUESTION,
        # One approach across the series. The fixture used to fold the
        # conditions into it, which made every point a different method --
        # exactly what the same-approach rule now catches.
        approach={"model": "ridge"},
        verdict=verdict,
        measurement=(measurement if measurement is not None
                     else measurement_of(240, interval, 0.5)),
        conditions=row,
        **({"created": created} if created is not None else {}))
    ledger.record(finding)
    return finding


# --------------------------------------------------------------------------
# the line between measurement and explanation
# --------------------------------------------------------------------------

def test_it_reports_the_change_and_the_condition_that_differed(ledger):
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=5000, depth=2, interval=[0.55, 0.62],
        verdict=ACCEPTED, created=2.0)

    read = series.read(QUESTION, ledger)
    assert len(read.isolated_flips) == 1
    said = read.isolated_flips[0].describe()
    assert "corpus_games" in said
    assert f"с {NOT_BETTER} на {BETTER}" in said


def test_it_never_says_why(ledger):
    """It may say "when corpus differed the class changed". It may not say
    "increasing the corpus improves the result" -- that is a causal claim
    from two points, and everything built on top would inherit it."""
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=5000, depth=2, interval=[0.55, 0.62],
        verdict=ACCEPTED, created=2.0)

    read = series.read(QUESTION, ledger)
    produced = " ".join([read.describe()]
                        + [c.describe() for c in read.comparisons])
    for causal in ("улучшает", "ухудшает", "вызывает", "приводит к",
                   "потому что", "因为", "causes", "because", "improves",
                   "следовательно", "значит, нужно"):
        assert causal not in produced, causal


def test_a_flip_with_two_changed_conditions_is_not_attributable(ledger):
    """With three differences the tempting move is to name whichever one
    you were interested in."""
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=5000, depth=4, interval=[0.55, 0.62],
        verdict=ACCEPTED, created=2.0)

    read = series.read(QUESTION, ledger)
    assert len(read.flips) == 1
    assert len(read.isolated_flips) == 0
    assert len(read.confounded_flips) == 1
    said = read.confounded_flips[0].describe()
    assert "corpus_games" in said and "depth" in said
    assert "нельзя" in said


def test_a_condition_recorded_on_only_one_side_confounds_the_pair(ledger):
    """Comparing against something that was not recorded is comparing
    against an assumption."""
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=1000, depth=2, interval=[0.55, 0.62],
        verdict=ACCEPTED, created=2.0, feature_count=30)

    comparison = series.read(QUESTION, ledger).comparisons[0]
    assert comparison.only_in_one == ("feature_count",)
    assert comparison.isolated is False


# --------------------------------------------------------------------------
# what it finds, and what it declines to find
# --------------------------------------------------------------------------

def test_no_flip_is_reported_as_no_flip(ledger):
    """The user's first example: two points, same class."""
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=1000, depth=4, interval=[0.40, 0.55], created=2.0)

    read = series.read(QUESTION, ledger)
    assert read.varied == ("depth",)
    assert read.constant == ("corpus_games",)
    assert read.flips == ()
    assert "переворотов класса не обнаружено" in read.describe()


def test_one_observation_is_not_a_series(ledger):
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533])
    read = series.read(QUESTION, ledger)
    assert read.comparisons == ()
    assert read.flips == ()
    assert "сравнивать не с чем" in read.describe()


def test_an_unknown_question_gives_an_empty_series(ledger):
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533])
    read = series.read("вопрос, которого никто не задавал", ledger)
    assert read.observations == ()
    assert read.summary()["observations"] == 0


def test_an_unclassifiable_finding_makes_a_pair_unusable(ledger):
    """A missing measurement is not evidence that the class stayed the
    same."""
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=5000, depth=2, interval=[0, 0], created=2.0,
        measurement={"score": 0.47})

    read = series.read(QUESTION, ledger)
    assert UNCLASSIFIED in read.classes
    comparison = read.comparisons[0]
    assert comparison.usable is False
    assert read.flips == ()
    assert "сравнивать нельзя" in comparison.describe()


def test_every_pair_is_compared(ledger):
    for index, corpus in enumerate((1000, 5000, 20000)):
        add(ledger, corpus=corpus, depth=2, interval=[0.376, 0.533],
            created=float(index))
    read = series.read(QUESTION, ledger)
    assert len(read.observations) == 3
    assert len(read.comparisons) == 3          # every pair, not just adjacent


def test_observations_are_oldest_first(ledger):
    """A series is read as a history; sorting by outcome would put the
    answer first and the evidence after it."""
    add(ledger, corpus=5000, depth=2, interval=[0.55, 0.62],
        verdict=ACCEPTED, created=200.0)
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=100.0)

    read = series.read(QUESTION, ledger)
    assert [o.conditions["corpus_games"] for o in read.observations] == [1000, 5000]


def test_varied_and_constant_are_reported_apart(ledger):
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0,
        feature_count=12)
    add(ledger, corpus=5000, depth=2, interval=[0.40, 0.55], created=2.0,
        feature_count=12)
    read = series.read(QUESTION, ledger)
    assert read.varied == ("corpus_games",)
    assert read.constant == ("depth", "feature_count")


# --------------------------------------------------------------------------
# it reads; it does not decide
# --------------------------------------------------------------------------

def test_nothing_here_records_or_chooses_anything():
    """A reader that wrote findings, or picked the next experiment, would
    be a second place where those decisions live."""
    import inspect

    source = inspect.getsource(series)
    for writing in (".record(", "Finding(", "def select", "def plan"):
        assert writing not in source


def test_the_summary_counts_isolated_and_confounded_apart(ledger):
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=5000, depth=2, interval=[0.55, 0.62],
        verdict=ACCEPTED, created=2.0)
    add(ledger, corpus=20000, depth=4, interval=[0.56, 0.63],
        verdict=ACCEPTED, created=3.0)

    summary = series.read(QUESTION, ledger).summary()
    assert summary["flips"] == summary["isolated_flips"] + summary["confounded_flips"]
    assert summary["isolated_flips"] >= 1
    assert summary["confounded_flips"] >= 1


def test_the_real_ledger_reads_without_error():
    """The chess finding is a series of one today. It must still read."""
    for read in series.all_series():
        assert read.summary()["observations"] >= 1
        assert read.describe()


# --------------------------------------------------------------------------
# a key absent on one side is not "held constant"
# --------------------------------------------------------------------------

def test_a_condition_recorded_on_one_side_only_is_not_constant(ledger):
    """Found by running it on the real ledger: a key written down on one
    side took a single value and was reported as held constant, which is
    the opposite of true. What was not recorded was not held."""
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=5000, depth=2, interval=[0.40, 0.55], created=2.0,
        feature_count=12)

    read = series.read(QUESTION, ledger)
    assert "feature_count" not in read.constant
    assert read.partial == ("feature_count",)
    assert "не известно" in read.describe()


def test_a_condition_in_every_observation_is_constant(ledger):
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0,
        feature_count=12)
    add(ledger, corpus=5000, depth=2, interval=[0.40, 0.55], created=2.0,
        feature_count=12)
    read = series.read(QUESTION, ledger)
    assert read.constant == ("depth", "feature_count")
    assert read.partial == ()


def test_varied_constant_and_partial_do_not_overlap(ledger):
    add(ledger, corpus=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus=5000, depth=2, interval=[0.40, 0.55], created=2.0,
        feature_count=12)
    read = series.read(QUESTION, ledger)
    assert not set(read.constant) & set(read.partial)
    assert not set(read.constant) & set(read.varied)


def test_the_real_series_finds_the_isolated_pair():
    """The chess series on this machine: corpus 1000 -> 5000, everything
    else held. Whatever it says, it must be comparing cleanly."""
    from mana.cognition.findings import Ledger as RealLedger

    question = ("Может ли MANA обучить оценку позиции, "
                "которая играет лучше подсчёта материала?")
    read = series.read(question, RealLedger())
    if len(read.observations) < 2:
        pytest.skip("серия на этой машине короче двух наблюдений")
    isolated = [c for c in read.comparisons if c.usable and c.isolated]
    assert isolated, "ни одной чисто сравнимой пары"
