"""Which axis to vary next, and what this layer refuses to decide.

Three tests carry the weight. `test_the_choice_uses_the_existing_selector`
guards against a second selector letting the findings space choose
experiments on easier terms than everything else. `test_a_contextual
_condition_is_never_chosen` guards the defect running it found -- "vary
arena_version" ranked first. And `test_nothing_reads_the_guesses` guards
the rule the whole ledger turns on: a guess may seed an experiment and may
never be a premise.
"""
from __future__ import annotations

import pytest

from mana.core.gates import ACCEPTED, REJECTED
from mana.cognition import probes, series
from mana.cognition.experiments import MIN_EXPERIMENT_VALUE, VALUE_WEIGHTS
from mana.cognition.findings import Finding, Ledger, measurement_of

QUESTION = "Может ли обученная оценка играть лучше материала?"
CONTROLLABLE = ("corpus_games", "depth", "feature_count")


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "findings.jsonl")


def add(ledger, *, interval, created, verdict=REJECTED, **conditions):
    ledger.record(Finding(
        question=QUESTION, approach={"model": "ridge"}, verdict=verdict,
        measurement=measurement_of(240, interval, 0.5),
        conditions=conditions, created=created))


def read(ledger):
    return series.read(QUESTION, ledger)


def flat_series(ledger):
    """corpus varied twice with no flip; depth and feature_count untouched."""
    add(ledger, corpus_games=1000, depth=2, feature_count=12,
        interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus_games=5000, depth=2, feature_count=12,
        interval=[0.4495, 0.6144], created=2.0)
    return read(ledger)


# --------------------------------------------------------------------------
# the connection: the decision rule is shared, the shape is not
# --------------------------------------------------------------------------

def test_the_choice_uses_the_existing_selector():
    """A second selector would let the findings space choose experiments
    on easier terms than everything else."""
    import inspect

    source = inspect.getsource(probes)
    assert "from .experiments import" in source
    assert "select(" in source
    # No thresholds of its own: the floor everything else is held to.
    assert "MIN_EXPERIMENT_VALUE =" not in source
    assert "VALUE_WEIGHTS =" not in source


def test_a_probe_is_shaped_for_the_existing_selector(ledger):
    """`select` touches exactly `estimated_calls` and `value`."""
    probe = probes.probes(flat_series(ledger))[0]
    assert isinstance(probe.estimated_calls, int)
    assert isinstance(probe.value, float)


def test_the_value_uses_the_shared_weights(ledger):
    probe = probes.probes(flat_series(ledger), cost=lambda c: 0)[0]
    assert probe.value == pytest.approx(
        VALUE_WEIGHTS["information_gain"] * probe.information)


def test_capability_is_left_out_rather_than_counted_as_zero(ledger):
    """"Unmeasured is not zero" is the project's own rule. There is no
    capability table for an arbitrary question, so the term is omitted and
    every probe says so."""
    probe = probes.probes(flat_series(ledger))[0]
    assert "не измерена" in probe.as_dict()["capability_gain"]
    assert "capability_gain" not in str(probe.value)


# --------------------------------------------------------------------------
# information, from what was observed
# --------------------------------------------------------------------------

def test_an_axis_nobody_varied_is_the_most_informative(ledger):
    found = {p.condition: p for p in probes.probes(flat_series(ledger))}
    assert found["depth"].information == probes.NEVER_VARIED
    assert found["feature_count"].information == probes.NEVER_VARIED


def test_an_axis_varied_without_a_flip_says_less_each_time(ledger):
    found = {p.condition: p for p in probes.probes(flat_series(ledger))}
    assert found["corpus_games"].information < found["depth"].information
    assert found["corpus_games"].information == pytest.approx(1 / 3)


def test_an_axis_in_a_confounded_flip_is_worth_isolating(ledger):
    add(ledger, corpus_games=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus_games=5000, depth=4, interval=[0.55, 0.62],
        verdict=ACCEPTED, created=2.0)

    found = {p.condition: p for p in probes.probes(read(ledger))}
    assert found["corpus_games"].information == probes.IN_CONFOUNDED_FLIP
    assert found["depth"].information == probes.IN_CONFOUNDED_FLIP
    assert "не удалось приписать" in found["depth"].reason


def test_an_axis_with_an_isolated_flip_is_already_bounded(ledger):
    add(ledger, corpus_games=1000, depth=2, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus_games=5000, depth=2, interval=[0.55, 0.62],
        verdict=ACCEPTED, created=2.0)

    found = {p.condition: p for p in probes.probes(read(ledger))}
    assert found["corpus_games"].information == probes.FLIP_FOUND


def test_an_empty_series_proposes_nothing():
    assert probes.probes(series.Series(question="ничего")) == []


# --------------------------------------------------------------------------
# not every condition is an axis
# --------------------------------------------------------------------------

def test_a_contextual_condition_is_never_chosen(ledger):
    """Found by running it: "vary arena_version" ranked first. A component
    version is recorded so a comparison can be trusted, never set to learn
    anything."""
    add(ledger, corpus_games=1000, depth=2, feature_count=12,
        arena_version="1.0", interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus_games=5000, depth=2, feature_count=12,
        arena_version="1.0", interval=[0.4495, 0.6144], created=2.0)

    chosen = probes.choose(read(ledger), budget=10_000,
                           cost=lambda c: 100, controllable=CONTROLLABLE)
    assert chosen is not None
    assert chosen.condition != "arena_version"
    assert chosen.condition in CONTROLLABLE


def test_a_contextual_condition_is_still_reported(ledger):
    """Dropping it would let a reader think it was never considered."""
    add(ledger, corpus_games=1000, arena_version="1.0",
        interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus_games=5000, arena_version="1.0",
        interval=[0.4495, 0.6144], created=2.0)

    found = {p.condition: p for p in
             probes.probes(read(ledger), controllable=CONTROLLABLE)}
    assert "arena_version" in found
    assert found["arena_version"].controllable is False
    assert found["arena_version"].worth_running is False
    assert "НЕ ось" in found["arena_version"].describe()


def test_declaring_nothing_is_visible_rather_than_silent(ledger):
    reported = probes.report(flat_series(ledger), cost=lambda c: 100)
    assert reported["controllable_declared"] is None
    assert any("не объявлены" in note for note in reported["notes"])


# --------------------------------------------------------------------------
# cost is supplied, never invented
# --------------------------------------------------------------------------

def test_an_unpriced_probe_says_so_rather_than_looking_cheap(ledger):
    probe = probes.probes(flat_series(ledger))[0]
    assert probe.priced is False
    assert "цена не задана" in probe.describe()


def test_no_unit_is_printed(ledger):
    """What a probe costs is the caller's fact -- seconds for chess, calls
    for a pipeline experiment -- and inventing a unit would be inventing a
    measurement."""
    probe = probes.probes(flat_series(ledger), cost=lambda c: 450)[0]
    said = probe.describe()
    assert "цена 450" in said
    for unit in ("сек", "ед.", "calls", "мин"):
        assert unit not in said


def test_a_dearer_probe_is_worth_less(ledger):
    cheap = probes.probes(flat_series(ledger), cost=lambda c: 0)
    dear = probes.probes(flat_series(ledger), cost=lambda c: 5000)
    assert dear[0].value < cheap[0].value


def test_a_broken_cost_function_leaves_the_probe_unpriced(ledger):
    def explodes(_condition):
        raise RuntimeError("no idea what this costs")

    probe = probes.probes(flat_series(ledger), cost=explodes)[0]
    assert probe.priced is False


# --------------------------------------------------------------------------
# it chooses an axis, not a value; and it may choose nothing
# --------------------------------------------------------------------------

def test_a_probe_names_the_axis_and_what_was_tried_not_a_new_value(ledger):
    """Choosing a new value would be this module deciding something it has
    not measured."""
    found = {p.condition: p for p in probes.probes(flat_series(ledger))}
    assert found["corpus_games"].tried_values == (1000, 5000)
    assert not hasattr(found["corpus_games"], "next_value")


def test_nothing_affordable_is_a_real_answer(ledger):
    chosen = probes.choose(flat_series(ledger), budget=1,
                           cost=lambda c: 100_000, controllable=CONTROLLABLE)
    assert chosen is None


def test_an_exhausted_axis_is_not_worth_running(ledger):
    """Below the shared floor, running it because there is budget left is
    how a research loop converts compute into noise."""
    add(ledger, corpus_games=1000, interval=[0.376, 0.533], created=1.0)
    add(ledger, corpus_games=5000, interval=[0.40, 0.55], created=2.0)
    add(ledger, corpus_games=20000, interval=[0.41, 0.56], created=3.0)

    found = {p.condition: p for p in
             probes.probes(read(ledger), cost=lambda c: 4000)}
    assert found["corpus_games"].value < MIN_EXPERIMENT_VALUE
    assert found["corpus_games"].worth_running is False


# --------------------------------------------------------------------------
# the rule the ledger turns on
# --------------------------------------------------------------------------

def test_nothing_reads_the_guesses():
    """A guess may seed the next experiment and may never be a premise in
    a conclusion. If this module ever read `suspected`, the choice would
    rest on a self-assigned label."""
    import inspect

    source = inspect.getsource(probes)
    assert "suspected" not in source
    assert "note" not in inspect.getsource(probes.probes)


def test_it_records_nothing():
    """Choosing an experiment and recording its result are different
    decisions and stay in different places."""
    import inspect

    source = inspect.getsource(probes)
    for writing in (".record(", "Finding("):
        assert writing not in source


def test_the_real_series_chooses_a_real_axis():
    """On this machine: corpus varied twice with no flip, so an untouched
    axis should win. Whatever it picks, it must be one a person could set."""
    from mana.cognition.findings import Ledger as RealLedger

    question = ("Может ли MANA обучить оценку позиции, "
                "которая играет лучше подсчёта материала?")
    read_series = series.read(question, RealLedger())
    if len(read_series.observations) < 2:
        pytest.skip("серия на этой машине короче двух наблюдений")

    controllable = ("corpus_games", "depth", "feature_count",
                    "corpus_source", "positions_per_game")
    chosen = probes.choose(read_series, budget=10_000,
                           cost=lambda c: 450, controllable=controllable)
    assert chosen is not None
    assert chosen.condition in controllable
