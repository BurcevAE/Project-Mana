"""
tests/test_population.py — the ecology, and the specialist that
single-champion search throws away.

The property everything else here serves: a program that is superb at one
thing and mediocre elsewhere must survive a comparison with one that is
uniformly middling. `evolve_pipeline` keeps one champion and loses exactly
that candidate, which is why open-endedness needs a different structure.
"""
from __future__ import annotations

import pytest

from mana.cognition.novelty import Behaviour
from mana.cognition.population import (COMPETENT, Candidate, Population,
                                       niche_of)


def behaviour(cid, pattern, steps=("OBSERVE", "GENERATE", "ANSWER"),
              failures=None, calls=2.0):
    return Behaviour(cid, tuple(steps), tuple(bool(x) for x in pattern),
                     failures or {}, calls, 1.0, cid)


def candidate(cid, scores, pattern=None, steps=("OBSERVE", "GENERATE", "ANSWER"),
              failures=None, calls=2.0, **kw):
    return Candidate(candidate_id=cid, steps=tuple(steps),
                     behaviour=behaviour(cid, pattern, steps, failures, calls)
                     if pattern else None,
                     scores=dict(scores), **kw)


# ---------------------------------------------------------------------------
# the specialist survives
# ---------------------------------------------------------------------------

def test_a_specialist_is_not_lost_to_a_better_average():
    """The failure single-champion search makes, in one assertion."""
    population = Population()
    generalist = candidate("gen", {"arithmetic/easy": 0.70, "arithmetic/hard": 0.65,
                                   "logic/easy": 0.68}, [1, 1, 0, 1])
    specialist = candidate("spec", {"arithmetic/hard": 0.95}, [0, 0, 1, 1])
    population.admit(generalist)
    result = population.admit(specialist)
    assert result.admitted is True
    assert "arithmetic/hard" in result.as_elite
    assert population.elites()["arithmetic/hard"].candidate_id == "spec"
    assert population.elites()["arithmetic/easy"].candidate_id == "gen"


def test_breadth_distinguishes_a_specialist_from_a_generalist():
    """A specialist and a generalist with the same peak are different
    animals, and one number cannot show it."""
    generalist = candidate("gen", {"a/easy": 0.8, "a/hard": 0.8, "b/easy": 0.8})
    specialist = candidate("spec", {"a/hard": 0.8, "b/easy": 0.2})
    assert generalist.breadth() == 3
    assert specialist.breadth() == 1


def test_an_empty_niche_is_filled_by_whoever_arrives():
    population = Population()
    result = population.admit(candidate("first", {"logic/medium": 0.3}, [1, 0]))
    assert result.as_elite == ["logic/medium"]


def test_a_better_candidate_displaces_the_incumbent_in_that_niche_only():
    population = Population()
    population.admit(candidate("a", {"x/easy": 0.5, "x/hard": 0.9}, [1, 1, 0]))
    result = population.admit(candidate("b", {"x/easy": 0.8}, [0, 1, 1]))
    assert result.as_elite == ["x/easy"]
    assert population.elites()["x/hard"].candidate_id == "a"


# ---------------------------------------------------------------------------
# novelty is the second route in
# ---------------------------------------------------------------------------

def test_a_behaviourally_new_candidate_is_kept_even_when_it_wins_nothing():
    """The productive direction usually looks bad at first: it fails
    differently rather than less."""
    population = Population()
    population.admit(candidate("incumbent", {"x/easy": 0.9}, [1, 1, 1, 1]))
    odd = candidate("odd", {"x/easy": 0.2}, [0, 0, 0, 0],
                    steps=("OBSERVE", "ABSTRACT", "DECOMPOSE", "ANSWER"))
    result = population.admit(odd)
    assert result.admitted is True
    assert result.as_novel is True
    assert odd in population.novel()


def test_a_familiar_loser_is_retired():
    population = Population()
    population.admit(candidate("incumbent", {"x/easy": 0.9}, [1, 1, 1, 1]))
    twin = candidate("twin", {"x/easy": 0.4}, [1, 1, 1, 1])
    result = population.admit(twin)
    assert result.admitted is False
    assert population.was_retired("twin")


def test_elite_status_is_checked_before_novelty():
    """Being best at something is a stronger reason to keep a candidate
    than being unusual."""
    population = Population()
    population.admit(candidate("a", {"x/easy": 0.5}, [1, 1, 1]))
    winner = candidate("b", {"x/easy": 0.9}, [1, 1, 1])       # identical behaviour
    result = population.admit(winner)
    assert result.as_elite == ["x/easy"]
    assert result.as_novel is False


def test_a_displaced_elite_that_is_still_novel_is_not_deleted():
    """The single-champion mistake in miniature: the candidate that just
    lost a niche may be the only one exploring a direction."""
    population = Population()
    unusual = candidate("unusual", {"x/easy": 0.5}, [0, 0, 1, 0],
                        steps=("OBSERVE", "ABSTRACT", "ANSWER"))
    population.admit(unusual)
    population.admit(candidate("better", {"x/easy": 0.9}, [1, 1, 1, 1]))
    assert population.was_retired("unusual") is None
    assert any(c.candidate_id == "unusual" for c in population.novel())


def test_the_novelty_pool_is_bounded():
    """An unbounded archive turns novelty pressure into a memory leak."""
    population = Population(novelty_slots=3)
    population.admit(candidate("elite", {"x/easy": 0.9}, [1, 1, 1, 1, 1, 1]))
    for i in range(8):
        pattern = [(i >> k) & 1 for k in range(6)]
        population.admit(candidate(f"n{i}", {"x/easy": 0.1}, pattern,
                                   steps=("OBSERVE", "ABSTRACT", f"OP{i}", "ANSWER")))
    assert len(population.novel()) <= 3


def test_eviction_removes_the_least_novel_not_the_lowest_scoring():
    """Evicting by score would slowly turn the novelty pool into a second,
    worse elite grid."""
    population = Population(novelty_slots=1)
    population.admit(candidate("elite", {"x/easy": 0.9}, [1, 1, 1, 1]))
    near = candidate("near", {"x/easy": 0.85}, [1, 1, 1, 0])
    far = candidate("far", {"x/easy": 0.05}, [0, 0, 0, 0],
                    steps=("ABSTRACT", "DECOMPOSE", "PREDICT"), failures={"format": 4})
    population.admit(near)
    population.admit(far)
    remaining = {c.candidate_id for c in population.novel()}
    assert remaining == {"far"}, "the low scorer was the novel one"


# ---------------------------------------------------------------------------
# what the population is for
# ---------------------------------------------------------------------------

def test_the_best_program_for_a_niche_comes_from_that_niche():
    population = Population()
    population.admit(candidate("math", {"arithmetic/hard": 0.9}, [1, 1]))
    population.admit(candidate("logic", {"logic/hard": 0.95}, [0, 1]))
    picked = population.best_for("arithmetic", 0.9)
    assert picked.candidate_id == "math"


def test_an_empty_niche_falls_back_within_its_domain_not_globally():
    """A program that wins on easy arithmetic is a better guess for medium
    arithmetic than one that wins on hard logic."""
    population = Population()
    population.admit(candidate("math-easy", {"arithmetic/easy": 0.7}, [1, 1]))
    population.admit(candidate("logic-hard", {"logic/hard": 0.99}, [0, 1]))
    picked = population.best_for("arithmetic", 0.5)
    assert picked.candidate_id == "math-easy"


def test_nothing_is_returned_for_a_domain_with_no_members():
    assert Population().best_for("text_ops", 0.5) is None


# ---------------------------------------------------------------------------
# coverage is the measurable part of "expanded space"
# ---------------------------------------------------------------------------

def test_coverage_separates_occupied_from_competent():
    """Filling a cell with something that usually fails is not covering
    it."""
    population = Population()
    population.admit(candidate("good", {"arithmetic/easy": 0.9}, [1, 1]))
    population.admit(candidate("weak", {"arithmetic/hard": 0.2}, [0, 0]))
    coverage = population.coverage(["arithmetic"])
    assert coverage["occupied"] == 2
    assert coverage["competent"] == 1
    assert "arithmetic/medium" in coverage["empty"]


def test_coverage_grows_as_niches_are_filled():
    """The number that has to move for "MANA can do more than before" to
    mean anything."""
    population = Population()
    before = population.coverage(["arithmetic", "logic"])["occupancy"]
    population.admit(candidate("a", {"arithmetic/easy": 0.8}, [1]))
    population.admit(candidate("b", {"logic/hard": 0.8}, [0]))
    after = population.coverage(["arithmetic", "logic"])["occupancy"]
    assert after > before


def test_diversity_sees_a_population_that_only_looks_like_an_ecology():
    """Coverage alone cannot: one program can occupy every niche."""
    identical = Population()
    for i in range(3):
        identical.admit(candidate(f"same{i}", {f"d{i}/easy": 0.5 + i * 0.1}, [1, 1, 0, 1]))
    varied = Population()
    for i, pattern in enumerate(([1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 1, 0])):
        varied.admit(candidate(f"v{i}", {f"d{i}/easy": 0.5}, pattern,
                               steps=("OBSERVE", f"OP{i}", "ANSWER")))
    assert varied.diversity()["mean_distance"] > identical.diversity()["mean_distance"]


def test_diversity_is_undefined_rather_than_zero_for_a_single_member():
    population = Population()
    population.admit(candidate("only", {"x/easy": 0.5}, [1, 1]))
    assert population.diversity()["mean_distance"] is None


def test_the_report_surfaces_specialists_explicitly():
    population = Population()
    population.admit(candidate("gen", {"a/easy": 0.8, "a/hard": 0.8, "b/easy": 0.8}, [1, 1]))
    population.admit(candidate("spec", {"c/hard": 0.95}, [0, 1]))
    specialists = [s["candidate"] for s in population.report()["specialists"]]
    assert "spec" in specialists


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_the_population_survives_a_round_trip(tmp_path):
    population = Population(novelty_slots=5)
    population.admit(candidate("elite", {"arithmetic/hard": 0.9}, [1, 1, 0]))
    population.admit(candidate("odd", {"arithmetic/hard": 0.1}, [0, 0, 1],
                               steps=("ABSTRACT", "PREDICT")))
    path = tmp_path / "population.json"
    population.save(path)
    restored = Population.load(path)
    assert restored.elites()["arithmetic/hard"].candidate_id == "elite"
    assert restored.novelty_slots == 5
    assert len(restored) == len(population)


def test_a_corrupt_population_file_does_not_stop_mana(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(Population.load(path)) == 0


def test_niches_are_named_the_same_way_the_self_model_names_them():
    """Two naming schemes for the same slice would make coverage and
    capability impossible to line up."""
    from mana.cognition.self_model import band_of
    assert niche_of("arithmetic", 0.9) == f"arithmetic/{band_of(0.9)}"
