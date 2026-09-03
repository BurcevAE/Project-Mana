"""
tests/test_transfer_novelty.py — the two things that separate a discovery
from a coincidence.

Transfer asks whether a mechanism works where it has never been. Novelty
asks whether it is actually new. Without both, a search rediscovers its
own previous findings and calls domain-specific tricks mechanisms.
"""
from __future__ import annotations

import pytest

from mana.cognition import novelty as nov
from mana.cognition import transfer as tr
from mana.cognition.novelty import Behaviour, NoveltyArchive
from mana.core.gates import PairedOutcome


def outcomes(n, base_correct, cand_correct, domain="logic"):
    """Paired outcomes at the requested marginal rates."""
    rows = []
    for i in range(n):
        rows.append(PairedOutcome(f"t{i}", domain,
                                  i < n * base_correct, i < n * cand_correct))
    return rows


def behaviour(cid, steps, pattern, failures=None, calls=2.0, latency=1.0, label=""):
    return Behaviour(cid, tuple(steps), tuple(bool(c) for c in pattern),
                     failures or {}, calls, latency, label)


# ---------------------------------------------------------------------------
# transfer: strength of evidence is kept apart
# ---------------------------------------------------------------------------

def test_an_effect_that_survives_elsewhere_transfers():
    result = tr.evaluate_domain("logic", tr.CROSS_DOMAIN,
                                outcomes(30, 0.40, 0.60), source_effect=0.20)
    assert result.transferred is True
    assert result.retention == pytest.approx(1.0, abs=0.05)


def test_an_effect_that_shrinks_to_nothing_does_not_transfer():
    """+0.20 at home and +0.02 elsewhere is a domain-specific trick that
    happens not to hurt."""
    result = tr.evaluate_domain("logic", tr.CROSS_DOMAIN,
                                outcomes(30, 0.50, 0.52), source_effect=0.20)
    assert result.transferred is False
    assert "сохранилось" in result.note


def test_a_target_effect_too_small_to_matter_is_not_transfer():
    """The absolute floor, which a retention ratio alone cannot provide: a
    tiny source effect makes almost any target effect look like excellent
    retention. (Whether a source effect that small should have produced a
    law at all is the acceptance gate's question, not this module's.)"""
    # 15/30 vs 15/30 plus one: an effect of 0.0 that a generous ratio
    # could still dress up if the source effect were small enough.
    rows = outcomes(30, 0.50, 0.50)
    result = tr.evaluate_domain("logic", tr.CROSS_DOMAIN, rows, source_effect=0.05)
    assert result.effect < tr.MIN_ABSOLUTE_EFFECT
    assert result.transferred is False
    assert "ниже порога" in result.note


def test_too_few_trials_is_not_a_transfer_failure():
    result = tr.evaluate_domain("logic", tr.CROSS_DOMAIN,
                                outcomes(5, 0.40, 0.80), source_effect=0.20)
    assert result.transferred is False
    assert "мало испытаний" in result.note


def test_the_source_domain_is_never_counted_as_a_target():
    """Measuring transfer onto the domain a mechanism came from is
    measuring the thing that produced it."""
    report = tr.measure("arithmetic", 0.2,
                        lambda d, n: outcomes(n, 0.4, 0.8, d),
                        cross_domains=("arithmetic", "sequence"),
                        held_out_domains=("arithmetic", "logic"))
    assert "arithmetic" not in [r.domain for r in report.results]
    assert {r.domain for r in report.results} == {"sequence", "logic"}


def test_held_out_confirmation_is_worth_more_than_cross_domain():
    """Weighting them equally lets cheap evidence substitute for the
    expensive kind."""
    cross_only = tr.measure("arithmetic", 0.2, lambda d, n: outcomes(n, 0.4, 0.58, d),
                            cross_domains=("sequence", "code"))
    with_held_out = tr.measure("arithmetic", 0.2, lambda d, n: outcomes(n, 0.4, 0.58, d),
                               cross_domains=("sequence",), held_out_domains=("logic",))
    assert with_held_out.held_out_confirmed == ["logic"]
    assert cross_only.held_out_confirmed == []
    assert "только development" in cross_only.verdict()


def test_the_verdict_says_which_kind_of_evidence_was_found():
    report = tr.measure("arithmetic", 0.2, lambda d, n: outcomes(n, 0.4, 0.58, d),
                        held_out_domains=("logic", "text_ops"))
    assert "скрытых доменах" in report.verdict()


def test_no_transfer_anywhere_is_reported_plainly():
    report = tr.measure("arithmetic", 0.2, lambda d, n: outcomes(n, 0.5, 0.5, d),
                        cross_domains=("sequence",), held_out_domains=("logic",))
    assert report.confirmed == []
    assert "не подтверждён" in report.verdict()
    assert report.score() == 0.0


def test_a_domain_that_cannot_be_run_does_not_abort_the_measurement():
    def flaky(domain, n):
        if domain == "sequence":
            raise RuntimeError("brain died")
        return outcomes(n, 0.4, 0.58, domain)

    report = tr.measure("arithmetic", 0.2, flaky,
                        cross_domains=("sequence", "code"))
    assert len(report.results) == 2
    assert report.confirmed == ["code"]


def test_targets_come_from_the_cores_own_split_definition():
    """A change to what counts as held out must not silently disagree with
    what transfer measures against."""
    from mana.core import splits
    cross, held_out = tr.default_targets("arithmetic")
    assert set(held_out) == set(splits.TRANSFER_DOMAINS)
    assert "arithmetic" not in cross and "arithmetic" not in held_out


# ---------------------------------------------------------------------------
# novelty: behaviour dominates
# ---------------------------------------------------------------------------

def test_the_same_idea_written_two_ways_is_not_novel():
    """A search that cannot see this rediscovers its own findings, each
    time spending a full experiment budget."""
    archive = NoveltyArchive([
        behaviour("a", ["GENERATE", "CRITIQUE", "REPAIR"], [1, 0, 1, 1, 0, 1], label="chain")])
    composed = behaviour("b", ["GENERATE", "CRITIQUE", "REPAIR"], [1, 0, 1, 1, 0, 1],
                         label="composite")
    verdict = archive.assess(composed)
    assert verdict.is_novel is False
    assert verdict.nearest_label == "chain"


def test_the_same_shape_behaving_differently_is_novel():
    """Two chains differing by one operator can succeed and fail on
    completely different tasks."""
    archive = NoveltyArchive([
        behaviour("a", ["GENERATE", "CRITIQUE"], [1, 1, 1, 0, 0, 0], label="first")])
    twin = behaviour("b", ["GENERATE", "CRITIQUE"], [0, 0, 0, 1, 1, 1], label="second")
    verdict = archive.assess(twin)
    assert verdict.is_novel is True
    assert verdict.channels["behaviour"] == 1.0


def test_the_same_accuracy_with_different_failures_reads_as_different():
    """Accuracy alone cannot see that one fails on format and the other on
    arithmetic."""
    a = behaviour("a", ["GENERATE"], [1, 1, 0, 0], {"format": 2})
    b = behaviour("b", ["GENERATE"], [1, 1, 0, 0], {"wrong": 2})
    result = nov.distance(a, b)
    assert result["channels"]["failure_profile"] == pytest.approx(1.0)
    assert result["distance"] > 0


def test_behaviour_outweighs_structure():
    """The weighting is the whole design: only behaviour recognises the
    same idea written two ways."""
    assert nov.CHANNEL_WEIGHTS["behaviour"] > nov.CHANNEL_WEIGHTS["structure"]
    assert sum(nov.CHANNEL_WEIGHTS.values()) == pytest.approx(1.0)


def test_incomparable_outcome_vectors_are_not_scored_as_identical():
    """Padding would silently invent agreement on tasks one candidate
    never attempted -- the direction that makes everything look the same."""
    a = behaviour("a", ["GENERATE"], [1, 1, 0])
    b = behaviour("b", ["GENERATE"], [1, 1, 0, 1, 0])
    assert nov.behaviour_distance(a, b) is None
    result = nov.distance(a, b)
    assert result["behaviour_comparable"] is False
    assert "behaviour" not in result["channels"]


def test_novelty_is_distance_to_the_nearest_not_the_average():
    """Averaging makes a candidate that duplicates one archive member
    exactly look novel as long as it differs from the rest."""
    archive = NoveltyArchive([
        behaviour("dup", ["GENERATE"], [1, 1, 1, 1], label="duplicate"),
        behaviour("far1", ["ABSTRACT", "PREDICT", "COMPARE"], [0, 0, 0, 0],
                  {"wrong": 4}, calls=9, latency=9),
        behaviour("far2", ["DECOMPOSE", "SYNTHESIZE"], [0, 0, 0, 0],
                  {"format": 4}, calls=9, latency=9),
    ])
    candidate = behaviour("new", ["GENERATE"], [1, 1, 1, 1])
    verdict = archive.assess(candidate)
    assert verdict.is_novel is False
    assert verdict.nearest == "dup"


def test_the_first_candidate_is_always_novel():
    assert NoveltyArchive().assess(behaviour("a", ["GENERATE"], [1])).is_novel is True


def test_a_candidate_is_not_compared_against_itself():
    archive = NoveltyArchive([behaviour("same", ["GENERATE"], [1, 1, 0])])
    verdict = archive.assess(behaviour("same", ["GENERATE"], [1, 1, 0]))
    assert verdict.is_novel is True


def test_the_verdict_says_which_channel_made_it_novel():
    """"Novel because it behaves differently" and "novel because it is
    written differently" are different findings, and only the first is
    worth an experiment."""
    archive = NoveltyArchive([behaviour("a", ["GENERATE"], [1, 1, 1, 0])])
    verdict = archive.assess(behaviour("b", ["GENERATE"], [0, 0, 0, 1]))
    assert "behaviour" in verdict.reason


def test_the_archive_keeps_rejected_candidates_too():
    """A duplicate today may be the neighbour that makes tomorrow's
    candidate look familiar."""
    archive = NoveltyArchive()
    archive.add(behaviour("a", ["GENERATE"], [1, 1]))
    archive.add(behaviour("b", ["GENERATE"], [1, 1]))
    assert len(archive) == 2


def test_the_archive_reports_where_the_search_repeats_itself():
    archive = NoveltyArchive([
        behaviour("a", ["GENERATE"], [1, 1, 0], label="one"),
        behaviour("b", ["GENERATE"], [1, 1, 0], label="two"),
        behaviour("c", ["ABSTRACT", "DECOMPOSE"], [0, 0, 1], {"wrong": 2},
                  calls=8, label="three"),
    ])
    closest = archive.most_similar_pairs(limit=1)[0]
    assert {closest["a"], closest["b"]} == {"one", "two"}


def test_the_archive_survives_a_round_trip(tmp_path):
    archive = NoveltyArchive([behaviour("a", ["GENERATE", "VERIFY"], [1, 0, 1],
                                        {"wrong": 1}, 2.0, 1.5, "probe")])
    path = tmp_path / "novelty.json"
    archive.save(path)
    restored = NoveltyArchive.load(path)
    assert len(restored) == 1
    assert restored.all()[0].steps == ("GENERATE", "VERIFY")
    assert restored.all()[0].failure_reasons == {"wrong": 1}


def test_behaviour_is_built_from_the_same_grades_everything_else_uses():
    """A novelty channel judged by a different standard from the
    acceptance gate would let a candidate be novel and useless at once."""
    from mana.core import oracle, tasks
    generated = tasks.generate("arithmetic", 5, seed=11)
    grades = [oracle.grade(t, str(t.answer)) for t in generated[:3]] + \
             [oracle.grade(t, "неверно") for t in generated[3:]]
    built = nov.behaviour_from_run("cand", ["OBSERVE", "GENERATE"], grades,
                                   calls=[1] * 5, latencies=[0.5] * 5)
    assert built.outcomes == (True, True, True, False, False)
    assert sum(built.failure_reasons.values()) == 2
