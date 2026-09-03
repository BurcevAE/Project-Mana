"""
tests/test_laws.py — laws that can be refuted, and a search that tries to.

The property under test throughout: MANA must be able to overturn its own
conclusions. A system whose laws only ever move up accumulates folklore,
and folklore is indistinguishable from knowledge until it is acted on.
"""
from __future__ import annotations

import json

import pytest

from mana.cognition import counterexamples as ce
from mana.cognition.counterexamples import Probe, design_probes, evaluate_probe, search
from mana.cognition.laws import (MIN_TRIALS_FOR_SUPPORT, CognitiveLaw, Condition,
                                 LawBook, PROPOSED, REFUTED, SUPPORTED, VALIDATED)
from mana.core.gates import PairedOutcome


def a_law(book=None, **kw):
    book = book or LawBook()
    return book.propose(
        condition=kw.get("condition", Condition(domain="arithmetic", min_difficulty=0.65)),
        intervention=kw.get("intervention", ("OBSERVE", "GENERATE", "VERIFY", "ANSWER")),
        claimed_effect=kw.get("claimed_effect", "+0.18 точности"),
        discovered_in=kw.get("discovered_in", "arithmetic"))


def outcomes(n, base_rate, cand_rate, domain="arithmetic"):
    """Paired outcomes with the requested marginal rates."""
    rows = []
    for i in range(n):
        rows.append(PairedOutcome(f"t{i}", domain,
                                  i >= n * (1 - base_rate),
                                  i >= n * (1 - cand_rate)))
    return rows


def promote_to_supported(law, experiment="e1"):
    law.record_evidence(effect=0.18, trials=MIN_TRIALS_FOR_SUPPORT,
                        hidden_confirmed=True, experiment_id=experiment)
    return law


# ---------------------------------------------------------------------------
# a law is a refutable claim
# ---------------------------------------------------------------------------

def test_a_condition_decides_scope_rather_than_being_reinterpreted_later():
    """The commonest way a folk theory survives contact with data."""
    condition = Condition(domain="arithmetic", min_difficulty=0.65)
    assert condition.matches("arithmetic", 0.8) is True
    assert condition.matches("arithmetic", 0.4) is False
    assert condition.matches("logic", 0.8) is False


def test_a_new_law_starts_as_proposed_whatever_it_claims():
    assert a_law().status == PROPOSED


def test_status_cannot_be_set_only_derived():
    """A law cannot be promoted by a component that would like it true."""
    law = a_law()
    assert not hasattr(law, "set_status")
    public = [n for n in dir(law) if not n.startswith("_")]
    assert not any(n.startswith("set_") for n in public)


def test_one_experiment_is_not_enough_for_support():
    law = a_law()
    law.record_evidence(effect=0.4, trials=10, hidden_confirmed=True)
    assert law.status == PROPOSED


def test_support_requires_the_hidden_set_as_well_as_the_visible_one():
    law = a_law()
    law.record_evidence(effect=0.2, trials=50)          # no hidden confirmation
    assert law.status == PROPOSED
    law.record_evidence(hidden_confirmed=True)
    assert law.status == SUPPORTED


def test_validation_requires_both_a_transfer_and_a_search():
    law = promote_to_supported(a_law())
    assert law.status == SUPPORTED
    law.record_evidence(transfer_domain="sequence", transfer_ok=True)
    assert law.status == SUPPORTED, "transfer alone is not validation"
    law.record_evidence(counterexamples_sought=6, counterexamples_found=0)
    assert law.status == VALIDATED


def test_a_law_nobody_probed_cannot_be_validated():
    """A law that was never attacked has survived nothing."""
    law = promote_to_supported(a_law())
    law.record_evidence(transfer_domain="logic", transfer_ok=True)
    assert law.status != VALIDATED


# ---------------------------------------------------------------------------
# demotion is symmetric and automatic
# ---------------------------------------------------------------------------

def test_a_validated_law_is_refuted_by_a_counterexample():
    law = promote_to_supported(a_law())
    law.record_evidence(transfer_domain="sequence", transfer_ok=True,
                        counterexamples_sought=6, counterexamples_found=0)
    assert law.status == VALIDATED
    law.record_evidence(counterexamples_sought=3, counterexamples_found=1)
    assert law.status == REFUTED


def test_a_law_whose_effect_stops_going_its_way_is_refuted():
    """A mean built from +0.4 and -0.3 is not a law; it is two behaviours
    averaged into one misleading number."""
    law = a_law()
    law.record_evidence(effect=0.4, trials=20, hidden_confirmed=True)
    law.record_evidence(effect=-0.3, trials=20)
    law.record_evidence(effect=-0.2, trials=20)
    assert law.evidence.positive_share < 0.5
    assert law.status == REFUTED


def test_hidden_contradictions_outweighing_confirmations_refute_a_law():
    law = promote_to_supported(a_law())
    law.record_evidence(hidden_confirmed=False)
    law.record_evidence(hidden_confirmed=False)
    assert law.status == REFUTED


def test_every_status_change_is_recorded_with_what_caused_it():
    law = a_law()
    promote_to_supported(law, experiment="exp-42")
    assert law.history
    assert law.history[-1]["to"] == SUPPORTED
    assert law.history[-1]["experiment"] == "exp-42"


def test_a_refuted_law_is_kept_but_never_acted_on():
    """A system that deletes its failures repeats them."""
    book = LawBook()
    law = a_law(book)
    law.record_evidence(effect=-0.3, trials=40, hidden_confirmed=False)
    assert law.status == REFUTED
    assert law in book.all()
    assert book.applicable("arithmetic", 0.8) == []


def test_an_exception_is_not_a_refutation():
    """A law that holds except under a stated condition is more useful
    than one deleted for being imperfect."""
    law = promote_to_supported(a_law())
    law.add_exception("не работает без вычислимого оракула")
    assert law.status == SUPPORTED
    assert law.exceptions == ["не работает без вычислимого оракула"]


def test_the_book_survives_a_round_trip(tmp_path):
    book = LawBook()
    law = promote_to_supported(a_law(book))
    law.add_exception("нет оракула")
    path = tmp_path / "laws.json"
    book.save(path)
    restored = LawBook.load(path)
    back = restored.get(law.law_id)
    assert back.status == SUPPORTED
    assert back.exceptions == ["нет оракула"]
    assert back.condition.min_difficulty == 0.65


# ---------------------------------------------------------------------------
# probes are derived from the law, not sampled at random
# ---------------------------------------------------------------------------

DOMAINS = ("arithmetic", "sequence", "logic", "text_ops")


def test_probes_include_the_boundary_the_law_claims():
    """If the effect is identical on both sides, the stated boundary is
    not where the mechanism changes."""
    probes = design_probes(a_law(), DOMAINS)
    boundary = [p for p in probes if p.kind == ce.BOUNDARY]
    assert boundary
    assert any(p.difficulty < 0.65 for p in boundary)


def test_probes_reach_domains_the_law_does_not_claim():
    probes = design_probes(a_law(), DOMAINS)
    adjacent = {p.domain for p in probes if p.kind == ce.ADJACENT_DOMAIN}
    assert "sequence" in adjacent and "logic" in adjacent
    assert "arithmetic" not in adjacent


def test_probes_come_from_the_operators_own_declared_failure_modes():
    """Not a guess: the law's own machinery names them."""
    from mana.cognition.ir import primitive_operators
    law = a_law(intervention=("OBSERVE", "GENERATE", "CRITIQUE", "ANSWER"))
    probes = design_probes(law, DOMAINS, primitive_operators())
    operator_probes = [p for p in probes if p.kind == ce.OPERATOR_FAILURE]
    assert operator_probes
    assert any("self-review" in p.rationale for p in operator_probes)


def test_the_cheapest_and_likeliest_probes_come_first():
    """A budget that runs out partway should have spent itself where a
    finding was likeliest."""
    probes = design_probes(a_law(), DOMAINS)
    kinds = [p.kind for p in probes]
    assert kinds.index(ce.BOUNDARY) < kinds.index(ce.ADJACENT_DOMAIN)


# ---------------------------------------------------------------------------
# what a probe finds
# ---------------------------------------------------------------------------

def test_a_reversed_effect_is_a_counterexample():
    probe = Probe("p", ce.ADJACENT_DOMAIN, Condition(domain="logic"), "why", "logic", 0.8)
    result = evaluate_probe(probe, outcomes(20, 0.8, 0.4), claimed_effect=0.18)
    assert result.is_counterexample is True
    assert "развернулся" in result.note


def test_a_vanished_effect_is_a_limit_not_a_counterexample():
    """Collapsing the two would lose the useful middle."""
    probe = Probe("p", ce.ADJACENT_BAND, Condition(domain="arithmetic"), "why", "arithmetic", 0.3)
    result = evaluate_probe(probe, outcomes(20, 0.5, 0.5), claimed_effect=0.18)
    assert result.is_counterexample is False
    assert result.is_limit is True


def test_a_holding_effect_is_neither():
    probe = Probe("p", ce.BOUNDARY, Condition(domain="arithmetic"), "why", "arithmetic", 0.7)
    result = evaluate_probe(probe, outcomes(20, 0.4, 0.75), claimed_effect=0.18)
    assert not result.is_counterexample and not result.is_limit


def test_one_bad_task_is_noise_not_a_counterexample():
    probe = Probe("p", ce.BOUNDARY, Condition(domain="arithmetic"), "why", "arithmetic", 0.7)
    result = evaluate_probe(probe, outcomes(3, 0.9, 0.0), claimed_effect=0.18)
    assert result.is_counterexample is False
    assert "мало испытаний" in result.note


# ---------------------------------------------------------------------------
# the search folds back into the law
# ---------------------------------------------------------------------------

def test_a_search_that_finds_nothing_still_counts_as_having_looked():
    """It is what separates VALIDATED from merely SUPPORTED."""
    law = promote_to_supported(a_law())
    law.record_evidence(transfer_domain="sequence", transfer_ok=True)
    report = search(law, lambda probe, _law: outcomes(20, 0.4, 0.75), DOMAINS)
    assert report["counterexamples"] == []
    assert law.evidence.counterexamples_sought > 0
    assert law.status == VALIDATED


def test_a_search_that_finds_a_reversal_refutes_the_law():
    law = promote_to_supported(a_law())
    law.record_evidence(transfer_domain="sequence", transfer_ok=True,
                        counterexamples_sought=4, counterexamples_found=0)
    assert law.status == VALIDATED
    report = search(law, lambda probe, _law: outcomes(20, 0.8, 0.3), DOMAINS)
    assert report["counterexamples"]
    assert law.status == REFUTED


def test_limits_are_written_onto_the_law_as_exceptions():
    """An unstated exception is indistinguishable from a wrong law."""
    law = promote_to_supported(a_law())

    def runner(probe, _law):
        return outcomes(20, 0.5, 0.5) if probe.kind == ce.ADJACENT_BAND \
            else outcomes(20, 0.4, 0.75)

    search(law, runner, DOMAINS)
    assert law.exceptions
    assert any("эффект исчез" in e for e in law.exceptions)


def test_a_probe_that_cannot_run_does_not_abort_the_search():
    law = promote_to_supported(a_law())
    calls = {"n": 0}

    def flaky(probe, _law):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("brain died")
        return outcomes(20, 0.4, 0.75)

    report = search(law, flaky, DOMAINS)
    assert report["probes_run"] > 1


def test_the_search_is_bounded():
    law = promote_to_supported(a_law())
    report = search(law, lambda p, l: outcomes(20, 0.4, 0.75), DOMAINS, max_probes=2)
    assert report["probes_run"] == 2
