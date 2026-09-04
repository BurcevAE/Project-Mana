"""
tests/test_search.py — the wire that was never connected.

`genome.compose_operators` turns a proven chain into an operator that
can take part in further composition -- the mutation the cognitive layer
was designed around -- and a grep for it outside `genome.py` returned
nothing. The compiler ranked five baseline templates and never looked at
the space they were drawn from.

The test that matters most here is the last one: adopting an operator
must make the reachable space LARGER, or the search is exploring a fixed
space rather than extending one, and the project's central claim is
unsupported.
"""
from __future__ import annotations

import pytest

from mana.cognition import genome as genome_mod
from mana.cognition import search
from mana.cognition.genome import CognitiveGenome
from mana.cognition.ir import TASK, CompositionError, check_chain
from mana.cognition.search import (Candidate, as_template, enumerate_chains,
                                   novelty_against, propose_candidates,
                                   reachable_space)


# ---------------------------------------------------------------------------
# every chain it emits can actually run
# ---------------------------------------------------------------------------

def test_every_enumerated_chain_type_checks():
    """What makes this different from generating strings: a chain that
    reaches the caller is executable, because every step's inputs are
    produced by something before it."""
    genome = CognitiveGenome()
    chains = enumerate_chains(genome, max_depth=4)
    assert chains
    for steps in chains:
        check_chain([genome.operators[s] for s in steps], available=(TASK,))


def test_a_chain_that_cannot_run_is_rejected_not_scored_low():
    """ANSWER needs a draft. A chain that never makes one is not a weak
    program -- no runtime accepts it."""
    genome = CognitiveGenome()
    with pytest.raises(CompositionError):
        check_chain([genome.operators[s] for s in ("OBSERVE", "ABSTRACT", "ANSWER")],
                    available=(TASK,))
    assert ("OBSERVE", "ABSTRACT", "ANSWER") not in enumerate_chains(genome, max_depth=4)


def test_every_chain_starts_by_looking_and_ends_by_answering():
    for steps in enumerate_chains(CognitiveGenome(), max_depth=4):
        assert steps[0] == "OBSERVE"
        assert steps[-1] == "ANSWER"


def test_order_matters_so_permutations_are_enumerated():
    """OBSERVE→GENERATE→CRITIQUE and OBSERVE→CRITIQUE→GENERATE are
    different programs, and only one of them can run."""
    chains = set(enumerate_chains(CognitiveGenome(), max_depth=4))
    assert ("OBSERVE", "GENERATE", "CRITIQUE", "ANSWER") in chains
    assert ("OBSERVE", "CRITIQUE", "GENERATE", "ANSWER") not in chains


# ---------------------------------------------------------------------------
# what it proposes, and what it refuses to propose
# ---------------------------------------------------------------------------

def test_a_chain_the_genome_already_has_is_not_proposed():
    """A duplicate is not a weak candidate -- it is the same thing again,
    and testing it twice splits the evidence for one mechanism."""
    genome = CognitiveGenome()
    existing = {tuple(t.steps) for t in genome.program_templates.values()}
    proposed = {c.steps for c in propose_candidates(genome, limit=50)}
    assert not (proposed & existing)


def test_candidates_are_ranked_by_novelty_per_call():
    """An unfamiliar chain costing six calls has to be six times more
    interesting than one costing one; the experiment budget is what the
    layer is short of."""
    candidates = propose_candidates(CognitiveGenome(), budget_calls=8, limit=20)
    values = [c.novelty / max(1, c.estimated_calls) for c in candidates]
    assert values == sorted(values, reverse=True)


def test_a_chain_over_budget_is_not_proposed():
    assert all(c.estimated_calls <= 3
               for c in propose_candidates(CognitiveGenome(), budget_calls=3, limit=50))


def test_nothing_is_proposed_that_costs_a_model_call_to_find():
    """Enumeration is free by construction. A model asked for "a good
    program" returns plausible ones, and plausible is the failure mode
    this project exists to avoid."""
    import inspect
    source = inspect.getsource(search).lower()
    for forbidden in (".ask(", "llm", "prompt", "brain"):
        assert forbidden not in source


def test_a_discovered_template_claims_no_domain():
    """A chain found by enumeration has been proven nowhere. Claiming a
    domain it was never tested on is the overgeneralisation synthesis
    refuses."""
    candidate = propose_candidates(CognitiveGenome(), limit=1)[0]
    template = as_template(candidate, "found_1")
    assert template.applicability == ()
    assert "Нигде не доказана" in template.description


def test_novelty_is_zero_against_an_identical_chain():
    genome = CognitiveGenome()
    steps = genome.program_templates["critique_loop"].steps
    assert novelty_against(steps, genome) == 0.0


# ---------------------------------------------------------------------------
# the claim the whole project rests on
# ---------------------------------------------------------------------------

def test_adopting_an_operator_enlarges_the_reachable_space():
    """The difference between searching inside a space and extending one.

    Measured, not asserted: one composed operator took the executable
    chains at depth 4 from 13 to 29. If this ever stops being true, the
    cognitive layer is a fixed search and the central claim is
    unsupported.
    """
    genome = CognitiveGenome()
    before = reachable_space(genome, max_depth=4)
    proposal = genome_mod.propose(
        genome, "compose_operators", rationale="найдено перебором",
        steps=("GENERATE", "CRITIQUE", "REPAIR"), op_id="DRAFT_FIX")
    after = reachable_space(proposal.candidate, max_depth=4)
    assert proposal.expands_space is True
    assert after > before, f"space did not grow: {before} -> {after}"


def test_the_new_operator_appears_inside_new_chains():
    """Growth has to come from composition, not from counting the
    operator itself once."""
    genome = CognitiveGenome()
    proposal = genome_mod.propose(
        genome, "compose_operators", rationale="test",
        steps=("GENERATE", "CRITIQUE", "REPAIR"), op_id="DRAFT_FIX")
    chains = enumerate_chains(proposal.candidate, max_depth=4)
    containing = [c for c in chains if "DRAFT_FIX" in c]
    assert len(containing) > 1


def test_the_enumeration_stays_affordable():
    """The space grows as operators^depth. A guard against a future
    depth increase quietly turning search into an all-night job."""
    genome = CognitiveGenome()
    assert len(enumerate_chains(genome, max_depth=4, limit=10000)) < 1000
