"""
tests/test_cognitive_genome.py — the space of what MANA can think with.

Two properties are load-bearing here and the rest is bookkeeping:

  1. a mutation can add a member the space did not previously contain
     (otherwise this is PipelineSpec with more fields);
  2. nothing becomes the current genome without a verdict that belongs to
     it (otherwise the gate is decoration).
"""
from __future__ import annotations

import pytest

from mana.cognition import genome as g
from mana.cognition import ir
from mana.core import gates


def base():
    return g.CognitiveGenome()


def accepted_verdict():
    return gates.Verdict(accepted=True, reason="accepted")


# ---------------------------------------------------------------------------
# the starting genome describes MANA as it is
# ---------------------------------------------------------------------------

def test_the_starting_genome_is_valid_and_describes_existing_behaviour():
    """A genome whose operators nothing implements would make every early
    measurement meaningless."""
    genome = base()
    assert genome.validate() == []
    assert "GENERATE" in genome.operators
    assert genome.operators["GENERATE"].derived_from == ("graph_nodes:LLM",)
    assert "critique_loop" in genome.program_templates


def test_every_baseline_template_type_checks():
    genome = base()
    for name, tmpl in genome.program_templates.items():
        chain = [genome.operators[s] for s in tmpl.steps]
        ir.check_chain(chain)          # raises if the chain cannot run


def test_declared_cost_and_measured_evidence_are_kept_apart():
    """cost='high' is an opinion; 180 measured runs is a fact. A planner
    that cannot tell them apart trusts the opinion forever."""
    op = base().operators["GENERATE"]
    assert op.cost == "high"
    assert op.evidence.is_measured is False
    assert op.evidence.success_rate is None


# ---------------------------------------------------------------------------
# composition: where a new operator comes from
# ---------------------------------------------------------------------------

def test_a_chain_that_cannot_run_is_refused():
    """CRITIQUE before anything has drafted has no draft to judge."""
    genome = base()
    with pytest.raises(ir.CompositionError, match="CRITIQUE"):
        ir.check_chain([genome.operators["CRITIQUE"]])


def test_composing_a_chain_produces_a_usable_new_operator():
    genome = base()
    chain = [genome.operators[s] for s in ("GENERATE", "CRITIQUE", "REPAIR")]
    new_op = ir.compose("COUNTERFACTUAL_REFINEMENT", chain)
    assert new_op.implementation == "composite"
    assert new_op.components == ("GENERATE", "CRITIQUE", "REPAIR")
    assert ir.DRAFT in new_op.outputs


def test_composite_uncertainty_compounds_rather_than_averaging():
    """Four steps at 0.5 do not give 0.5. A planner told otherwise keeps
    choosing long chains."""
    genome = base()
    chain = [genome.operators[s] for s in ("GENERATE", "CRITIQUE", "REPAIR")]
    composite = ir.compose("X", chain)
    assert composite.uncertainty > max(op.uncertainty for op in chain)


def test_a_composite_inherits_every_way_its_parts_can_fail():
    genome = base()
    chain = [genome.operators[s] for s in ("RETRIEVE", "GENERATE")]
    composite = ir.compose("X", chain)
    assert "retrieves plausible but irrelevant material" in composite.failure_modes
    assert "fabricates specifics" in composite.failure_modes


def test_structural_identity_ignores_the_name():
    """Two operators reached by different search paths that do the same
    thing are the same idea; novelty will need to say so."""
    genome = base()
    chain = [genome.operators[s] for s in ("GENERATE", "CRITIQUE")]
    assert ir.compose("ALPHA", chain).signature() == ir.compose("BETA", chain).signature()


# ---------------------------------------------------------------------------
# mutation expands the space
# ---------------------------------------------------------------------------

def test_composition_adds_a_member_the_space_did_not_contain():
    """The difference between open-endedness and tuning, in one assertion."""
    parent = base()
    proposal = g.propose(parent, "compose_operators",
                         steps=("GENERATE", "CRITIQUE", "REPAIR"),
                         op_id="COUNTERFACTUAL_REFINEMENT",
                         rationale="critique before answering may help on hard tasks")
    assert proposal.expands_space is True
    assert proposal.candidate.size()["operators"] == parent.size()["operators"] + 1
    assert proposal.summary()["delta"]["composite_operators"] == 1


def test_a_new_operator_can_itself_be_composed_further():
    """Without this the vocabulary grows once and stops."""
    parent = base()
    first = g.propose(parent, "compose_operators", steps=("GENERATE", "CRITIQUE"),
                      op_id="SELF_CHECK").candidate
    second = g.propose(first, "compose_operators", steps=("SELF_CHECK", "REPAIR"),
                       op_id="SELF_CORRECT")
    assert "SELF_CORRECT" in second.candidate.operators
    assert second.candidate.operators["SELF_CORRECT"].components == ("SELF_CHECK", "REPAIR")


def test_a_new_representation_is_a_level_three_change():
    parent = base()
    proposal = g.propose(parent, "create_representation",
                         name="causal_view",
                         fields=("state", "transition", "constraint", "invariant"),
                         rationale="the current task_view cannot express invariants")
    assert proposal.expands_space is True
    assert proposal.candidate.size()["representation_fields"] > parent.size()["representation_fields"]


def test_widening_an_existing_representation_counts_as_expansion():
    parent = base()
    current = parent.representations["task_view"]
    proposal = g.propose(parent, "modify_representation", name="task_view",
                         fields=current.fields + ("counterexample",))
    assert proposal.expands_space is True
    assert proposal.candidate.representations["task_view"].derived_from == "task_view"


def test_narrowing_one_is_a_mutation_but_not_an_expansion():
    parent = base()
    proposal = g.propose(parent, "modify_representation", name="attempt",
                         fields=("draft", "verification"))
    assert proposal.expands_space is False


# ---------------------------------------------------------------------------
# malformed proposals cost nothing
# ---------------------------------------------------------------------------

def test_an_unknown_mutation_is_refused():
    with pytest.raises(g.GenomeError, match="unknown mutation"):
        g.propose(base(), "improve_everything")


def test_composing_unknown_steps_is_refused_before_any_measurement():
    """Measuring a nonsensical candidate wastes the budget a real one
    needed."""
    with pytest.raises(g.GenomeError, match="unknown steps"):
        g.propose(base(), "compose_operators", steps=("GENERATE", "TELEPATHY"), op_id="X")


def test_composing_an_untypeable_chain_is_refused():
    with pytest.raises(g.GenomeError):
        g.propose(base(), "compose_operators", steps=("REPAIR", "GENERATE"), op_id="X")


def test_an_operator_still_used_by_a_template_cannot_be_removed():
    with pytest.raises(g.GenomeError, match="used by templates"):
        g.propose(base(), "remove_operator", op_id="GENERATE")


def test_an_operator_that_something_is_built_from_cannot_be_removed():
    """COUNTEREXAMPLE is in no baseline template, so only the component
    guard can stop this -- which is what the test is for."""
    parent = g.propose(base(), "compose_operators", steps=("GENERATE", "COUNTEREXAMPLE"),
                       op_id="ADVERSARIAL_DRAFT").candidate
    with pytest.raises(g.GenomeError, match="component of"):
        g.propose(parent, "remove_operator", op_id="COUNTEREXAMPLE")


def test_a_learning_rule_cannot_be_pushed_outside_its_bounds():
    """A rule that can set its own bounds is not a rule."""
    proposal = g.propose(base(), "modify_learning_rule", name="brain_reputation", value=9.0)
    assert proposal.candidate.learning_rules["brain_reputation"].value == 0.60


def test_a_genome_may_not_redefine_what_the_core_means_by_a_name():
    """Not a safety list -- a circularity list. A genome that could
    redefine VERIFY could improve its score by changing what verification
    means."""
    reserved = ir.CognitiveOperator("oracle", (ir.TASK,), (ir.ANSWER,))
    with pytest.raises(g.GenomeError, match="reserved"):
        g.propose(base(), "add_operator", operator=reserved)


# ---------------------------------------------------------------------------
# nothing is adopted without a verdict that belongs to it
# ---------------------------------------------------------------------------

def test_a_proposal_is_not_a_change():
    parent = base()
    g.propose(parent, "compose_operators", steps=("GENERATE", "CRITIQUE"), op_id="SELF_CHECK")
    assert "SELF_CHECK" not in parent.operators


def test_adoption_requires_an_accepted_verdict():
    proposal = g.propose(base(), "compose_operators", steps=("GENERATE", "CRITIQUE"),
                         op_id="SELF_CHECK")
    rejected = gates.Verdict(accepted=False, reason="failed: significance")
    with pytest.raises(g.NotAccepted, match="not accepted"):
        g.apply(proposal, rejected, proposal.proposal_id)


def test_a_verdict_earned_by_one_candidate_cannot_admit_another():
    """The cheapest imaginable way to smuggle a change past the gate."""
    parent = base()
    measured = g.propose(parent, "compose_operators", steps=("GENERATE", "CRITIQUE"),
                         op_id="SELF_CHECK")
    smuggled = g.propose(parent, "compose_operators", steps=("GENERATE", "VERIFY"),
                         op_id="SOMETHING_ELSE")
    with pytest.raises(g.NotAccepted, match="belongs to claim"):
        g.apply(smuggled, accepted_verdict(), measured.proposal_id)


def test_an_accepted_proposal_becomes_the_new_genome_with_traceable_lineage():
    parent = base()
    proposal = g.propose(parent, "compose_operators", steps=("GENERATE", "CRITIQUE", "REPAIR"),
                         op_id="COUNTERFACTUAL_REFINEMENT", rationale="hard tasks")
    adopted = g.apply(proposal, accepted_verdict(), proposal.proposal_id)
    assert "COUNTERFACTUAL_REFINEMENT" in adopted.operators
    assert adopted.parent_id == parent.genome_id
    assert adopted.mutation == "compose_operators"


# ---------------------------------------------------------------------------
# persistence: a capability must survive a restart
# ---------------------------------------------------------------------------

def test_a_genome_survives_a_round_trip_unchanged():
    """§37 requires a discovered capability to still exist after MANA is
    restarted. If the genome cannot be reloaded exactly, it does not."""
    parent = base()
    evolved = g.propose(parent, "compose_operators",
                        steps=("GENERATE", "CRITIQUE", "REPAIR"),
                        op_id="COUNTERFACTUAL_REFINEMENT").candidate
    restored = g.CognitiveGenome.from_dict(evolved.to_dict())
    assert restored.signature() == evolved.signature()
    assert restored.genome_id == evolved.genome_id
    assert restored.operators["COUNTERFACTUAL_REFINEMENT"].components == \
        evolved.operators["COUNTERFACTUAL_REFINEMENT"].components


def test_structurally_identical_genomes_share_a_signature():
    """Two search paths that arrive at the same space arrived at the same
    genome, whatever their ids say."""
    a = g.propose(base(), "compose_operators", steps=("GENERATE", "CRITIQUE"),
                  op_id="SELF_CHECK").candidate
    b = g.propose(base(), "compose_operators", steps=("GENERATE", "CRITIQUE"),
                  op_id="SELF_CHECK").candidate
    assert a.genome_id != b.genome_id
    assert a.signature() == b.signature()


def test_size_reports_the_number_that_has_to_grow():
    """"Expanded cognitive space" has to be measurable or it is a slogan."""
    parent = base()
    child = g.propose(parent, "compose_operators", steps=("GENERATE", "CRITIQUE"),
                      op_id="SELF_CHECK").candidate
    assert child.size()["operators"] > parent.size()["operators"]
