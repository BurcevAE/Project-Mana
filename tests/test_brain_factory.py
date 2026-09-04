"""
tests/test_brain_factory.py — deciding what machinery a gap needs.

The factory assembles candidates and accepts nothing. Most of these
tests are about the decisions it makes *before* building, because those
are where the cost is, and about the one thing that would destroy every
verdict this system has ever issued: training on the set that judges it.
"""
from __future__ import annotations

import pytest

from mana.cognition import brain_factory as bf
from mana.cognition import genome as genome_mod
from mana.cognition.brain_factory import (ALGORITHMIC, CLASSICAL_ML, KEEP_MODEL,
                                          BrainCandidate, HoldoutLeak,
                                          MechanismChoice, TrainingSet,
                                          assemble_dataset, assert_no_holdout_leak,
                                          choose_mechanism, needs_new_brain)
from mana.cognition.gaps import COMPETENCE, KNOWLEDGE, detect
from mana.cognition.genome import BrainGene, CognitiveGenome, GenomeError
from mana.cognition.self_model import Observation, SelfModel


def weak_model(domain="arithmetic", n=24, by_brain=None):
    model = SelfModel()
    for i in range(n):
        model.record(Observation(f"t{i}", domain, 0.8, False, reason="wrong",
                                 brain=(by_brain or "m"), calls=1))
    return model


# ---------------------------------------------------------------------------
# the decision made before anything is built
# ---------------------------------------------------------------------------

def test_a_computable_slice_gets_an_algorithm_not_a_model():
    """A slice whose answer is computable does not need a model that
    approximates it."""
    assert choose_mechanism("arithmetic", exactly_computable=True).mechanism == ALGORITHMIC


def test_too_few_examples_means_keep_the_model():
    """Below the threshold a fit memorises the sample and its
    cross-validation score describes that sample."""
    choice = choose_mechanism("code", examples=200, exactly_computable=False)
    assert choice.mechanism == KEEP_MODEL
    assert "запомнит выборку" in choice.reason


def test_enough_examples_and_no_exact_solver_means_classical_ml():
    assert choose_mechanism("x", examples=5000,
                            exactly_computable=False).mechanism == CLASSICAL_ML


def test_the_rule_answers_algorithmic_for_four_of_manas_five_domains():
    """A finding about the task set, recorded as a test so that it is
    noticed if it ever stops being true."""
    computable = {"arithmetic": True, "sequence": True, "logic": True,
                  "text_ops": True, "code": False}
    chosen = {d: choose_mechanism(d, examples=200, exactly_computable=e).mechanism
              for d, e in computable.items()}
    assert sum(1 for m in chosen.values() if m == ALGORITHMIC) == 4
    assert chosen["code"] == KEEP_MODEL


# ---------------------------------------------------------------------------
# three conditions, all of them
# ---------------------------------------------------------------------------

def test_an_unsettled_gap_builds_nothing():
    model = SelfModel()
    model.record(Observation("t", "arithmetic", 0.8, False, reason="wrong"))
    gap = detect(model)[0]
    decision = needs_new_brain(gap, model, expected_saving=1.0)
    assert decision.build is False
    assert decision.checks["gap_settled"] is False


def test_distinguishable_brains_mean_routing_not_building():
    """Building instead would spend a training budget to rediscover
    something the router could have used for free."""
    model = weak_model()
    gap = detect(model)[0]
    decision = needs_new_brain(gap, model,
                               existing_scores={"a": 0.2, "b": 0.9},
                               expected_saving=1.0)
    assert decision.build is False
    assert "маршрутизация" in decision.reason


def test_an_uncomputed_saving_is_not_a_justification():
    """Could not even be stated before cost stopped being counted in
    "calls"."""
    model = weak_model()
    gap = detect(model)[0]
    assert needs_new_brain(gap, model, expected_saving=None).build is False


def test_all_three_conditions_together_allow_building():
    model = weak_model()
    gap = detect(model)[0]
    decision = needs_new_brain(gap, model, existing_scores={"a": 0.2, "b": 0.25},
                               expected_saving=0.5)
    assert decision.build is True
    assert all(decision.checks.values())


# ---------------------------------------------------------------------------
# the one leak that would destroy every verdict
# ---------------------------------------------------------------------------

def test_a_clean_dataset_passes_the_leak_check(isolated_config):
    model = weak_model()
    texts = {f"t{i}": f"Вычисли: {i} + 1" for i in range(24)}
    data = assemble_dataset(model, "arithmetic", texts, generated=10, seed=4242)
    assert_no_holdout_leak(data)
    assert len(data) > 0


def test_a_holdout_task_in_the_dataset_is_caught(isolated_config):
    """"Unreachable" is an argument; a test is a fact."""
    from mana.core import splits
    hidden = splits.generate_mixed(3, splits._SEED_HIDDEN, splits.DEVELOPMENT_DOMAINS)
    data = TrainingSet(domain="arithmetic")
    data.prompts.append(hidden[0].prompt)
    data.answers.append("")
    with pytest.raises(HoldoutLeak):
        assert_no_holdout_leak(data, per_domain=3)


def test_the_leak_check_never_returns_the_hidden_tasks():
    """It regenerates them inside itself and hands back nothing."""
    import inspect
    source = inspect.getsource(assert_no_holdout_leak)
    assert "return" not in source.replace("returns", "")


def test_failures_are_kept_because_they_are_the_valuable_part(isolated_config):
    """A brain trained only on what the system already gets right learns
    the easy half of the slice and is then measured on the hard half."""
    model = SelfModel()
    for i in range(6):
        model.record(Observation(f"ok{i}", "arithmetic", 0.3, True, calls=1))
        model.record(Observation(f"bad{i}", "arithmetic", 0.9, False,
                                 reason="wrong", calls=1))
    texts = {f"ok{i}": f"Вычисли: {i} + 1" for i in range(6)}
    texts.update({f"bad{i}": f"Вычисли: ({i} + 1) * 7" for i in range(6)})
    data = assemble_dataset(model, "arithmetic", texts)
    assert data.sources["real_failure"] == 6
    assert data.sources["real_correct"] == 6


# ---------------------------------------------------------------------------
# adoption, and how narrow it is
# ---------------------------------------------------------------------------

def test_a_brain_is_adopted_only_for_the_slice_it_was_proven_on(isolated_config):
    gene = BrainGene(brain_id="b1", substrate=ALGORITHMIC,
                     applicability=("arithmetic/hard",))
    proposal = genome_mod.propose(CognitiveGenome(), "create_brain",
                                  rationale="test", gene=gene)
    assert proposal.candidate.brains["b1"].applicability == ("arithmetic/hard",)
    assert proposal.expands_space is True


def test_a_brain_claiming_no_slice_is_refused(isolated_config):
    """Claiming nothing is the widest claim there is."""
    gene = BrainGene(brain_id="b1", substrate=ALGORITHMIC)
    with pytest.raises(GenomeError):
        genome_mod.propose(CognitiveGenome(), "create_brain", gene=gene)


def test_the_factory_cannot_adopt_without_a_verdict(isolated_config):
    candidate = BrainCandidate(brain_id="b1", substrate=ALGORITHMIC,
                               domain="arithmetic", band="hard",
                               mechanism=MechanismChoice(ALGORITHMIC, "test"))

    class Refused:
        accepted = False
        reason = "не принято"

    with pytest.raises(genome_mod.NotAccepted):
        bf.adopt(candidate, Refused(), CognitiveGenome())


def test_retiring_a_brain_needs_no_gate(isolated_config):
    """Requiring proof to undo a change leaves it in force exactly when
    the proof for it has evaporated."""
    import inspect
    assert "judge" not in inspect.getsource(bf.retire)

    gene = BrainGene(brain_id="b1", substrate=ALGORITHMIC,
                     applicability=("arithmetic/hard",))
    genome = genome_mod.propose(CognitiveGenome(), "create_brain",
                                rationale="t", gene=gene).candidate
    after = bf.retire(gene, genome, "перестал окупаться")
    assert "b1" not in after.brains


def test_a_brain_is_judged_against_the_baseline_it_beat(isolated_config):
    gene = BrainGene(brain_id="b1", substrate=ALGORITHMIC,
                     applicability=("arithmetic/hard",))
    model = SelfModel()
    for i in range(10):
        model.record(Observation(f"t{i}", "arithmetic", 0.8, i < 5,
                                 brain="b1", calls=1))
    assert bf.should_retire(gene, model, baseline=0.30)[0] is False
    assert bf.should_retire(gene, model, baseline=0.90)[0] is True


def test_a_brain_with_too_few_uses_is_not_judged_yet(isolated_config):
    gene = BrainGene(brain_id="b1", substrate=ALGORITHMIC,
                     applicability=("arithmetic/hard",))
    retire_it, reason = bf.should_retire(gene, SelfModel(), baseline=0.9)
    assert retire_it is False
    assert "слишком мало" in reason
