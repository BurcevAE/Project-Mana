"""
tests/test_synthesis.py — the step that turns a proven mechanism into
something the system uses next time.

Before this module the loop did not close: the gate said SUPPORTED and
the next task compiled from exactly the genome as before. The single
most important test in this file is
`test_after_adoption_the_compiler_actually_picks_the_new_capability`,
because everything else here is machinery in service of that one fact.
"""
from __future__ import annotations

import pytest

from mana.cognition import synthesis
from mana.cognition.compiler import Capabilities, compile_program
from mana.cognition.experiments import Discovery, REFUTED, SUPPORTED
from mana.cognition.genome import CognitiveGenome, NotAccepted
from mana.cognition.programs import Budget
from mana.cognition.self_model import Observation, SelfModel
from mana.cognition.synthesis import (ADOPTED, CapabilityProposal,
                                      CapabilitySynthesizer, PROPOSED, REJECTED,
                                      RETIRED, adopt, applicability_for, confirm,
                                      propose_capability, should_retire)
from mana.core.cost import CostVector
from mana.core.gates import PairedOutcome


#: Deliberately NOT one of the baseline templates -- "critique_loop" is
#: already OBSERVE→GENERATE→CRITIQUE→REPAIR→ANSWER, and proposing it back
#: is correctly refused as a duplicate (see the test for that).
NOVEL_CHAIN = ("OBSERVE", "RETRIEVE", "GENERATE", "VERIFY", "ANSWER")


def discovery(steps=NOVEL_CHAIN,
              domain="arithmetic", band="hard", status=SUPPORTED, margin=0.28):
    return Discovery(
        discovery_id="disc1",
        hypothesis={"domain": domain, "band": band,
                    "baseline_steps": ["OBSERVE", "GENERATE", "ANSWER"],
                    "candidate_steps": list(steps),
                    "statement": "критика перед ответом поможет"},
        measurement={},
        verdict={"accepted": status == SUPPORTED, "reason": "ok",
                 "measurements": {"dev_margin": margin, "dev_baseline": 0.40}},
        status=status)


def outcomes(n=40, baseline_ok=0.4, candidate_ok=0.8):
    """Paired outcomes strong enough to clear every gate."""
    rows = []
    for i in range(n):
        rows.append(PairedOutcome(
            task_id=f"t{i}", domain="arithmetic",
            baseline_correct=(i / n) < baseline_ok,
            candidate_correct=(i / n) < candidate_ok))
    return rows


def confirmed(proposal, **kw):
    return confirm(proposal, outcomes(), hidden=(0.40, 0.72),
                   counterexamples=(3, 0), cost=CostVector(calls=80), **kw)


# ---------------------------------------------------------------------------
# the loop closes
# ---------------------------------------------------------------------------

def test_after_adoption_the_compiler_actually_picks_the_new_capability(isolated_config):
    """The whole point. Proving a chain works and then not using it is
    measurement, not learning."""
    base = CognitiveGenome()
    task = "Вычисли: (91767 - 690) * 86 + 8"
    caps = Capabilities(brains=2, has_memory=True, has_web=True, has_sandbox=True)
    # The known difficulty, not the routing heuristic -- see classify().
    # With the heuristic this task scores 0.0 and a capability proven on
    # the hard band can never fire.
    hard = 0.72
    before = compile_program(task, base, caps, Budget(calls=12), difficulty=hard)

    proposal = propose_capability(discovery(), base)
    assert proposal is not None
    after_genome = adopt(proposal, confirmed(proposal))

    after = compile_program(task, after_genome, caps, Budget(calls=12), difficulty=hard)
    assert after is not None
    assert after.template == proposal.name, (
        f"compiler still chose {after.template!r}; the capability was installed "
        f"but never fires")
    assert before is None or before.template != after.template


def test_the_routing_heuristic_cannot_see_the_hard_band(isolated_config):
    """Two quantities under one name, found by a capability that was
    installed and could never fire. estimate_difficulty answers "does
    this need a big brain?" and subtracts 0.20 for bare arithmetic on
    purpose; the self-model bands on how often the answer is wrong."""
    from mana.cognition.compiler import classify
    from mana.core import tasks
    hard = tasks.generate("arithmetic", 6, seed=3, difficulty_range=(0.65, 1.01))
    assert all(t.difficulty >= 0.65 for t in hard)
    assert all(classify(t.prompt)[1] == 0.0 for t in hard),         "the heuristic scores every hard arithmetic task at zero"
    assert all(classify(t.prompt, t.difficulty)[1] >= 0.65 for t in hard)


def test_a_caller_without_a_known_difficulty_still_gets_an_estimate():
    """Free text from a user has no ground truth, and the heuristic is
    what it was written for."""
    from mana.cognition.compiler import classify
    kind, difficulty = classify("Объясни, почему это спроектировано именно так, "
                                "и сравни с альтернативой")
    assert kind == "reasoning"
    assert difficulty > 0.0


def test_the_genome_the_compiler_reads_is_the_one_that_changed(isolated_config):
    base = CognitiveGenome()
    proposal = propose_capability(discovery(), base)
    adopted = adopt(proposal, confirmed(proposal))
    assert proposal.name in adopted.program_templates
    assert proposal.name not in base.program_templates, "the parent must be untouched"


# ---------------------------------------------------------------------------
# proven where it was measured, nowhere else
# ---------------------------------------------------------------------------

def test_a_capability_claims_only_the_slice_it_was_proven_on():
    assert applicability_for("arithmetic", "hard") == ("math", "hard")
    assert applicability_for("logic", "easy") == ("reasoning", "easy")


def test_a_medium_band_capability_claims_no_band():
    """"medium" is not a word the scorer understands, and inventing one
    it ignores would read as a narrower claim than was installed."""
    assert applicability_for("arithmetic", "medium") == ("math",)


def test_the_domain_is_translated_into_the_compilers_vocabulary():
    """A capability proven on a domain and installed under that domain's
    own name is scored against nothing and never fires."""
    from mana.cognition.compiler import classify
    for domain, kind in synthesis.DOMAIN_KIND.items():
        assert kind in ("math", "sequence", "reasoning", "general", "programming")
    assert classify("Вычисли: 2 + 2")[0] == synthesis.DOMAIN_KIND["arithmetic"]


def test_an_unknown_domain_claims_nothing_rather_than_everything():
    assert applicability_for("telepathy", "medium") == ()


# ---------------------------------------------------------------------------
# one result is not a capability
# ---------------------------------------------------------------------------

def test_a_refuted_discovery_becomes_nothing():
    assert propose_capability(discovery(status=REFUTED), CognitiveGenome()) is None


def test_the_discoverys_own_verdict_cannot_adopt_the_proposal(isolated_config):
    """The same evidence counted twice. `genome.apply` refuses because
    the claim id belongs to the experiment, not to this proposal."""
    base = CognitiveGenome()
    disc = discovery()
    proposal = propose_capability(disc, base)

    class Verdict:
        accepted = True
        reason = "the experiment's own verdict"

    with pytest.raises(NotAccepted):
        from mana.cognition import genome as genome_mod
        genome_mod.apply(proposal.mutation, Verdict(), disc.discovery_id)


def test_a_confirmation_that_fails_the_gate_installs_nothing(isolated_config):
    base = CognitiveGenome()
    proposal = propose_capability(discovery(), base)
    verdict = confirm(proposal, outcomes(n=40, baseline_ok=0.6, candidate_ok=0.6))
    assert not verdict.accepted
    assert proposal.status == REJECTED
    with pytest.raises(NotAccepted):
        adopt(proposal, verdict)


def test_too_few_confirming_trials_is_refused(isolated_config):
    base = CognitiveGenome()
    proposal = propose_capability(discovery(), base)
    verdict = confirm(proposal, outcomes(n=4))
    assert not verdict.accepted
    assert "sample_size" in verdict.failed_gates


def test_confirmation_leaves_a_finished_transaction(isolated_config):
    from mana.core import transaction
    proposal = propose_capability(discovery(), CognitiveGenome())
    confirmed(proposal)
    assert transaction.unfinished() == []
    assert "synthesis" in {t.kind for t in transaction.read_journal()}


# ---------------------------------------------------------------------------
# no duplicate capabilities
# ---------------------------------------------------------------------------

def test_a_chain_the_genome_already_has_is_not_installed_twice():
    """Two records for one mechanism split its evidence and make both
    look weaker than the thing they measure."""
    base = CognitiveGenome()
    existing = base.program_templates["critique_loop"].steps
    assert propose_capability(discovery(steps=existing), base) is None


def test_a_one_step_chain_is_not_a_capability():
    assert propose_capability(discovery(steps=("ANSWER",)), CognitiveGenome()) is None


def test_the_name_says_what_it_does_and_where_it_was_proven():
    proposal = propose_capability(discovery(), CognitiveGenome())
    assert "arithmetic_hard" in proposal.name
    assert "verify" in proposal.name


# ---------------------------------------------------------------------------
# a capability that stops working is withdrawn
# ---------------------------------------------------------------------------

def adopted_proposal(isolated=None):
    base = CognitiveGenome()
    proposal = propose_capability(discovery(), base)
    adopt(proposal, confirmed(proposal))
    return proposal


def model_using(name, correct, n=12):
    model = SelfModel()
    for i in range(n):
        model.record(Observation(f"t{i}", "arithmetic", 0.8, correct,
                                 program=name, calls=1))
    return model


def test_a_capability_that_underperforms_its_baseline_is_retired(isolated_config):
    proposal = adopted_proposal()
    retire_it, reason = should_retire(proposal, model_using(proposal.name, False))
    assert retire_it
    assert "ниже базовой" in reason


def test_a_capability_that_works_is_kept(isolated_config):
    proposal = adopted_proposal()
    retire_it, reason = should_retire(proposal, model_using(proposal.name, True))
    assert not retire_it
    assert "работает" in reason


def test_a_capability_is_judged_against_the_baseline_it_beat(isolated_config):
    """Not an absolute bar: a capability installed where everything
    scores 0.3 is doing its job at 0.4, and an absolute threshold would
    retire it for the difficulty of the slice."""
    proposal = adopted_proposal()
    model = SelfModel()
    for i in range(12):
        model.record(Observation(f"t{i}", "arithmetic", 0.8, i < 6,
                                 program=proposal.name, calls=1))
    retire_it, _ = should_retire(proposal, model)
    assert not retire_it, "0.50 is above the 0.40 baseline it was adopted over"


def test_a_capability_with_too_few_uses_is_not_judged_yet(isolated_config):
    proposal = adopted_proposal()
    retire_it, reason = should_retire(proposal, model_using(proposal.name, False, n=2))
    assert not retire_it
    assert "слишком мало" in reason


def test_a_capability_never_applied_is_not_retired(isolated_config):
    proposal = adopted_proposal()
    retire_it, reason = should_retire(proposal, SelfModel())
    assert not retire_it
    assert "не применялась" in reason


def test_success_is_read_from_the_record_not_from_the_capability(isolated_config):
    """A capability trusted on its own account cannot be withdrawn."""
    proposal = adopted_proposal()
    model = SelfModel()
    for i in range(6):
        model.record(Observation(f"other{i}", "arithmetic", 0.8, True,
                                 program="something_else", calls=1))
    assert synthesis.measured_success(model, proposal.name) is None


def test_retiring_removes_it_from_the_genome(isolated_config):
    base = CognitiveGenome()
    proposal = propose_capability(discovery(), base)
    installed = adopt(proposal, confirmed(proposal))
    after = synthesis.retire(proposal, installed, "перестала работать")
    assert proposal.name not in after.program_templates
    assert proposal.status == RETIRED
    assert after.parent_id == installed.genome_id


def test_retiring_needs_no_gate(isolated_config):
    """Requiring the same evidence to undo a change would leave a
    capability installed precisely when the evidence for it evaporated."""
    import inspect
    source = inspect.getsource(synthesis.retire)
    assert "judge" not in source and "Verdict" not in source


# ---------------------------------------------------------------------------
# the synthesizer
# ---------------------------------------------------------------------------

def test_the_synthesizer_installs_and_reports(isolated_config):
    s = CapabilitySynthesizer(CognitiveGenome())
    proposal = s.consider(discovery())
    assert proposal is not None
    assert s.install(proposal, outcomes(), hidden=(0.40, 0.72),
                     counterexamples=(3, 0), cost=CostVector(calls=80))
    report = s.report()
    assert proposal.name in report["templates"]
    assert report["adopted"] == [proposal.name]


def test_the_synthesizer_keeps_a_rejected_proposal_out_of_the_genome(isolated_config):
    s = CapabilitySynthesizer(CognitiveGenome())
    proposal = s.consider(discovery())
    assert not s.install(proposal, outcomes(n=40, baseline_ok=0.6, candidate_ok=0.6))
    assert proposal.name not in s.report()["templates"]
    assert s.adopted() == []


def test_a_review_withdraws_what_stopped_working(isolated_config):
    s = CapabilitySynthesizer(CognitiveGenome())
    proposal = s.consider(discovery())
    s.install(proposal, outcomes(), hidden=(0.40, 0.72),
              counterexamples=(3, 0), cost=CostVector(calls=80))
    assert proposal.name in s.genome.program_templates
    s.review(model_using(proposal.name, False))
    assert proposal.name not in s.genome.program_templates
    assert proposal.status == RETIRED
