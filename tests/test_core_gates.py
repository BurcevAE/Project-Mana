"""
tests/test_core_gates.py — the acceptance gate and the transaction journal.

The gate is the single point where MANA is allowed to conclude that
something improved, so most of these tests are about what it must
*refuse*. A gate that accepts readily is not a gate.
"""
from __future__ import annotations

import json
import random

import pytest

from mana.core import evaluation, gates, transaction
from mana.core.cost import CostVector
from mana.core.gates import Claim, Evidence, PairedOutcome


def paired(n, base_rate, cand_rate, domain="arithmetic", seed=1):
    """Paired outcomes with the requested marginal accuracies.

    Built so the discordant pairs are what the rates imply, since those
    are the only pairs McNemar reads.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        b = rng.random() < base_rate
        c = rng.random() < cand_rate
        out.append(PairedOutcome(f"t{i}", domain, b, c))
    return out


def full_evidence(outcomes, **overrides):
    """Evidence that passes everything except what a test overrides."""
    data = dict(paired_dev=outcomes, baseline_hidden=0.60, candidate_hidden=0.72,
                counterexamples_sought=25, counterexamples_found=0, cost=CostVector(calls=900))
    data.update(overrides)
    return Evidence(**data)


CLAIM = Claim("c1", "operator", "simulation before execution")
TRANSFER_CLAIM = Claim("c2", "operator", "same, claimed to transfer", asserts_transfer=True)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_mcnemar_reads_only_the_discordant_pairs():
    """Tasks both variants got right say nothing about which is better.
    Counting them would let a large easy set drown a real difference."""
    both_right = [PairedOutcome(f"t{i}", "d", True, True) for i in range(100)]
    discordant = [PairedOutcome("x", "d", False, True), PairedOutcome("y", "d", False, True)]
    assert gates.mcnemar(both_right)["discordant"] == 0
    assert gates.mcnemar(both_right + discordant)["discordant"] == 2


def test_identical_variants_are_never_significant():
    same = [PairedOutcome(f"t{i}", "d", i % 3 == 0, i % 3 == 0) for i in range(60)]
    assert gates.mcnemar(same)["p_value"] == 1.0


def test_a_clear_one_sided_difference_is_significant():
    outcomes = ([PairedOutcome(f"w{i}", "d", False, True) for i in range(20)] +
                [PairedOutcome(f"l{i}", "d", True, False) for i in range(2)] +
                [PairedOutcome(f"s{i}", "d", True, True) for i in range(40)])
    stats = gates.mcnemar(outcomes)
    assert stats["p_value"] < 0.001
    assert stats["c"] > stats["b"]


# ---------------------------------------------------------------------------
# what the gate refuses
# ---------------------------------------------------------------------------

def test_too_few_trials_is_refused_however_good_it_looks():
    """The gate this replaces computed a z-score over 21 observations."""
    outcomes = [PairedOutcome(f"t{i}", "d", False, True) for i in range(10)]
    verdict = gates.judge(CLAIM, full_evidence(outcomes))
    assert verdict.accepted is False
    assert "sample_size" in verdict.failed_gates


def test_a_tie_is_a_rejection():
    """Preserved deliberately from _strict_acceptance: 'no measurable win'
    is not a reason to change anything."""
    outcomes = [PairedOutcome(f"t{i}", "d", i % 2 == 0, i % 2 == 0) for i in range(60)]
    verdict = gates.judge(CLAIM, full_evidence(outcomes))
    assert "dev_improvement" in verdict.failed_gates


def test_a_lucky_streak_is_refused_by_significance():
    """Three wins and no losses out of sixty is an improvement of 5% that
    a coin produces often enough to matter."""
    outcomes = ([PairedOutcome(f"w{i}", "d", False, True) for i in range(3)] +
                [PairedOutcome(f"s{i}", "d", True, True) for i in range(57)])
    verdict = gates.judge(CLAIM, full_evidence(outcomes))
    assert "significance" in verdict.failed_gates


def test_a_gain_bought_by_wrecking_one_domain_is_refused():
    good = ([PairedOutcome(f"a{i}", "arithmetic", False, True) for i in range(25)] +
            [PairedOutcome(f"b{i}", "arithmetic", True, True) for i in range(15)])
    wrecked = [PairedOutcome(f"c{i}", "sequence", True, False) for i in range(20)]
    verdict = gates.judge(CLAIM, full_evidence(good + wrecked))
    assert "no_regression" in verdict.failed_gates
    assert "sequence" in verdict.measurements["regressed_domains"]


def test_an_improvement_that_the_hidden_set_contradicts_is_refused():
    outcomes = ([PairedOutcome(f"w{i}", "d", False, True) for i in range(25)] +
                [PairedOutcome(f"s{i}", "d", True, True) for i in range(35)])
    verdict = gates.judge(CLAIM, full_evidence(outcomes, baseline_hidden=0.70,
                                               candidate_hidden=0.61))
    assert "hidden_confirms" in verdict.failed_gates


def test_hidden_that_was_never_measured_counts_as_a_failure_not_an_absence():
    outcomes = ([PairedOutcome(f"w{i}", "d", False, True) for i in range(25)] +
                [PairedOutcome(f"s{i}", "d", True, True) for i in range(35)])
    verdict = gates.judge(CLAIM, full_evidence(outcomes, baseline_hidden=None,
                                               candidate_hidden=None))
    assert "hidden_confirms" in verdict.failed_gates


def test_claiming_transfer_without_measuring_it_is_refused():
    """Selecting your own gates is the loophole this design closes."""
    outcomes = ([PairedOutcome(f"w{i}", "d", False, True) for i in range(25)] +
                [PairedOutcome(f"s{i}", "d", True, True) for i in range(35)])
    verdict = gates.judge(TRANSFER_CLAIM, full_evidence(outcomes))
    assert "transfer" in verdict.failed_gates
    assert verdict.measurements["transfer"] == "claimed but not measured"


def test_a_claim_nobody_tried_to_break_has_survived_nothing():
    outcomes = ([PairedOutcome(f"w{i}", "d", False, True) for i in range(25)] +
                [PairedOutcome(f"s{i}", "d", True, True) for i in range(35)])
    verdict = gates.judge(CLAIM, full_evidence(outcomes, counterexamples_sought=0))
    assert "counterexamples" in verdict.failed_gates
    assert "none sought" in verdict.measurements["counterexample_note"]


def test_a_found_counterexample_is_fatal():
    outcomes = ([PairedOutcome(f"w{i}", "d", False, True) for i in range(25)] +
                [PairedOutcome(f"s{i}", "d", True, True) for i in range(35)])
    verdict = gates.judge(CLAIM, full_evidence(outcomes, counterexamples_found=1))
    assert "counterexamples" in verdict.failed_gates


def test_a_rejected_claim_reports_everything_wrong_with_it():
    """One failure at a time turns a review into a guessing game."""
    outcomes = [PairedOutcome(f"t{i}", "d", True, False) for i in range(40)]
    verdict = gates.judge(TRANSFER_CLAIM, Evidence(paired_dev=outcomes))
    assert {"dev_improvement", "significance", "hidden_confirms",
            "transfer", "counterexamples"} <= set(verdict.failed_gates)


# ---------------------------------------------------------------------------
# what the gate accepts
# ---------------------------------------------------------------------------

def test_a_real_improvement_with_complete_evidence_is_accepted():
    outcomes = ([PairedOutcome(f"w{i}", "arithmetic", False, True) for i in range(22)] +
                [PairedOutcome(f"l{i}", "arithmetic", True, False) for i in range(3)] +
                [PairedOutcome(f"s{i}", "sequence", True, True) for i in range(40)])
    verdict = gates.judge(CLAIM, full_evidence(outcomes))
    assert verdict.accepted is True, verdict.reason
    assert verdict.failed_gates == ()
    assert verdict.measurements["dev_margin"] > 0


def test_transfer_is_accepted_only_when_it_actually_holds():
    outcomes = ([PairedOutcome(f"w{i}", "arithmetic", False, True) for i in range(22)] +
                [PairedOutcome(f"l{i}", "arithmetic", True, False) for i in range(3)] +
                [PairedOutcome(f"s{i}", "sequence", True, True) for i in range(40)])
    weak = gates.judge(TRANSFER_CLAIM, full_evidence(
        outcomes, baseline_transfer=0.55, candidate_transfer=0.555))
    strong = gates.judge(TRANSFER_CLAIM, full_evidence(
        outcomes, baseline_transfer=0.55, candidate_transfer=0.68))
    assert "transfer" in weak.failed_gates
    assert strong.accepted is True, strong.reason


# ---------------------------------------------------------------------------
# evaluation mode
# ---------------------------------------------------------------------------

def test_held_out_runs_do_not_learn():
    with evaluation.open_evaluation(evaluation.HELD_OUT, "hidden") as ctx:
        assert ctx.learning_enabled is False
        assert ctx.is_measured is True
        assert ctx.authorized is True


def test_dev_runs_are_measured_but_still_learn():
    with evaluation.open_evaluation(evaluation.DEV, "dev") as ctx:
        assert ctx.learning_enabled is True
        assert ctx.is_measured is True


def test_a_mode_nobody_issued_is_marked_unauthorized():
    """Not an exception: a forged flag that crashes tells you less than a
    record showing one was used."""
    forged = evaluation.EvaluationMode(mode=evaluation.HELD_OUT, token="made-up")
    assert forged.authorized is False
    assert forged.learning_enabled is False


def test_the_mode_cannot_outlive_the_run_that_needed_it():
    """The failure the old boolean invited: leaving it set and silently
    disabling learning for the rest of the session."""
    with evaluation.open_evaluation(evaluation.HELD_OUT) as ctx:
        token = ctx.token
        assert ctx.authorized is True
    assert token not in evaluation._issued
    assert ctx.authorized is False


def test_the_default_state_of_the_world_needs_no_proof():
    assert evaluation.normal().learning_enabled is True
    assert evaluation.normal().is_measured is False


# ---------------------------------------------------------------------------
# transactions
# ---------------------------------------------------------------------------

@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_DATA_DIR", str(tmp_path))
    return tmp_path / "experiments" / "transactions.jsonl"


def test_a_completed_transaction_is_not_reported_as_unfinished(journal):
    with transaction.TransactionScope("c1", "operator", "test") as txn:
        txn.step(transaction.SNAPSHOT, path="x")
        txn.step(transaction.MEASURED, accuracy=0.7)
        txn.commit(verdict="accepted")
    assert transaction.unfinished() == []


def test_an_interrupted_transaction_is_findable_afterwards(journal):
    """The crash-recovery contract: after a kill, the system must know a
    self-modification was left half-done."""
    txn_id = transaction.open_transaction("c2", "code", "patch _local_fallback")
    transaction.record(txn_id, transaction.SNAPSHOT, backup="x.bak")
    transaction.record(txn_id, transaction.DECIDED, accepted=True)
    # process dies here -- nothing writes COMMITTED
    pending = transaction.unfinished()
    assert [t.txn_id for t in pending] == [txn_id]
    assert pending[0].last_state == transaction.DECIDED
    assert pending[0].data["backup"] == "x.bak"


def test_recovery_does_not_act_on_its_own(journal):
    """A transaction that died after writing a patch but before logging
    COMMITTED must not be rolled back blindly -- only the owner knows."""
    transaction.open_transaction("c3", "code", "x")
    assert hasattr(transaction, "unfinished")
    assert not hasattr(transaction, "auto_rollback")


def test_leaving_a_scope_without_a_decision_is_recorded_as_failed(journal):
    with pytest.raises(RuntimeError):
        with transaction.TransactionScope("c4", "genome", "x") as txn:
            txn.step(transaction.SNAPSHOT)
            raise RuntimeError("boom")
    entries = transaction.read_journal()
    assert entries[0].last_state == transaction.FAILED
    assert "boom" in entries[0].data["error"]


def test_a_truncated_final_line_does_not_make_the_journal_unreadable(journal):
    """A process killed mid-write leaves exactly that, and refusing to
    read would break recovery in the case it exists for."""
    txn_id = transaction.open_transaction("c5", "operator", "x")
    with open(journal, "a", encoding="utf-8") as fh:
        fh.write('{"txn_id": "broken", "sta')
    recovered = transaction.read_journal()
    assert [t.txn_id for t in recovered] == [txn_id]


def test_closing_with_a_non_terminal_state_is_rejected(journal):
    txn_id = transaction.open_transaction("c6", "operator", "x")
    with pytest.raises(ValueError, match="terminal"):
        transaction.close(txn_id, transaction.MEASURED)


def test_the_journal_lives_outside_the_package_it_describes(journal, monkeypatch):
    """Evidence about the code stored inside the code disappears with it."""
    from mana.core import CORE_ROOT
    path = transaction.journal_path()
    assert CORE_ROOT not in path.parents


# ---------------------------------------------------------------------------
# the flag no longer lives on the agent (audit conflict Б)
# ---------------------------------------------------------------------------

def test_the_agent_cannot_declare_itself_under_evaluation(isolated_agent):
    """It was `self._benchmark_holdout = True` -- the measured thing owned
    the flag saying whether it was being measured."""
    with pytest.raises(AttributeError):
        isolated_agent._benchmark_holdout = True


def test_a_forged_evaluation_context_is_rejected_at_the_point_of_use(isolated_agent):
    forged = evaluation.EvaluationMode(mode=evaluation.HELD_OUT, token="made-up")
    with pytest.raises(PermissionError, match="open_evaluation"):
        isolated_agent.enter_evaluation(forged)


def test_a_core_issued_context_switches_learning_off(isolated_agent):
    assert isolated_agent._benchmark_holdout is False
    with evaluation.open_evaluation(evaluation.HELD_OUT, "hidden") as ctx:
        isolated_agent.enter_evaluation(ctx)
        assert isolated_agent._benchmark_holdout is True
        assert isolated_agent.evaluation_mode.learning_enabled is False
    isolated_agent.leave_evaluation()
    assert isolated_agent._benchmark_holdout is False


def test_normal_work_needs_no_authorization(isolated_agent):
    isolated_agent.enter_evaluation(evaluation.normal())
    assert isolated_agent._benchmark_holdout is False


# ---------------------------------------------------------------------------
# the holdout has to confirm THIS claim
# ---------------------------------------------------------------------------

def _strong(n=40, base=0.4, cand=0.8):
    return [PairedOutcome(task_id=f"t{i}", domain="arithmetic",
                          baseline_correct=(i / n) < base,
                          candidate_correct=(i / n) < cand) for i in range(n)]


def test_a_narrow_claim_is_not_confirmed_by_an_average_over_other_domains():
    """The live case: two arms tied at 0.50 with opposite profiles --
    model arithmetic 0.33 / sequence 0.67, brain 1.00 / 0.00. The gate saw
    "not worse" and passed a claim about arithmetic on the strength of an
    average dominated by a domain it never mentioned."""
    claim = Claim(claim_id="c", kind="program", description="про арифметику",
                  asserts_domains=("arithmetic",))
    evidence = Evidence(
        paired_dev=_strong(),
        baseline_hidden=0.50, candidate_hidden=0.50,
        baseline_hidden_by_domain={"arithmetic": 0.33, "sequence": 0.67},
        candidate_hidden_by_domain={"arithmetic": 1.00, "sequence": 0.00},
        counterexamples_sought=4, counterexamples_found=0)
    verdict = gates.judge(claim, evidence)
    # Confirmed where it claims, refused for wrecking where it does not.
    assert verdict.measurements["hidden_scope"] == ["arithmetic"]
    assert verdict.measurements["hidden_scoped_margin"] > 0
    assert verdict.measurements["hidden_collapsed_domains"] == ["sequence"]
    assert "hidden_confirms" in verdict.failed_gates


def test_a_claim_confirmed_in_its_own_domain_passes():
    claim = Claim(claim_id="c", kind="program", description="про арифметику",
                  asserts_domains=("arithmetic",))
    evidence = Evidence(
        paired_dev=_strong(),
        baseline_hidden=0.50, candidate_hidden=0.60,
        baseline_hidden_by_domain={"arithmetic": 0.33, "sequence": 0.67},
        candidate_hidden_by_domain={"arithmetic": 1.00, "sequence": 0.60},
        counterexamples_sought=4, counterexamples_found=0)
    verdict = gates.judge(claim, evidence)
    assert "hidden_confirms" not in verdict.failed_gates


def test_a_claim_about_a_domain_the_holdout_never_measured_is_refused():
    """Falling back to an average that says nothing about the asserted
    domain is how the previous version passed claims it had not tested."""
    claim = Claim(claim_id="c", kind="program", description="про код",
                  asserts_domains=("code",))
    evidence = Evidence(
        paired_dev=_strong(),
        baseline_hidden=0.40, candidate_hidden=0.90,
        baseline_hidden_by_domain={"arithmetic": 0.40},
        candidate_hidden_by_domain={"arithmetic": 0.90},
        counterexamples_sought=4, counterexamples_found=0)
    verdict = gates.judge(claim, evidence)
    assert "hidden_confirms" in verdict.failed_gates
    assert verdict.measurements["hidden_unmeasured_domains"] == ["code"]


def test_the_subset_comes_from_the_claim_not_from_the_caller():
    """A caller choosing which domains confirm it is choosing the
    benchmark it wins on. There is no parameter for it."""
    import inspect
    from mana.core import gates
    signature = inspect.signature(gates.judge)
    assert list(signature.parameters) == ["claim", "evidence"]
    assert not hasattr(Evidence(), "hidden_domains_to_use")


def test_a_claim_naming_no_domain_still_uses_the_overall_comparison():
    claim = Claim(claim_id="c", kind="program", description="общее")
    evidence = Evidence(
        paired_dev=_strong(), baseline_hidden=0.40, candidate_hidden=0.55,
        counterexamples_sought=4, counterexamples_found=0)
    verdict = gates.judge(claim, evidence)
    assert verdict.measurements["hidden_scope"] == "overall"
    assert "hidden_confirms" not in verdict.failed_gates


def test_with_hidden_takes_the_strict_accuracy():
    """Passing `.accuracy` by hand is how a refusing brain ends up
    compared against a guessing model over different denominators."""
    class Result:
        accuracy = 0.90              # graded only -- flatters a refuser
        strict_accuracy = 0.30       # over everything attempted
        strict_by_domain = {"arithmetic": 1.0}

    evidence = Evidence().with_hidden(Result(), Result())
    assert evidence.baseline_hidden == 0.30
    assert evidence.baseline_hidden_by_domain == {"arithmetic": 1.0}


def test_an_unmeasured_domain_leaves_the_claim_unevaluated_not_refuted():
    """"We tested it and it did not hold" and "we could not test it here"
    are different facts. Collapsing the second into the first writes a
    refutation into the record for a claim nothing measured, and then
    stops it being retried on evidence that never existed."""
    claim = Claim(claim_id="c", kind="program", description="про логику",
                  asserts_domains=("logic",))
    evidence = Evidence(
        paired_dev=_strong(),
        baseline_hidden=0.40, candidate_hidden=0.90,
        baseline_hidden_by_domain={"arithmetic": 0.40},
        candidate_hidden_by_domain={"arithmetic": 0.90},
        counterexamples_sought=4, counterexamples_found=0)
    verdict = gates.judge(claim, evidence)
    assert verdict.status == gates.NOT_EVALUATED
    assert verdict.evaluated is False
    assert verdict.accepted is False, "nothing may be adopted on an untested claim"
    assert "не оценено" in verdict.reason


def test_an_ordinary_rejection_is_still_rejected():
    claim = Claim(claim_id="c", kind="program", description="общее")
    verdict = gates.judge(claim, Evidence(paired_dev=_strong(40, 0.6, 0.6)))
    assert verdict.status == gates.REJECTED
    assert verdict.evaluated is True


def test_a_caller_that_only_asks_accepted_still_gets_the_safe_answer():
    """`accepted` stays a bool and stays false for NOT_EVALUATED."""
    import inspect
    source = inspect.getsource(gates.Verdict)
    assert "accepted: bool" in source
