"""
mana.core.gates — the only place that may say "accepted".

Everything MANA might change about itself -- a genome, a cognitive
program, a new operator, a patch to its own source -- arrives here as a
claim with evidence, and leaves with a verdict. Nothing outside this
module is allowed to conclude that something improved, which is the whole
content of "MANA must not be able to improve itself by editing the thing
that declares it improved".

What changed versus the gates this replaces
-------------------------------------------
`evolution._strict_acceptance` had the right instincts -- a tie is a
rejection, a regression is a rejection -- and one statistical flaw that
mattered: it computed a z-score over the mean of 21 substring checks.
Twenty-one paired binary outcomes do not support a normal approximation on
means, and the gates were reading noise as signal at the resolution they
were being asked to work at.

Here the comparison is **paired and binary**, because that is what the
data actually is: the same task, answered by baseline and by candidate,
each either right or wrong. McNemar's test is the standard instrument for
exactly that shape, needs only the two discordant counts, and does not
pretend the outcomes are continuous. It also needs no scipy, which keeps
the immutable core free of a heavy dependency.

The gates
---------
  1. sample_size      enough paired trials to say anything at all
  2. dev_improvement   candidate beats baseline on visible tasks
  3. significance      the difference is not the coin landing that way
  4. no_regression     no domain got materially worse
  5. hidden_confirms   the improvement survives the set nobody can read
  6. transfer          holds where the claim says it holds
  7. counterexamples   nothing found breaking it

A claim states which gates apply to it. A claim that asserts transfer and
skips gate 6 is rejected as malformed rather than accepted cheaply --
selecting your own gates is the loophole this design exists to close.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cost import CostVector

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.4"

#: Minimum paired observations before any verdict is meaningful. Chosen
#: against what the tests can actually deliver (100+ per split), not
#: against what is convenient.
MIN_PAIRED_TRIALS = 30

#: Two-sided significance level. Not tunable by a caller: a gate whose
#: threshold the claimant chooses is not a gate.
ALPHA = 0.05

#: How much worse a single domain may get while the whole still improves.
#: Non-zero because demanding zero regression anywhere makes every real
#: trade-off unacceptable; small because a "gain" bought by breaking a
#: domain is what this number exists to catch.
DOMAIN_REGRESSION_TOLERANCE = 0.05

#: Absolute accuracy margin required on top of statistical significance.
#: Significance says "not chance"; this says "worth the cost".
MIN_ABSOLUTE_MARGIN = 0.02


@dataclass(frozen=True)
class PairedOutcome:
    """Baseline and candidate, graded on the same task.

    Paired is not a nicety here. Comparing two independent samples of
    generated tasks throws away the biggest source of variance -- task
    difficulty -- and needs far more trials to see the same effect.
    """
    task_id: str
    domain: str
    baseline_correct: bool
    candidate_correct: bool


@dataclass(frozen=True)
class Claim:
    """What is being asserted, and which gates therefore apply."""
    claim_id: str
    kind: str                      # program | operator | representation | genome | code
    description: str
    asserts_transfer: bool = False
    asserts_domains: Tuple[str, ...] = ()

    def required_gates(self) -> Tuple[str, ...]:
        gates = ["sample_size", "dev_improvement", "significance",
                 "no_regression", "hidden_confirms", "counterexamples"]
        if self.asserts_transfer:
            gates.append("transfer")
        return tuple(gates)


@dataclass
class Evidence:
    """Everything measured about the claim. Assembled by the caller,
    interpreted only here."""
    paired_dev: List[PairedOutcome] = field(default_factory=list)
    baseline_hidden: Optional[float] = None
    candidate_hidden: Optional[float] = None
    baseline_transfer: Optional[float] = None
    candidate_transfer: Optional[float] = None
    counterexamples_sought: int = 0
    counterexamples_found: int = 0
    #: Hidden-set accuracy per domain, for both arms. The gate scopes the
    #: confirmation to the domains the CLAIM asserts -- and it derives that
    #: subset from the claim, never from the caller, because a caller
    #: choosing which slice confirms it is choosing the benchmark it wins
    #: on. Use `with_hidden` to fill these from two HiddenResults rather
    #: than by hand.
    baseline_hidden_by_domain: Dict[str, float] = field(default_factory=dict)
    candidate_hidden_by_domain: Dict[str, float] = field(default_factory=dict)
    #: What the evidence cost to produce. Recorded, never judged: no gate
    #: reads this field, and a gate that started to would make acceptance
    #: depend on how expensive the proof was, which is not a property of
    #: whether the claim is true. It replaced a plain `cost_calls: int`
    #: because a call to a 120B remote model and a call to a local 7B were
    #: the same number, which made "cheaper" unmeasurable.
    cost: CostVector = field(default_factory=CostVector)
    notes: Dict[str, Any] = field(default_factory=dict)

    def with_hidden(self, baseline: Any, candidate: Any) -> "Evidence":
        """Fill every hidden field from two evaluation results at once.

        Takes the STRICT accuracies, where an ungradable answer counts as
        not correct. Passing `.accuracy` by hand is how a refusing brain
        ends up compared against a guessing model over different
        denominators, and this exists so that is not the easy path.

        Duck-typed rather than importing HiddenResult: the core's gate
        module must not depend on the splits module, or the acceptance
        rules would import the thing they are meant to judge.
        """
        self.baseline_hidden = float(getattr(baseline, "strict_accuracy", 0.0))
        self.candidate_hidden = float(getattr(candidate, "strict_accuracy", 0.0))
        self.baseline_hidden_by_domain = dict(
            getattr(baseline, "strict_by_domain", {}) or {})
        self.candidate_hidden_by_domain = dict(
            getattr(candidate, "strict_by_domain", {}) or {})
        return self


#: A claim can end in three states, not two. "We tested it and it did
#: not hold" and "we could not test it here" are different facts, and
#: collapsing the second into the first writes a refutation into the
#: record for a claim nothing measured -- which then stops it being
#: retried on evidence that never existed.
ACCEPTED = "ACCEPTED"
REJECTED = "REJECTED"
NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: str
    failed_gates: Tuple[str, ...] = ()
    measurements: Dict[str, Any] = field(default_factory=dict)
    #: Which of the three. `accepted` stays a bool and stays false for
    #: NOT_EVALUATED, because a caller that only asks "may I adopt this?"
    #: must keep getting the safe answer without knowing about states.
    status: str = ""

    def __post_init__(self) -> None:
        if not self.status:
            object.__setattr__(self, "status",
                               ACCEPTED if self.accepted else REJECTED)

    @property
    def evaluated(self) -> bool:
        return self.status != NOT_EVALUATED

    def as_dict(self) -> Dict[str, Any]:
        return {"accepted": self.accepted, "status": self.status,
                "reason": self.reason,
                "failed_gates": list(self.failed_gates),
                "measurements": dict(self.measurements)}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def mcnemar(outcomes: Sequence[PairedOutcome]) -> Dict[str, Any]:
    """Paired binary comparison.

    Only the *discordant* pairs carry information: tasks both variants got
    right, or both got wrong, say nothing about which is better. `b` is
    baseline-right/candidate-wrong, `c` is the reverse, and the question is
    whether c exceeds b by more than a fair coin would produce.

    Continuity-corrected chi-square with one degree of freedom; the
    p-value comes from erfc rather than a distribution table, so this
    stays dependency-free inside the immutable core.
    """
    b = sum(1 for o in outcomes if o.baseline_correct and not o.candidate_correct)
    c = sum(1 for o in outcomes if not o.baseline_correct and o.candidate_correct)
    discordant = b + c
    if discordant == 0:
        return {"b": b, "c": c, "discordant": 0, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / discordant if discordant > 0 else 0.0
    statistic = max(0.0, statistic)
    p_value = math.erfc(math.sqrt(statistic / 2.0)) if statistic > 0 else 1.0
    return {"b": b, "c": c, "discordant": discordant,
            "statistic": round(statistic, 4), "p_value": round(p_value, 6)}


def accuracy(outcomes: Sequence[PairedOutcome], which: str) -> float:
    if not outcomes:
        return 0.0
    field_name = "baseline_correct" if which == "baseline" else "candidate_correct"
    return sum(1 for o in outcomes if getattr(o, field_name)) / len(outcomes)


def by_domain(outcomes: Sequence[PairedOutcome]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[PairedOutcome]] = {}
    for o in outcomes:
        grouped.setdefault(o.domain, []).append(o)
    return {d: {"baseline": accuracy(v, "baseline"),
                "candidate": accuracy(v, "candidate"),
                "n": len(v)}
            for d, v in grouped.items()}


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

#: How far a domain the claim does NOT assert may fall on the hidden set
#: before the claim is refused. Non-zero because the holdout is small and
#: a single task flipping in a three-task domain is noise, not a
#: regression.
HIDDEN_COLLAPSE_TOLERANCE = 0.15


def _judge_hidden(claim: Claim, evidence: Evidence) -> Tuple[bool, Dict[str, Any]]:
    """Does the holdout confirm THIS claim?

    Until now it compared one number against another, averaged over every
    development domain, however narrow the claim. A claim asserting
    arithmetic was confirmed by an average dominated by domains it never
    mentioned -- and a live run showed exactly that: two arms tied at 0.50
    with opposite profiles (0.33/0.67 against 1.00/0.00), and the gate saw
    "not worse" and passed.

    Two checks now, and both are tighter than what they replace:

    **Where it claims to help, it must help.** Scoped to
    `claim.asserts_domains` when the per-domain evidence is there. A claim
    that asserts a domain the holdout has no measurement for fails, rather
    than falling back to an average that says nothing about it.

    **Where it claims nothing, it must not collapse.** Every other domain
    is checked for a fall beyond tolerance, so buying a narrow win by
    wrecking something else is refused here as well as on the dev set.

    The subset comes from the claim, never from the caller. A caller
    choosing which domains confirm it is choosing the benchmark it wins
    on, which is the whole failure this layer exists to prevent.
    """
    notes: Dict[str, Any] = {
        "hidden_baseline": round(evidence.baseline_hidden, 4),
        "hidden_candidate": round(evidence.candidate_hidden, 4),
        "hidden_margin": round(evidence.candidate_hidden - evidence.baseline_hidden, 4),
    }
    base_by = evidence.baseline_hidden_by_domain or {}
    cand_by = evidence.candidate_hidden_by_domain or {}

    if not claim.asserts_domains or not base_by or not cand_by:
        # No per-domain evidence, or a claim that names no domain: the
        # overall comparison is all there is. Weaker than the dev margin
        # on purpose -- the holdout is small and consulted rarely, so
        # demanding the same effect size would reject real improvements
        # for lack of resolution. It must simply not go backwards.
        notes["hidden_scope"] = "overall"
        return (evidence.candidate_hidden - evidence.baseline_hidden) < 0, notes

    asserted = tuple(claim.asserts_domains)
    notes["hidden_scope"] = list(asserted)
    missing = [d for d in asserted if d not in cand_by or d not in base_by]
    if missing:
        # Not a refusal. The holdout in use has no measurement for this
        # domain, which says nothing about the claim.
        notes["hidden_unmeasured_domains"] = missing
        notes["hidden_state"] = NOT_EVALUATED
        return True, notes

    scoped_base = sum(base_by[d] for d in asserted) / len(asserted)
    scoped_cand = sum(cand_by[d] for d in asserted) / len(asserted)
    notes["hidden_scoped_baseline"] = round(scoped_base, 4)
    notes["hidden_scoped_candidate"] = round(scoped_cand, 4)
    notes["hidden_scoped_margin"] = round(scoped_cand - scoped_base, 4)

    collapsed = sorted(
        d for d in base_by
        if d not in asserted
        and cand_by.get(d, 0.0) < base_by[d] - HIDDEN_COLLAPSE_TOLERANCE)
    if collapsed:
        notes["hidden_collapsed_domains"] = collapsed

    return (scoped_cand < scoped_base) or bool(collapsed), notes


def min_discordant_pairs() -> int:
    """The fewest one-directional discordant pairs THIS gate calls
    significant. Computed by asking the gate rather than deriving it, so
    a change to the correction cannot leave a second opinion behind."""
    for count in range(2, 60):
        outcomes = ([PairedOutcome(f"d{i}", "x", False, True) for i in range(count)] +
                    [PairedOutcome(f"s{i}", "x", True, True) for i in range(count)])
        if mcnemar(outcomes)["p_value"] < ALPHA:
            return count
    return 60                                          # pragma: no cover


def required_trials(min_effect: float) -> int:
    """How many paired trials are needed to see an effect this small.

    The significance bar does not move. If a 0.13 effect needs 60 pairs,
    the answer is 60 pairs -- which is what a live sequence experiment
    actually needed, after 30 refused it for want of resolution.
    """
    if min_effect <= 0:
        raise ValueError("min_effect must be positive")
    needed = int(math.ceil(min_discordant_pairs() / float(min_effect)))
    return max(MIN_PAIRED_TRIALS, needed)


def judge(claim: Claim, evidence: Evidence) -> Verdict:
    """Apply every gate the claim requires. Returns, never raises.

    Order matters only for readability of the failure list; all gates are
    evaluated, so a rejected claim reports everything wrong with it rather
    than the first thing.
    """
    failed: List[str] = []
    m: Dict[str, Any] = {}

    outcomes = evidence.paired_dev
    n = len(outcomes)
    m["paired_trials"] = n
    if n < MIN_PAIRED_TRIALS:
        failed.append("sample_size")

    base_acc = accuracy(outcomes, "baseline")
    cand_acc = accuracy(outcomes, "candidate")
    margin = cand_acc - base_acc
    m.update({"dev_baseline": round(base_acc, 4), "dev_candidate": round(cand_acc, 4),
              "dev_margin": round(margin, 4)})
    # A tie is a rejection, preserved deliberately from _strict_acceptance:
    # "no measurable win" is not a reason to change anything.
    if margin < MIN_ABSOLUTE_MARGIN:
        failed.append("dev_improvement")

    stats = mcnemar(outcomes)
    m["mcnemar"] = stats
    if not (stats["p_value"] < ALPHA and stats["c"] > stats["b"]):
        failed.append("significance")

    domains = by_domain(outcomes)
    m["by_domain"] = {d: {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in vals.items()} for d, vals in domains.items()}
    regressed = [d for d, v in domains.items()
                 if v["candidate"] < v["baseline"] - DOMAIN_REGRESSION_TOLERANCE]
    if regressed:
        failed.append("no_regression")
        m["regressed_domains"] = sorted(regressed)

    if evidence.baseline_hidden is None or evidence.candidate_hidden is None:
        failed.append("hidden_confirms")
        m["hidden"] = "not measured"
    else:
        hidden_failed, hidden_notes = _judge_hidden(claim, evidence)
        m.update(hidden_notes)
        if hidden_failed:
            failed.append("hidden_confirms")

    if claim.asserts_transfer:
        if evidence.baseline_transfer is None or evidence.candidate_transfer is None:
            failed.append("transfer")
            m["transfer"] = "claimed but not measured"
        else:
            transfer_margin = evidence.candidate_transfer - evidence.baseline_transfer
            m["transfer_baseline"] = round(evidence.baseline_transfer, 4)
            m["transfer_candidate"] = round(evidence.candidate_transfer, 4)
            m["transfer_margin"] = round(transfer_margin, 4)
            if transfer_margin < MIN_ABSOLUTE_MARGIN:
                failed.append("transfer")

    # Looking for counterexamples and finding none is evidence. Not
    # looking is not: a claim nobody tried to break has not survived
    # anything.
    m["counterexamples_sought"] = evidence.counterexamples_sought
    m["counterexamples_found"] = evidence.counterexamples_found
    if evidence.counterexamples_sought <= 0:
        failed.append("counterexamples")
        m["counterexample_note"] = "none sought; survival cannot be claimed"
    elif evidence.counterexamples_found > 0:
        failed.append("counterexamples")

    required = set(claim.required_gates())
    failed_required = tuple(g for g in claim.required_gates() if g in set(failed))
    accepted = not failed_required
    reason = "accepted" if accepted else "failed: " + ", ".join(failed_required)
    m["required_gates"] = sorted(required)
    m["cost"] = evidence.cost.as_dict()
    # An asserted domain the holdout never measured leaves the claim
    # UNEVALUATED rather than refuted, however the other gates came out.
    # "We tested it and it did not hold" and "we could not test it here"
    # are different facts, and collapsing the second into the first writes
    # a refutation into the record for a claim nothing measured -- which
    # then stops it being retried on evidence that never existed.
    unevaluated = m.get("hidden_state") == NOT_EVALUATED
    if unevaluated:
        reason = ("не оценено: скрытая выборка не покрывает "
                  + ", ".join(m.get("hidden_unmeasured_domains", [])))
    status = (NOT_EVALUATED if unevaluated
              else (ACCEPTED if accepted else REJECTED))
    return Verdict(accepted=accepted and not unevaluated, reason=reason,
                   failed_gates=failed_required, measurements=m, status=status)
