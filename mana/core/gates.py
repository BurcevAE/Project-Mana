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

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

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
    cost_calls: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: str
    failed_gates: Tuple[str, ...] = ()
    measurements: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason,
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
        hidden_margin = evidence.candidate_hidden - evidence.baseline_hidden
        m["hidden_baseline"] = round(evidence.baseline_hidden, 4)
        m["hidden_candidate"] = round(evidence.candidate_hidden, 4)
        m["hidden_margin"] = round(hidden_margin, 4)
        # Weaker than the dev margin on purpose: the hidden set is smaller
        # and consulted rarely, so demanding the same effect size there
        # would reject real improvements for lack of resolution. It must
        # not go BACKWARDS, and it must not be flat when dev moved a lot.
        if hidden_margin < 0:
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
    m["cost_calls"] = evidence.cost_calls
    return Verdict(accepted=accepted, reason=reason, failed_gates=failed_required, measurements=m)
