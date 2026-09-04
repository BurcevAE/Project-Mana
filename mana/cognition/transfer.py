"""
mana.cognition.transfer — does the mechanism work where it has never been?

The claim this exists to test
-----------------------------
A cognitive mechanism that works only in the domain it was found in has
not been shown to be a mechanism. It may be a fact about that domain's
task format, about which brain happens to suit it, or about a quirk of
the generator. Transfer is what separates "we found a mechanism" from "we
fitted a domain", and it is the load-bearing claim of the whole project.

Two strengths of evidence, deliberately kept apart
--------------------------------------------------
  * **Cross-domain** -- the mechanism holds in another *development*
    domain it was not found in. Real evidence, and cheap: those domains
    are visible, so this can be measured freely.
  * **Held-out** -- it holds in a domain that took no part in development
    at all (`core.splits.TRANSFER_DOMAINS`). Much stronger, and scarce:
    those tasks are never returned as data and every consultation is
    counted against a budget.

Collapsing them into one number would let cheap evidence stand in for
expensive evidence, which is exactly the substitution a system optimising
for its own scores would make.

Transfer is a ratio, not a yes/no
---------------------------------
"Did it also improve there?" throws away the interesting part. A
mechanism giving +0.20 at home and +0.18 elsewhere is a mechanism; the
same mechanism giving +0.20 and +0.02 is a domain-specific trick that
happens not to hurt. `retention` is the effect in the target divided by
the effect at the source, and it is what makes those two cases
distinguishable.

Not "similar tasks"
-------------------
The brief is explicit that transfer must not be checked by task
similarity, and the reason is that similarity is judged by the same
system whose generalisation is in question. Here the target distributions
come from `core.tasks` generators that share no structure with the source
-- `text_ops` and `arithmetic` have neither format nor solution method in
common -- so there is nothing to judge and nothing to get wrong.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import gates, splits
from ..core.gates import PairedOutcome

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

CROSS_DOMAIN = "cross_domain"     # another development domain
HELD_OUT = "held_out"             # a domain development never touched

#: Below this retention the effect is treated as not having travelled.
#: A fifth of the original effect is close enough to nothing that calling
#: it transfer would make the word useless.
MIN_RETENTION = 0.35

#: Absolute floor as well as a ratio. A mechanism whose source effect was
#: tiny can show high retention of almost nothing, and a ratio alone would
#: report that as excellent transfer.
MIN_ABSOLUTE_EFFECT = 0.03

#: Minimum paired trials per target domain. Lower than the acceptance
#: gate's 30 because transfer is measured across several domains at once
#: and the total is what matters; a single domain at 20 with three others
#: agreeing is stronger evidence than one domain at 60 alone.
MIN_TRIALS_PER_DOMAIN = 20


@dataclass
class DomainResult:
    """What happened in one target domain."""
    domain: str
    kind: str                       # CROSS_DOMAIN | HELD_OUT
    trials: int
    baseline: float
    candidate: float
    effect: float
    retention: float
    transferred: bool
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransferReport:
    source_domain: str
    source_effect: float
    results: List[DomainResult] = field(default_factory=list)

    @property
    def confirmed(self) -> List[str]:
        return [r.domain for r in self.results if r.transferred]

    @property
    def failed(self) -> List[str]:
        return [r.domain for r in self.results
                if not r.transferred and r.trials >= MIN_TRIALS_PER_DOMAIN]

    @property
    def held_out_confirmed(self) -> List[str]:
        return [r.domain for r in self.results if r.transferred and r.kind == HELD_OUT]

    def score(self) -> float:
        """One number for ranking, built from the parts that matter.

        Held-out confirmations count double. That is not a fudge: those
        domains took no part in development, so a confirmation there rules
        out a class of contamination that a cross-domain confirmation
        cannot. Weighting them equally would make the cheap evidence
        substitute for the expensive kind.
        """
        if not self.results:
            return 0.0
        weight = 0.0
        earned = 0.0
        for r in self.results:
            w = 2.0 if r.kind == HELD_OUT else 1.0
            weight += w
            if r.transferred:
                earned += w * min(1.0, max(0.0, r.retention))
        return earned / weight if weight else 0.0

    def verdict(self) -> str:
        if not self.results:
            return "не проверялся"
        if self.held_out_confirmed:
            return f"перенос подтверждён на скрытых доменах: {', '.join(self.held_out_confirmed)}"
        if self.confirmed:
            return f"перенос подтверждён на: {', '.join(self.confirmed)} (только development)"
        return "перенос не подтверждён ни в одном домене"

    def as_dict(self) -> Dict[str, Any]:
        return {"source_domain": self.source_domain,
                "source_effect": round(self.source_effect, 4),
                "score": round(self.score(), 4),
                "verdict": self.verdict(),
                "confirmed": self.confirmed,
                "failed": self.failed,
                "held_out_confirmed": self.held_out_confirmed,
                "results": [r.as_dict() for r in self.results]}


def evaluate_domain(domain: str, kind: str, outcomes: Sequence[PairedOutcome],
                    source_effect: float) -> DomainResult:
    """Decide whether the effect survived in one domain."""
    trials = len(outcomes)
    baseline = gates.accuracy(outcomes, "baseline")
    candidate = gates.accuracy(outcomes, "candidate")
    effect = candidate - baseline
    retention = effect / source_effect if source_effect > 0 else 0.0

    if trials < MIN_TRIALS_PER_DOMAIN:
        return DomainResult(domain, kind, trials, baseline, candidate, effect, retention,
                            transferred=False,
                            note=f"мало испытаний ({trials}/{MIN_TRIALS_PER_DOMAIN})")
    if effect < MIN_ABSOLUTE_EFFECT:
        return DomainResult(domain, kind, trials, baseline, candidate, effect, retention,
                            transferred=False,
                            note=f"эффект {effect:+.3f} ниже порога {MIN_ABSOLUTE_EFFECT}")
    if retention < MIN_RETENTION:
        return DomainResult(domain, kind, trials, baseline, candidate, effect, retention,
                            transferred=False,
                            note=(f"сохранилось {retention:.0%} исходного эффекта — "
                                  f"ниже {MIN_RETENTION:.0%}"))
    return DomainResult(domain, kind, trials, baseline, candidate, effect, retention,
                        transferred=True,
                        note=f"эффект {effect:+.3f}, сохранилось {retention:.0%}")


#: Runs baseline and candidate over N tasks of one domain, paired.
DomainRunner = Callable[[str, int], List[PairedOutcome]]


def measure(source_domain: str, source_effect: float, runner: DomainRunner,
            cross_domains: Sequence[str] = (),
            held_out_domains: Sequence[str] = (),
            trials: int = MIN_TRIALS_PER_DOMAIN) -> TransferReport:
    """Measure the same intervention across domains it was not found in.

    The source domain is excluded even if a caller passes it: measuring
    transfer onto the domain a mechanism came from is measuring the thing
    that produced it, and reporting that as transfer would be the single
    most effective way to make this whole module lie.
    """
    report = TransferReport(source_domain=source_domain, source_effect=source_effect)
    targets = [(d, CROSS_DOMAIN) for d in cross_domains if d != source_domain]
    targets += [(d, HELD_OUT) for d in held_out_domains if d != source_domain]

    for domain, kind in targets:
        try:
            outcomes = runner(domain, trials)
        except Exception as exc:
            report.results.append(DomainResult(
                domain, kind, 0, 0.0, 0.0, 0.0, 0.0, transferred=False,
                note=f"не выполнено: {type(exc).__name__}: {exc}"))
            continue
        report.results.append(evaluate_domain(domain, kind, outcomes, source_effect))
    return report


def default_targets(source_domain: str) -> Tuple[List[str], List[str]]:
    """Which domains to try, taken from the core's own split definition.

    Read from `core.splits` rather than restated here, so a change to what
    counts as held out cannot silently disagree with what transfer
    measures against.
    """
    cross = [d for d in splits.DEVELOPMENT_DOMAINS if d != source_domain]
    held_out = [d for d in splits.TRANSFER_DOMAINS if d != source_domain]
    return cross, held_out
