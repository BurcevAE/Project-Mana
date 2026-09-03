"""
mana.cognition.laws — conditional claims about MANA's own behaviour, and
the statuses they have to earn.

What a law is
-------------
Not "critique loops help". That cannot be refuted, so it is not a law --
it is an opinion with a citation. A law here names a condition, an
intervention and an effect:

    condition:     domain=arithmetic, difficulty >= 0.65
    intervention:  VERIFY before ANSWER
    effect:        +0.18 accuracy [0.06-0.29]
    evidence:      142 paired trials across 3 experiments
    transfer:      sequence: yes, logic: untested
    exceptions:    fails when the task has no computable oracle

Every field is something a later experiment can contradict, which is the
only property that makes the object worth having.

Four statuses, and the last one is the point
--------------------------------------------
    PROPOSED   one supported experiment. Suggestive, nothing more.
    SUPPORTED  holds on the hidden set as well as the visible one.
    VALIDATED  survived a deliberate search for counterexamples AND
               held in at least one domain it was not found in.
    REFUTED    later evidence contradicted it.

A system that can only promote its own laws accumulates folklore. The
demotion path is therefore symmetric and automatic: `record_evidence`
moves a law down as readily as up, and a VALIDATED law that fails on new
evidence in its own stated condition becomes REFUTED without anyone
deciding to look.

Status is derived, never set
----------------------------
There is no `set_status()`. `_derive_status` reads the accumulated
evidence and returns what it supports -- so a law cannot be promoted by a
component that would like it to be true, which is the same rule the
self-model follows for capabilities and for the same reason.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

PROPOSED = "PROPOSED"
SUPPORTED = "SUPPORTED"
VALIDATED = "VALIDATED"
REFUTED = "REFUTED"

#: Minimum paired trials before a law may be more than PROPOSED. Matches
#: the acceptance gate's threshold: a law is a claim about the same kind
#: of evidence, and two different bars would let one launder the other.
MIN_TRIALS_FOR_SUPPORT = 30

#: Domains outside the one a law was found in that must confirm it before
#: it can be VALIDATED. One is enough to distinguish a mechanism from a
#: local quirk; demanding more would make validation unreachable on the
#: budget this runs on.
MIN_TRANSFER_DOMAINS = 1


@dataclass(frozen=True)
class Condition:
    """When a law claims to apply.

    Deliberately narrow and machine-checkable: `matches()` decides, so a
    law cannot be defended after the fact by reinterpreting its scope --
    which is the commonest way a folk theory survives contact with data.
    """
    domain: str = ""                     # "" = any
    band: str = ""                       # "" = any
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    failure_pattern: str = ""            # "" = any

    def matches(self, domain: str, difficulty: float, band: str = "",
                failure_pattern: str = "") -> bool:
        if self.domain and self.domain != domain:
            return False
        if self.band and self.band != band:
            return False
        if not (self.min_difficulty <= difficulty <= self.max_difficulty):
            return False
        if self.failure_pattern and self.failure_pattern != failure_pattern:
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.domain:
            parts.append(f"домен={self.domain}")
        if self.band:
            parts.append(f"полоса={self.band}")
        if (self.min_difficulty, self.max_difficulty) != (0.0, 1.0):
            parts.append(f"сложность {self.min_difficulty:.2f}–{self.max_difficulty:.2f}")
        if self.failure_pattern:
            parts.append(f"отказ={self.failure_pattern}")
        return ", ".join(parts) or "любые задачи"


@dataclass
class LawEvidence:
    """Everything measured about a law, accumulated across experiments."""
    trials: int = 0
    effects: List[float] = field(default_factory=list)
    hidden_confirmations: int = 0
    hidden_contradictions: int = 0
    transfer_confirmed: List[str] = field(default_factory=list)
    transfer_failed: List[str] = field(default_factory=list)
    counterexamples_sought: int = 0
    counterexamples_found: int = 0
    experiments: List[str] = field(default_factory=list)

    @property
    def mean_effect(self) -> float:
        return sum(self.effects) / len(self.effects) if self.effects else 0.0

    @property
    def positive_share(self) -> float:
        """How often the effect went the way the law claims.

        Kept apart from the mean because they answer different questions:
        a mean of +0.05 built from +0.4 and -0.3 is not a law, it is two
        different behaviours averaged into one misleading number.
        """
        if not self.effects:
            return 0.0
        return sum(1 for e in self.effects if e > 0) / len(self.effects)


@dataclass
class CognitiveLaw:
    law_id: str
    condition: Condition
    intervention: Tuple[str, ...]
    claimed_effect: str
    evidence: LawEvidence = field(default_factory=LawEvidence)
    exceptions: List[str] = field(default_factory=list)
    status: str = PROPOSED
    discovered_in: str = ""              # the domain it was found in
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)

    # ---------- status, derived only ----------

    def _derive_status(self) -> str:
        """What the accumulated evidence supports. Never what anyone wants.

        Checked in order of severity: refutation first, because a law that
        has been contradicted must not be able to hold a higher status by
        also having some supporting evidence.
        """
        ev = self.evidence
        if ev.trials and ev.positive_share < 0.5 and ev.trials >= MIN_TRIALS_FOR_SUPPORT:
            return REFUTED
        if ev.hidden_contradictions > ev.hidden_confirmations:
            return REFUTED
        if ev.counterexamples_found > 0:
            return REFUTED
        if ev.trials < MIN_TRIALS_FOR_SUPPORT or ev.hidden_confirmations < 1:
            return PROPOSED
        transferred = len(ev.transfer_confirmed) >= MIN_TRANSFER_DOMAINS
        searched = ev.counterexamples_sought > 0
        if transferred and searched:
            return VALIDATED
        return SUPPORTED

    def record_evidence(self, *, effect: Optional[float] = None, trials: int = 0,
                        hidden_confirmed: Optional[bool] = None,
                        transfer_domain: str = "", transfer_ok: Optional[bool] = None,
                        counterexamples_sought: int = 0, counterexamples_found: int = 0,
                        experiment_id: str = "") -> str:
        """Fold in one experiment's result and re-derive the status.

        Returns the new status. Demotion is as automatic as promotion --
        a VALIDATED law that fails on fresh evidence inside its own stated
        condition becomes REFUTED here, without anyone deciding to check.
        """
        before = self.status
        ev = self.evidence
        if effect is not None:
            ev.effects.append(float(effect))
        ev.trials += max(0, int(trials))
        if hidden_confirmed is True:
            ev.hidden_confirmations += 1
        elif hidden_confirmed is False:
            ev.hidden_contradictions += 1
        if transfer_domain and transfer_ok is not None:
            target = ev.transfer_confirmed if transfer_ok else ev.transfer_failed
            if transfer_domain not in target:
                target.append(transfer_domain)
        ev.counterexamples_sought += max(0, int(counterexamples_sought))
        ev.counterexamples_found += max(0, int(counterexamples_found))
        if experiment_id:
            ev.experiments.append(experiment_id)

        self.status = self._derive_status()
        self.updated = time.time()
        if self.status != before:
            self.history.append({"at": self.updated, "from": before, "to": self.status,
                                 "experiment": experiment_id,
                                 "trials": ev.trials, "mean_effect": round(ev.mean_effect, 4)})
        return self.status

    def add_exception(self, description: str) -> None:
        """Record a case the law does not cover.

        An exception is not a refutation: a law that holds except under a
        stated condition is more useful than one deleted for being
        imperfect. What matters is that the exception is written down,
        because an unstated one is indistinguishable from a law that is
        simply wrong.
        """
        if description and description not in self.exceptions:
            self.exceptions.append(description)
            self.updated = time.time()

    # ---------- reporting ----------

    def describe(self) -> str:
        ev = self.evidence
        return (f"[{self.status}] если {self.condition.describe()}, "
                f"то {' → '.join(self.intervention)} даёт {self.claimed_effect} "
                f"(эффект {ev.mean_effect:+.3f} по {ev.trials} испытаниям, "
                f"положительных {ev.positive_share:.0%})")

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["condition"] = asdict(self.condition)
        payload["intervention"] = list(self.intervention)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveLaw":
        ev = data.get("evidence") or {}
        return cls(
            law_id=data["law_id"],
            condition=Condition(**(data.get("condition") or {})),
            intervention=tuple(data.get("intervention") or ()),
            claimed_effect=data.get("claimed_effect", ""),
            evidence=LawEvidence(**ev),
            exceptions=list(data.get("exceptions") or []),
            status=data.get("status", PROPOSED),
            discovered_in=data.get("discovered_in", ""),
            created=float(data.get("created") or time.time()),
            updated=float(data.get("updated") or time.time()),
            history=list(data.get("history") or []))


class LawBook:
    """The laws MANA currently holds, and the ones it has abandoned.

    Refuted laws are kept. A system that deletes its failures repeats
    them, and "we tried this and it did not hold" is one of the more
    valuable things a research loop can know.
    """

    def __init__(self, laws: Optional[Sequence[CognitiveLaw]] = None) -> None:
        self._laws: Dict[str, CognitiveLaw] = {law.law_id: law for law in (laws or ())}

    def propose(self, condition: Condition, intervention: Sequence[str],
                claimed_effect: str, discovered_in: str = "") -> CognitiveLaw:
        law = CognitiveLaw(law_id=uuid.uuid4().hex[:12], condition=condition,
                           intervention=tuple(intervention), claimed_effect=claimed_effect,
                           discovered_in=discovered_in)
        self._laws[law.law_id] = law
        return law

    def get(self, law_id: str) -> Optional[CognitiveLaw]:
        return self._laws.get(law_id)

    def all(self) -> List[CognitiveLaw]:
        return list(self._laws.values())

    def by_status(self, status: str) -> List[CognitiveLaw]:
        return [law for law in self._laws.values() if law.status == status]

    def applicable(self, domain: str, difficulty: float, band: str = "",
                   failure_pattern: str = "") -> List[CognitiveLaw]:
        """Laws that claim to cover this situation and have not been refuted.

        A refuted law is never returned. It stays in the book as a record
        of what was tried, but acting on it would be acting on something
        the evidence has already contradicted.
        """
        return [law for law in self._laws.values()
                if law.status != REFUTED
                and law.condition.matches(domain, difficulty, band, failure_pattern)]

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for law in self._laws.values():
            counts[law.status] = counts.get(law.status, 0) + 1
        return {"total": len(self._laws), "by_status": counts,
                "validated": [law.describe() for law in self.by_status(VALIDATED)],
                "refuted": [law.describe() for law in self.by_status(REFUTED)]}

    # ---------- persistence ----------

    def to_dict(self) -> Dict[str, Any]:
        return {"laws": [law.as_dict() for law in self._laws.values()]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LawBook":
        return cls([CognitiveLaw.from_dict(row) for row in (data.get("laws") or [])])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "LawBook":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return cls()
