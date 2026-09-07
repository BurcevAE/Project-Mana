"""
mana.cognition.series — reading a run of findings, and refusing to explain it.

What this is for
----------------
`findings.py` answers "did we try this". This answers the next question:
across everything tried against one question, **where did the outcome
change, and what was different there**.

    depth=2, corpus=1000  -> NOT_BETTER
    depth=2, corpus=5000  -> BETTER

    varied: corpus
    class changed: NOT_BETTER -> BETTER
    isolated: yes (exactly one condition differed)

Deliberately stupid
-------------------
It compares recorded conditions and derived classes. It does not model,
weigh, or infer. Everything it reports is a difference between two rows
that were written down.

The sentence it may not say
---------------------------
It may say:

    "при изменении corpus между двумя наблюдениями класс изменился
     с NOT_BETTER на BETTER"

It may not say:

    "увеличение корпуса улучшает результат"

The first is a description of two records. The second is a causal claim
from two points, and this project's whole discipline is that a claim
rests on a measurement. A reader that quietly crossed that line would put
an explanation into the ledger with the authority of an observation, and
everything built on top would inherit it. A test asserts no causal
vocabulary appears in what this module produces.

Confounding is the thing that makes the line real
--------------------------------------------------
Two observations differing in one condition let you name the condition
that differed. Two observations differing in three do not -- and the
tempting move is to name whichever one you were interested in.

So every comparison is marked `isolated` or not, and a confounded one is
reported with all of what changed, never with a favourite. This is the
whole reason the reader is worth having: without it, a series of
experiments becomes a place to find the answer you came with.

What it cannot classify, it says so about
------------------------------------------
A finding whose measurement lacks what `classify` needs comes back
UNCLASSIFIED. A comparison involving one is reported as unusable rather
than being read as "the class stayed the same", which would turn a
missing measurement into evidence of stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .findings import Finding, Ledger, UNCLASSIFIED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.2"


@dataclass(frozen=True)
class Observation:
    """A finding reduced to what a series cares about."""
    finding_id: str
    conditions: Dict[str, Any]
    failure: str
    verdict: str
    created: float
    #: Which method this observation used. Carried because a comparison
    #: that looks only at conditions would call two different methods
    #: "isolated on corpus_games" and hide the larger difference.
    approach_id: str = ""

    @classmethod
    def of(cls, finding: Finding) -> "Observation":
        return cls(approach_id=finding.approach_id,
                   finding_id=finding.finding_id,
                   conditions=dict(finding.conditions),
                   failure=finding.failure.failure,
                   verdict=finding.verdict,
                   created=finding.created)

    def as_dict(self) -> Dict[str, Any]:
        return {"finding_id": self.finding_id, "conditions": self.conditions,
                "failure": self.failure, "verdict": self.verdict,
                "created": self.created}


@dataclass(frozen=True)
class Comparison:
    """Two observations, what differed, and whether that is attributable."""
    left: str
    right: str
    changed: Dict[str, Dict[str, Any]]
    held: Tuple[str, ...]
    only_in_one: Tuple[str, ...]
    from_class: str
    to_class: str
    #: True when both sides used the same method. False makes the pair
    #: confounded whatever the conditions say.
    same_approach: bool = True

    @property
    def class_changed(self) -> bool:
        return self.from_class != self.to_class

    @property
    def usable(self) -> bool:
        """False when either side could not be classified.

        Reported rather than treated as "the class stayed the same": a
        missing measurement is not evidence of stability.
        """
        return UNCLASSIFIED not in (self.from_class, self.to_class)

    @property
    def isolated(self) -> bool:
        """Exactly one condition differed, and both sides recorded the
        same set of conditions.

        Only an isolated comparison lets a reader name the condition that
        differed. With three differences the tempting move is to name
        whichever one you were interested in, which is how a series of
        experiments becomes a place to find the answer you came with.
        """
        return (self.same_approach and len(self.changed) == 1
                and not self.only_in_one)

    def describe(self) -> str:
        if not self.usable:
            return ("сравнивать нельзя: одну из сторон не удалось "
                    "классифицировать")
        names = ", ".join(sorted(self.changed)) or "ничего"
        if not self.class_changed:
            return (f"при изменении {names} класс не изменился "
                    f"({self.from_class})")
        head = (f"при изменении {names} между двумя наблюдениями класс "
                f"изменился с {self.from_class} на {self.to_class}")
        if self.isolated:
            return head
        if not self.same_approach:
            return (head + ". Подход тоже другой — приписать перемену "
                    "условию нельзя")
        return (head + ". Изменилось больше одного условия — приписать "
                "перемену какому-то одному нельзя")

    def as_dict(self) -> Dict[str, Any]:
        return {"left": self.left, "right": self.right,
                "changed": self.changed, "held": list(self.held),
                "only_in_one": list(self.only_in_one),
                "from_class": self.from_class, "to_class": self.to_class,
                "class_changed": self.class_changed,
                "isolated": self.isolated, "usable": self.usable,
                "same_approach": self.same_approach,
                "describe": self.describe()}


def _compare(left: Observation, right: Observation) -> Comparison:
    keys_left, keys_right = set(left.conditions), set(right.conditions)
    shared = keys_left & keys_right
    changed: Dict[str, Dict[str, Any]] = {}
    held: List[str] = []
    for key in sorted(shared):
        was, now = left.conditions[key], right.conditions[key]
        if was == now:
            held.append(key)
        else:
            changed[key] = {"was": was, "now": now}
    return Comparison(left=left.finding_id, right=right.finding_id,
                      changed=changed, held=tuple(held),
                      only_in_one=tuple(sorted(keys_left ^ keys_right)),
                      from_class=left.failure, to_class=right.failure,
                      same_approach=(left.approach_id == right.approach_id))


@dataclass(frozen=True)
class Series:
    """Everything tried against one question, and where it differed."""
    question: str
    observations: Tuple[Observation, ...] = ()
    comparisons: Tuple[Comparison, ...] = ()

    @property
    def varied(self) -> Tuple[str, ...]:
        """Conditions that took more than one value across the series."""
        values: Dict[str, set] = {}
        for observation in self.observations:
            for key, value in observation.conditions.items():
                values.setdefault(key, set()).add(_hashable(value))
        return tuple(sorted(k for k, v in values.items() if len(v) > 1))

    @property
    def constant(self) -> Tuple[str, ...]:
        """Conditions recorded in EVERY observation with one value.

        Presence in all of them is required, and that was the defect:
        a key written down on one side only took a single value and was
        reported as held constant, which is the opposite of true. What
        was not recorded was not held -- it is unknown.
        """
        counts: Dict[str, int] = {}
        values: Dict[str, set] = {}
        for observation in self.observations:
            for key, value in observation.conditions.items():
                counts[key] = counts.get(key, 0) + 1
                values.setdefault(key, set()).add(_hashable(value))
        total = len(self.observations)
        return tuple(sorted(k for k, v in values.items()
                            if len(v) == 1 and counts.get(k) == total))

    @property
    def partial(self) -> Tuple[str, ...]:
        """Conditions recorded in some observations but not all.

        Neither varied nor held: simply unknown where they are missing,
        and every comparison touching them is confounded.
        """
        counts: Dict[str, int] = {}
        for observation in self.observations:
            for key in observation.conditions:
                counts[key] = counts.get(key, 0) + 1
        total = len(self.observations)
        return tuple(sorted(k for k, n in counts.items() if n != total))

    @property
    def approaches(self) -> Tuple[str, ...]:
        """How many distinct methods this question has been attacked with."""
        return tuple(sorted({o.approach_id for o in self.observations}))

    @property
    def classes(self) -> Tuple[str, ...]:
        return tuple(o.failure for o in self.observations)

    @property
    def flips(self) -> Tuple[Comparison, ...]:
        """Comparisons where the class changed. The point of the series."""
        return tuple(c for c in self.comparisons if c.usable and c.class_changed)

    @property
    def isolated_flips(self) -> Tuple[Comparison, ...]:
        """Flips where exactly one condition differed.

        The only ones that let a reader name a condition -- and even then
        only name it, never credit it.
        """
        return tuple(c for c in self.flips if c.isolated)

    @property
    def confounded_flips(self) -> Tuple[Comparison, ...]:
        return tuple(c for c in self.flips if not c.isolated)

    def summary(self) -> Dict[str, Any]:
        return {"question": self.question,
                "observations": len(self.observations),
                "classes": list(self.classes),
                "varied": list(self.varied),
                "constant": list(self.constant),
                "partial": list(self.partial),
                "approaches": len(self.approaches),
                "comparisons": len(self.comparisons),
                "unusable": sum(1 for c in self.comparisons if not c.usable),
                "flips": len(self.flips),
                "isolated_flips": len(self.isolated_flips),
                "confounded_flips": len(self.confounded_flips)}

    def describe(self) -> str:
        lines = [f"вопрос: {self.question}",
                 f"наблюдений: {len(self.observations)}"]
        for observation in self.observations:
            shown = ", ".join(f"{k}={v}" for k, v in
                              sorted(observation.conditions.items()))
            lines.append(f"  {observation.failure:26} {shown}")

        if len(self.observations) < 2:
            lines.append("серия из одного наблюдения — сравнивать не с чем")
            return "\n".join(lines)

        lines.append(f"варьировалось: {', '.join(self.varied) or 'ничего'}")
        lines.append(f"держалось постоянным: {', '.join(self.constant) or 'ничего'}")
        if self.partial:
            lines.append(f"записано не везде (не известно, а не постоянно): "
                         f"{', '.join(self.partial)}")
        if not self.flips:
            lines.append("переворотов класса не обнаружено")
        for comparison in self.flips:
            lines.append("  " + comparison.describe())
        return "\n".join(lines)


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def read(question: str, ledger: Optional[Ledger] = None) -> Series:
    """The series of findings about one question, oldest first.

    Oldest first because a series is read as a history: the order the
    observations were made in is itself information, and sorting by
    outcome would put the answer first and the evidence after it.
    """
    ledger = ledger if ledger is not None else Ledger()
    findings = sorted(ledger.about(question), key=lambda f: f.created)
    observations = tuple(Observation.of(f) for f in findings)
    comparisons = tuple(_compare(observations[i], observations[j])
                        for i in range(len(observations))
                        for j in range(i + 1, len(observations)))
    return Series(question=question, observations=observations,
                  comparisons=comparisons)


def questions(ledger: Optional[Ledger] = None) -> List[str]:
    """Every question the ledger has findings about."""
    ledger = ledger if ledger is not None else Ledger()
    return sorted({f.question for f in ledger.latest()})


def all_series(ledger: Optional[Ledger] = None) -> List[Series]:
    ledger = ledger if ledger is not None else Ledger()
    return [read(q, ledger) for q in questions(ledger)]
