"""
mana.core.standing — what an answer may claim to know for the task at hand.

Why this lives in core
-----------------------
`core/gates.py` is the one place that may say "accepted". This is the one
place that may say "verified for the task". The research loop outside core
observes, chooses probes and closes questions; what its answer is allowed
to claim is decided here, where MANA's own evolution cannot redefine it --
the same reason the hidden holdout lives here and not next to the thing it
measures.

The contract (docs/RESEARCH_CONTRACT.md)
-----------------------------------------
An answer does not say how likely it is to be right. It says where it is
backed, which part of the task lies outside that, and what it assumed in
order to go beyond what was seen:

    VERIFIED_FOR_TASK  every situation the task relies on is inside the
                       boundary
    CONDITIONAL        some are not; they travel with the answer
    UNEXPLAINED        no explanation the family can state fits

The boundary
------------
Every situation observed where the model agrees with what was seen is
inside (coverage). Beyond that only independent observations -- made while
the model was committed to its prediction, and confirming it -- reach
(evidence), as far as the inclusion rule allows. A model that reads the
conditions R claims, for every other condition d, "in this pattern of R the
outcome does not depend on d = v".

    observed  no further than what was seen
    claims    a situation each of whose claims an independent observation
              bore on, in a pattern seen independently
    pattern   every situation of a pattern seen independently once

The rule is an assumption, and the answer carries it (`ASSUMPTIONS`).
Measured in `cognition/inquiry.py` 1.14 on worlds with a known truth:
`observed` is honest by construction and verifies a task only by walking
it; `claims` put no error inside its boundary on four worlds and broke on
the fifth, built for it, in 9 runs of 400 -- every one where two conditions
the model ignored interacted; `pattern` put errors inside in up to 13 of
50 answers. It is defined here so that the audit can name it, not so that
anything should rely on it.

Honesty
-------
`audit()` takes the situations where an answer is wrong -- computed by an
oracle, never asked of the model -- and splits them by the boundary. An
answer is honest when none of them lies inside. That is a count, not a
calibration.

What is not here
-----------------
Which probe to take next, what an error costs, when to stop looking: all
of that is policy and lives outside core, in the research loop.

Preconditions the caller owns
------------------------------
`observed` holds only situations where the model's prediction matched the
outcome, and `independent` only situations the model predicted in advance
and got right.

Whether an observed situation is a known one is itself a claim: that the
outcome there is the same every time. The answer says how it stands
(`repeatability`). ASSUMED -- a situation seen once counts, and the answer
declares it assumed the world repeats itself. CHECKED -- `observed` and
`independent` hold only situations whose outcome was seen more than once,
the same every time. No number of repeats proves the claim; one different
outcome refutes it. Measured on `world/universe.py`: with the claim assumed,
an effect the world gives four times in five was called verified in seven
runs of ten.

When the answer is a class -- explanations no available probe tells apart
-- `independent` is the class's: every member predicted each such
observation alike. The research loop measured here counts evidence per
member, and the first audit through this module found 13-18% of answers
whose boundary depended on which member broke a tie of weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (Any, Dict, FrozenSet, Hashable, Iterable, Mapping, Sequence,
                    Tuple)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

# How an answer stands for the task at hand.
VERIFIED_FOR_TASK = "VERIFIED_FOR_TASK"
CONDITIONAL = "CONDITIONAL"
UNEXPLAINED = "UNEXPLAINED"

# How far an answer is taken to hold beyond what was observed.
OBSERVED = "observed"
CLAIMS = "claims"
PATTERN = "pattern"
RULES = (OBSERVED, CLAIMS, PATTERN)

#: What each rule assumes beyond the observations, said in words, because
#: the answer carries it to whoever acts on it.
ASSUMPTIONS = {
    OBSERVED: "никакого: только наблюдённое",
    CLAIMS: "условия, которые модель не читает, не взаимодействуют между собой",
    PATTERN: "исход не зависит ни от чего, чего модель не читает",
}

# Whether an outcome seen in a situation is taken to be the outcome there.
REPEATABLE_ASSUMED = "assumed"
REPEATABLE_CHECKED = "checked"
REPEATABILITY = {
    REPEATABLE_ASSUMED: "исход, увиденный в ситуации однажды, считается исходом этой ситуации",
    REPEATABLE_CHECKED: "ситуация считается известной, только если её исход повторился",
}

Conditions = Mapping[str, Any]
#: A situation: a key naming it, and the conditions that hold in it.
Situation = Tuple[Hashable, Conditions]


def pattern_of(reads: Sequence[str], conditions: Conditions) -> tuple:
    """The values of the conditions the model reads."""
    return tuple(bool(conditions[c]) for c in reads)


def claims_at(reads: Sequence[str], conditions: Conditions) -> FrozenSet[tuple]:
    """The claims of a model reading `reads` that a situation bears on: in
    its pattern, the outcome does not depend on d = v, for each condition d
    the model does not read."""
    pattern = pattern_of(reads, conditions)
    return frozenset((pattern, d, bool(v)) for d, v in conditions.items() if d not in reads)


def inside(space: Iterable[Situation], reads: Sequence[str], observed: Iterable[Hashable],
           independent: Iterable[Conditions], rule: str) -> FrozenSet[Hashable]:
    """The boundary: the situations of `space` where the answer is backed."""
    if rule not in RULES:
        raise ValueError(f"unknown inclusion rule: {rule!r}")
    reads = tuple(reads)
    backed = set(observed)
    independent = list(independent)
    if rule != OBSERVED and independent:
        patterns = {pattern_of(reads, c) for c in independent}
        if rule == PATTERN:
            backed |= {key for key, c in space if pattern_of(reads, c) in patterns}
        else:
            evidenced: set = set()
            for c in independent:
                evidenced |= claims_at(reads, c)
            backed |= {key for key, c in space
                       if pattern_of(reads, c) in patterns and claims_at(reads, c) <= evidenced}
    return frozenset(backed)


@dataclass(frozen=True)
class Answer:
    """An answer and how it stands for the task."""
    subject: str
    status: str
    model: str = ""
    rule: str = ""
    assumption: str = ""
    inside: FrozenSet[Hashable] = frozenset()
    uncovered: Tuple[Hashable, ...] = ()
    uncovered_share: float = 0.0
    observed: int = 0
    independent: int = 0
    repeatability: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "status": self.status, "model": self.model,
                "rule": self.rule, "assumption": self.assumption,
                "inside": len(self.inside), "uncovered": list(self.uncovered),
                "uncovered_share": self.uncovered_share, "observed": self.observed,
                "independent": self.independent, "repeatability": self.repeatability}


def standing(subject: str, model: str, space: Sequence[Situation], reads: Sequence[str],
             observed: Iterable[Hashable], independent: Iterable[Conditions], rule: str,
             task: Mapping[Hashable, float],
             repeatability: str = REPEATABLE_ASSUMED) -> Answer:
    """How an answer stands for a task that relies on the situations in
    `task`, each by its weight. A situation of weight 0 is not relied on.
    `repeatability`: whether `observed` and `independent` were limited to
    situations whose outcome repeated (see the module docstring)."""
    if repeatability not in REPEATABILITY:
        raise ValueError(f"unknown repeatability: {repeatability!r}")
    observed = frozenset(observed)
    independent = list(independent)
    backed = inside(space, reads, observed, independent, rule)
    relied = [(key, float(w)) for key, w in task.items() if w > 0.0]
    total = sum(w for _, w in relied)
    out = tuple(key for key, _ in relied if key not in backed)
    share = sum(w for key, w in relied if key not in backed) / total if total > 0 else 0.0
    return Answer(subject=subject, status=CONDITIONAL if out else VERIFIED_FOR_TASK,
                  model=model, rule=rule, assumption=ASSUMPTIONS[rule], inside=backed,
                  uncovered=out, uncovered_share=share, observed=len(observed),
                  independent=len(independent), repeatability=repeatability)


def unexplained(subject: str, observed: int = 0) -> Answer:
    """No explanation fits: the answer claims nothing, anywhere."""
    return Answer(subject=subject, status=UNEXPLAINED, observed=observed)


@dataclass(frozen=True)
class Audit:
    """The situations where an answer is wrong, split by its boundary."""
    inside: FrozenSet[Hashable]
    outside: FrozenSet[Hashable]

    @property
    def honest(self) -> bool:
        return not self.inside


def audit(answer: Answer, wrong: Iterable[Hashable]) -> Audit:
    """`wrong`: where the answer is wrong, as an oracle computed it."""
    wrong = frozenset(wrong)
    return Audit(inside=wrong & answer.inside, outside=wrong - answer.inside)
