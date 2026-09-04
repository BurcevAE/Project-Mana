"""
mana.cognition.gaps — from measured capability to what is worth learning
next.

    capability -> failure pattern -> weakness -> gap -> priority

The chain matters because each arrow discards something. A low score is
not a weakness (it may be a hard domain measured twice); a weakness is not
a gap (it may be one nothing can be done about); and a gap is not a
priority (it may be expensive to close and worth little when closed).

Two kinds of gap, and they compete
----------------------------------
  * **Competence gap** -- the score is genuinely low. Closing it needs a
    better mechanism.
  * **Knowledge gap** -- the score is uncertain. Closing it needs
    measurement, not invention.

Both belong in the same queue because they compete for the same budget,
and a system that always chases the lower score will keep improving
whatever it happened to measure badly first. Expected information gain is
what lets a merely uncertain capability outrank a confidently poor one --
that is §12's requirement, and it is computable here rather than being a
sentiment: the Wilson interval has a width, and more samples shrink it by
an amount arithmetic can predict.

Honest about the weights
------------------------
The priority formula combines severity, uncertainty, frequency, expected
information gain, expected capability gain and cost. The weights are a
design choice, not a measurement, and they are named as such in
`PRIORITY_WEIGHTS` so that changing them is a visible act. Nothing here
pretends the ranking is derived from data it does not have.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .self_model import (MIN_OBSERVATIONS, Capability, SelfModel,
                         wilson_interval)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

COMPETENCE = "competence"      # measured, and low
KNOWLEDGE = "knowledge"        # not measured well enough to say

#: Declared, not derived. Changing these changes what MANA studies next,
#: which is too consequential to leave implicit in a sum.
PRIORITY_WEIGHTS = {
    "severity": 1.0,           # how far below acceptable the capability is
    "uncertainty": 0.6,        # how little we know about it
    "frequency": 0.5,          # how often tasks of this kind actually appear
    "information_gain": 0.8,   # how much measuring would tell us
    "capability_gain": 0.7,    # how much closing it would raise overall competence
    "cost": -0.4,              # what closing it would take
}

#: Below this a capability is treated as needing work. Not a claim about
#: what is achievable -- a threshold for attention.
COMPETENT = 0.75

#: How many further observations an information-gain estimate assumes. A
#: fixed horizon so that gaps are compared on equal terms rather than each
#: being scored against the batch someone imagined for it.
PROBE_SIZE = 20


@dataclass(frozen=True)
class Gap:
    gap_id: str
    kind: str
    capability_id: str
    domain: str
    band: str
    description: str
    severity: float
    uncertainty: float
    frequency: float
    information_gain: float
    capability_gain: float
    cost: float
    priority: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return (f"[{self.priority:.2f}] {self.capability_id}: {self.description} "
                f"→ {self.suggested_action}")


def expected_information_gain(capability: Capability, probe: int = PROBE_SIZE) -> float:
    """How much narrower the interval would get after `probe` more trials.

    Computable, not felt. The Wilson width shrinks roughly as 1/sqrt(n), so
    the value of measuring again is high when n is small and collapses once
    it is large -- which is exactly the behaviour §12 asks for: an
    experiment with a slightly worse immediate score can outrank one with a
    better score if it resolves more.

    Assumes the observed rate continues. That is the honest neutral
    assumption: predicting the rate would change is predicting the result
    of the experiment being valued.
    """
    trials = capability.observations
    if trials <= 0:
        return 1.0
    rate = capability.score
    current_lo, current_hi = wilson_interval(capability.successes, trials)
    projected_successes = int(round(rate * (trials + probe)))
    future_lo, future_hi = wilson_interval(projected_successes, trials + probe)
    return max(0.0, (current_hi - current_lo) - (future_hi - future_lo))


def _frequency(capability: Capability, all_caps: Dict[str, Capability]) -> float:
    """Share of observed work this capability accounts for.

    A weakness in something that never comes up is a curiosity; the same
    weakness in the commonest task type is the system's main problem.
    """
    total = sum(c.observations for c in all_caps.values() if c.band == "all") or 1
    return capability.observations / total


def _capability_gain(capability: Capability) -> float:
    """How much overall competence would rise if this were fixed.

    Bounded by how far it is from competent AND by how much of the work it
    covers -- lifting a rare capability from 0.2 to 0.9 moves the system
    less than lifting a common one from 0.6 to 0.75.
    """
    headroom = max(0.0, COMPETENT - capability.score)
    return headroom


def _cost(capability: Capability) -> float:
    """Rough cost of working on this, in units of "a call".

    Uses the measured mean calls per attempt where there is one, because a
    capability that already costs three calls per task will cost more to
    study than one that costs one.
    """
    return max(0.2, capability.mean_calls or 1.0) * PROBE_SIZE / 20.0


def detect(model: SelfModel, competent: float = COMPETENT) -> List[Gap]:
    """Every gap the evidence supports, ranked.

    Only band-level slices are turned into gaps: a whole-domain roll-up
    says a domain deserves attention, but "arithmetic is 0.6" is not
    something a learning goal can be written against, while "arithmetic
    above difficulty 0.65 is 0.2" is.
    """
    caps = model.capabilities()
    gaps: List[Gap] = []

    for key, cap in sorted(caps.items()):
        if cap.band == "all":
            continue

        info_gain = expected_information_gain(cap)
        frequency = _frequency(cap, caps)
        cost = _cost(cap)

        if not cap.measured:
            # A knowledge gap. Severity is deliberately low: not knowing is
            # not the same as being bad, and treating it as equivalent
            # would send the system to fix things that may not be broken.
            gap = _build(
                kind=KNOWLEDGE, cap=cap, description=(
                    f"недостаточно наблюдений ({cap.observations}/{MIN_OBSERVATIONS}), "
                    f"утверждать что-либо нельзя"),
                severity=0.25, uncertainty=cap.uncertainty, frequency=frequency,
                information_gain=info_gain, capability_gain=0.0, cost=cost,
                action=f"измерить: {PROBE_SIZE} задач {cap.domain}/{cap.band}")
            gaps.append(gap)
            continue

        lo, hi = cap.confidence_interval
        if hi < competent:
            # A competence gap with the interval entirely below the bar --
            # the strongest statement the data can make.
            dominant = max(cap.failure_modes.items(), key=lambda kv: kv[1], default=("", 0))
            reason = f", преобладающий отказ: {dominant[0]}" if dominant[0] else ""
            gaps.append(_build(
                kind=COMPETENCE, cap=cap,
                description=(f"{cap.score:.2f} [{lo:.2f}–{hi:.2f}] — весь интервал "
                             f"ниже {competent:.2f}{reason}"),
                severity=competent - hi, uncertainty=cap.uncertainty, frequency=frequency,
                information_gain=info_gain, capability_gain=_capability_gain(cap), cost=cost,
                action=_action_for(cap, dominant[0])))
        elif lo < competent <= hi:
            # Straddles the bar: the honest reading is that we cannot yet
            # tell, so this is a knowledge gap even though it looks like a
            # competence one.
            gaps.append(_build(
                kind=KNOWLEDGE, cap=cap,
                description=(f"{cap.score:.2f} [{lo:.2f}–{hi:.2f}] — интервал пересекает "
                             f"{competent:.2f}, различить нельзя"),
                severity=max(0.0, competent - cap.score) * 0.5,
                uncertainty=cap.uncertainty, frequency=frequency,
                information_gain=info_gain, capability_gain=_capability_gain(cap) * 0.5,
                cost=cost, action=f"уточнить: ещё {PROBE_SIZE} задач {cap.domain}/{cap.band}"))

    gaps.sort(key=lambda gap: -gap.priority)
    return gaps


def _action_for(cap: Capability, dominant_failure: str) -> str:
    """What to try, named from the failure pattern rather than in general.

    "Improve arithmetic" is not an action. "Most failures are format
    violations, so try a program that ends with a formatting step" is.
    """
    if dominant_failure == "format":
        return "отказы по формату — попробовать программу с шагом приведения ответа к формату"
    if dominant_failure == "ungradable":
        return "нечем оценивать — проверить песочницу, а не способность"
    # Checked before the difficulty band: a measured split between brains
    # is both more specific and cheaper to act on than "this band is
    # hard", and the band test used to shadow it entirely -- a hard slice
    # where one brain scored 1.0 and another 0.0 was told to try
    # decomposition instead of to stop using the failing brain.
    if cap.by_brain and max(cap.by_brain.values()) - min(cap.by_brain.values()) > 0.25:
        best = max(cap.by_brain, key=cap.by_brain.get)
        return f"разброс между мозгами — направлять этот класс задач на {best}"
    if cap.band == "hard":
        return "сложные случаи — попробовать декомпозицию или проверку перед ответом"
    return "попробовать другую когнитивную программу на этом срезе"


def _build(kind: str, cap: Capability, description: str, severity: float,
           uncertainty: float, frequency: float, information_gain: float,
           capability_gain: float, cost: float, action: str) -> Gap:
    priority = (
        PRIORITY_WEIGHTS["severity"] * severity +
        PRIORITY_WEIGHTS["uncertainty"] * uncertainty +
        PRIORITY_WEIGHTS["frequency"] * frequency +
        PRIORITY_WEIGHTS["information_gain"] * information_gain +
        PRIORITY_WEIGHTS["capability_gain"] * capability_gain +
        PRIORITY_WEIGHTS["cost"] * min(1.0, cost)
    )
    return Gap(
        gap_id=f"{cap.capability_id}:{kind}", kind=kind,
        capability_id=cap.capability_id, domain=cap.domain, band=cap.band,
        description=description,
        severity=round(severity, 4), uncertainty=round(uncertainty, 4),
        frequency=round(frequency, 4), information_gain=round(information_gain, 4),
        capability_gain=round(capability_gain, 4), cost=round(cost, 4),
        priority=round(priority, 4),
        evidence={"observations": cap.observations, "successes": cap.successes,
                  "score": round(cap.score, 4),
                  "interval": [round(x, 4) for x in cap.confidence_interval],
                  "failure_modes": dict(cap.failure_modes),
                  "by_brain": {k: round(v, 3) for k, v in cap.by_brain.items()}},
        suggested_action=action)


def learning_goals(model: SelfModel, limit: int = 3) -> List[Dict[str, Any]]:
    """The top gaps, as things to act on.

    Deliberately few. A curriculum that pursues everything pursues
    nothing, and the budget that makes a discovery possible is the same
    budget a long queue spreads too thin to conclude anything with.
    """
    return [{"goal": gap.suggested_action, "gap": gap.gap_id, "kind": gap.kind,
             "priority": gap.priority, "why": gap.description}
            for gap in detect(model)[:limit]]
