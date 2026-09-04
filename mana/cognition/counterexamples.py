"""
mana.cognition.counterexamples — asking where a law fails, not whether it
works.

The difference is not rhetorical. "Does VERIFY-before-ANSWER help on hard
arithmetic?" is answered by sampling hard arithmetic, and sampling the
condition a law was derived from mostly re-measures the thing that
produced it. "Where does it stop helping?" is answered by probing the
*edges* -- and the edges are where a law's real scope lives.

Where the probes come from
--------------------------
Derived from the law itself, not sampled at random. Random adversarial
generation would be cheap to write and would mostly produce tasks the law
never claimed to cover, which refutes nothing and costs a budget the
project does not have. Four systematic sources:

  * **Boundary.** A law saying "difficulty >= 0.65" is probed at 0.60 and
    0.70. If the effect is identical on both sides, the stated boundary is
    not where the mechanism changes, and the law's condition is wrong even
    if its effect is real.
  * **Adjacent domain.** The same difficulty in a domain the law does not
    claim. A law that also holds there is understated; one that reverses
    there has found its edge.
  * **Adjacent band.** Easier and harder than claimed, for the same
    reason.
  * **Failure modes of its own operators.** Every operator declares how it
    can fail (`ir.CognitiveOperator.failure_modes`). A law built on
    CRITIQUE inherits "self-review when the critic is the author", so the
    probe that removes brain diversity is not a guess -- it is the law's
    own machinery, named by the machinery.

What counts as a counterexample
-------------------------------
Not "the candidate lost once". A counterexample is a *condition* under
which the effect the law claims is absent or reversed, measured with
enough trials to say so. One bad task is noise; a band where the effect
reverses is the law's boundary.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import gates
from ..core.gates import PairedOutcome
from .laws import CognitiveLaw, Condition

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

BOUNDARY = "boundary"
ADJACENT_DOMAIN = "adjacent_domain"
ADJACENT_BAND = "adjacent_band"
OPERATOR_FAILURE = "operator_failure"

#: How much the effect must reverse before a probe counts as a
#: counterexample rather than as noise. A law claiming +0.18 that
#: measures +0.02 somewhere has found a limit; one that measures -0.15 has
#: found a contradiction. Both matter, and only the second is fatal.
REVERSAL_THRESHOLD = -0.05

#: Minimum trials in a probe before its result may refute anything. Lower
#: than the acceptance gate's 30 on purpose: a probe is looking for a
#: reversal, which is a larger effect than the improvement the gate is
#: asked to detect, and probing four conditions at 30 trials each would
#: cost more than the experiment that produced the law.
MIN_PROBE_TRIALS = 12


@dataclass(frozen=True)
class Probe:
    """One place to look for failure, and why there."""
    probe_id: str
    kind: str
    condition: Condition
    rationale: str
    domain: str = ""
    difficulty: float = 0.5

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["condition"] = asdict(self.condition)
        return payload


@dataclass
class ProbeResult:
    probe: Probe
    outcomes: List[PairedOutcome] = field(default_factory=list)
    effect: float = 0.0
    is_counterexample: bool = False
    is_limit: bool = False
    note: str = ""

    def summary(self) -> Dict[str, Any]:
        return {"probe": self.probe.kind, "rationale": self.probe.rationale,
                "trials": len(self.outcomes), "effect": round(self.effect, 4),
                "counterexample": self.is_counterexample, "limit": self.is_limit,
                "note": self.note}


def design_probes(law: CognitiveLaw, all_domains: Sequence[str],
                  operators: Optional[Dict[str, Any]] = None) -> List[Probe]:
    """Where to look for this law's edges.

    Ordered cheapest-first: boundary probes reuse the law's own domain and
    are the likeliest to find a wrong condition, while operator-failure
    probes need a specific setup. A budget that runs out partway should
    have spent itself on the probes most likely to find something.
    """
    probes: List[Probe] = []
    condition = law.condition
    domain = condition.domain or law.discovered_in

    # 1. Boundary: just inside and just outside the claimed difficulty.
    for offset, side in ((-0.08, "ниже"), (0.08, "выше")):
        for edge, name in ((condition.min_difficulty, "нижней"),
                           (condition.max_difficulty, "верхней")):
            probe_difficulty = round(edge + offset, 3)
            if not 0.0 <= probe_difficulty <= 1.0:
                continue
            if condition.matches(domain, probe_difficulty) == condition.matches(
                    domain, round(edge - offset, 3)):
                continue          # the offset does not straddle this edge
            probes.append(Probe(
                probe_id=uuid.uuid4().hex[:8], kind=BOUNDARY,
                condition=Condition(domain=domain, min_difficulty=probe_difficulty,
                                    max_difficulty=probe_difficulty),
                rationale=(f"сложность {probe_difficulty} — {side} {name} границы условия; "
                           f"если эффект тот же, граница указана неверно"),
                domain=domain, difficulty=probe_difficulty))

    # 2. Adjacent domains: the same difficulty where the law does not claim.
    mid = (condition.min_difficulty + condition.max_difficulty) / 2
    for other in all_domains:
        if other == domain:
            continue
        probes.append(Probe(
            probe_id=uuid.uuid4().hex[:8], kind=ADJACENT_DOMAIN,
            condition=Condition(domain=other, min_difficulty=condition.min_difficulty,
                                max_difficulty=condition.max_difficulty),
            rationale=(f"домен {other}, который закон не заявляет; разворот эффекта "
                       f"здесь показывает его настоящую границу"),
            domain=other, difficulty=mid))

    # 3. Adjacent bands within the same domain.
    for probe_difficulty, label in ((max(0.0, condition.min_difficulty - 0.3), "легче"),
                                    (min(1.0, condition.max_difficulty + 0.3), "тяжелее")):
        if condition.matches(domain, probe_difficulty):
            continue
        probes.append(Probe(
            probe_id=uuid.uuid4().hex[:8], kind=ADJACENT_BAND,
            condition=Condition(domain=domain, min_difficulty=probe_difficulty,
                                max_difficulty=probe_difficulty),
            rationale=f"тот же домен, но {label} заявленного диапазона",
            domain=domain, difficulty=probe_difficulty))

    # 4. The declared failure modes of the law's own operators.
    if operators:
        for step in law.intervention:
            op = operators.get(step)
            for mode in getattr(op, "failure_modes", ()) or ():
                probes.append(Probe(
                    probe_id=uuid.uuid4().hex[:8], kind=OPERATOR_FAILURE,
                    condition=Condition(domain=domain,
                                        min_difficulty=condition.min_difficulty,
                                        max_difficulty=condition.max_difficulty),
                    rationale=f"{step} объявляет отказ «{mode}» — проверить его",
                    domain=domain, difficulty=mid))

    order = {BOUNDARY: 0, ADJACENT_BAND: 1, ADJACENT_DOMAIN: 2, OPERATOR_FAILURE: 3}
    probes.sort(key=lambda p: order.get(p.kind, 9))
    return probes


def evaluate_probe(probe: Probe, outcomes: Sequence[PairedOutcome],
                   claimed_effect: float) -> ProbeResult:
    """Decide what one probe found.

    Three outcomes, and collapsing them would lose the useful middle:

      * **counterexample** -- the effect reversed. Fatal to the law.
      * **limit** -- the effect vanished but did not reverse. Not fatal;
        it is an exception worth recording, and a law with a stated
        exception is more useful than one deleted for being imperfect.
      * **nothing** -- the effect held, or there were too few trials.
    """
    result = ProbeResult(probe=probe, outcomes=list(outcomes))
    if len(outcomes) < MIN_PROBE_TRIALS:
        result.note = f"мало испытаний ({len(outcomes)}/{MIN_PROBE_TRIALS})"
        return result
    effect = gates.accuracy(outcomes, "candidate") - gates.accuracy(outcomes, "baseline")
    result.effect = effect
    if effect <= REVERSAL_THRESHOLD:
        result.is_counterexample = True
        result.note = (f"эффект развернулся: {effect:+.3f} против заявленных "
                       f"{claimed_effect:+.3f}")
    elif claimed_effect > 0 and effect < claimed_effect * 0.25:
        result.is_limit = True
        result.note = (f"эффект исчез: {effect:+.3f} против заявленных "
                       f"{claimed_effect:+.3f} — граница применимости")
    else:
        result.note = f"эффект держится: {effect:+.3f}"
    return result


#: Runs the law's baseline and intervention over tasks matching a probe,
#: and returns paired outcomes. Injected so the engine is testable without
#: brains and so the same engine can drive a live agent or a simulation.
ProbeRunner = Callable[[Probe, CognitiveLaw], List[PairedOutcome]]


def search(law: CognitiveLaw, runner: ProbeRunner, all_domains: Sequence[str],
           operators: Optional[Dict[str, Any]] = None,
           max_probes: int = 6) -> Dict[str, Any]:
    """Hunt for this law's edges and fold what is found back into it.

    Records the search on the law whether or not anything was found --
    "we looked and found nothing" is the evidence that separates a
    VALIDATED law from a merely SUPPORTED one, and a law that was never
    probed has survived nothing.
    """
    probes = design_probes(law, all_domains, operators)[:max_probes]
    claimed = law.evidence.mean_effect
    results: List[ProbeResult] = []
    for probe in probes:
        try:
            outcomes = runner(probe, law)
        except Exception as exc:
            result = ProbeResult(probe=probe)
            result.note = f"проба не выполнена: {type(exc).__name__}: {exc}"
            results.append(result)
            continue
        results.append(evaluate_probe(probe, outcomes, claimed))

    counterexamples = [r for r in results if r.is_counterexample]
    limits = [r for r in results if r.is_limit]
    for limit in limits:
        law.add_exception(f"{limit.probe.condition.describe()}: {limit.note}")

    law.record_evidence(counterexamples_sought=len(results),
                        counterexamples_found=len(counterexamples))

    return {
        "law_id": law.law_id,
        "probes_run": len(results),
        "counterexamples": [r.summary() for r in counterexamples],
        "limits": [r.summary() for r in limits],
        "held": [r.summary() for r in results if not r.is_counterexample and not r.is_limit],
        "status_after": law.status,
    }
