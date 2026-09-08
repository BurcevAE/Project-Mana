"""
mana.cognition.probes — which condition to vary next, from what has been tried.

Where this sits
---------------
    findings.py   what was tried, and what came of it
    series.py     where the class changed and what differed
    THIS MODULE   which axis is worth probing next
    experiments.select   the choice itself -- already written

The connection, and what is not duplicated
-------------------------------------------
`experiments.py` plans experiments in the synthetic pipeline space: a
`Hypothesis` there carries a gap id, a domain, a band and two step
chains, and `plan()` prices it against the self-model's capability table.
A findings-space question -- "does a learned evaluation beat material at
corpus=5000" -- has none of those. Forcing it into that vocabulary would
mean inventing a gap id and letting `information_gain` be computed from a
capability table that knows nothing about the question.

So the **shape** differs and the **decision rule** is shared.
`experiments.select` touches exactly two attributes of a plan --
`estimated_calls` and `value` -- so a probe carrying those goes through
the existing selector, with the existing `VALUE_WEIGHTS` and the existing
`MIN_EXPERIMENT_VALUE` floor. Writing a second selector here would let
the findings space choose experiments on easier terms than everything
else, which is the loophole this project keeps closing.

Information, computed from what was observed
---------------------------------------------
No causal reasoning enters. The ordering below is about what has and has
not been looked at:

    never varied                     1.0   nothing is known about this axis
    in a confounded flip             0.9   isolating it would resolve an
                                           ambiguity that already exists
    varied, a flip was found         0.3   a boundary is already visible here
    varied, no flip, k values     1/(1+k)  each further point on an axis
                                           that has not moved says less

Those numbers are a stated policy, not a discovery. They are here to be
argued with from evidence, like `VALUE_WEIGHTS` and `PRIORITY_WEIGHTS`.

Capability gain is omitted, and that is said out loud
------------------------------------------------------
`experiments.plan` weighs how much a slice would gain if the change
worked, read off the self-model's capability table. For an arbitrary
question there is no such table, and this project's own rule is that
**unmeasured is not zero**. So the capability term is not silently set to
zero -- it is left out of the value entirely, and every probe says so.
A value computed from information and cost is a smaller claim than one
that also weighed capability, and pretending otherwise would be the
quiet kind of dishonesty.

Cost is supplied, never invented
---------------------------------
What a probe costs is a fact about the domain -- generating four thousand
chess games took an hour, measured. This module does not guess it. The
caller supplies a cost per probe, and a probe with no cost supplied is
reported as unpriced rather than assumed cheap.

It proposes an axis, not a value
---------------------------------
A probe names the condition to vary and the values already tried. Which
new value to use is a choice, and choosing one would be this module
deciding something it has not measured.

Not every condition is an axis
-------------------------------
Measured, the first time this ran: the top-ranked probe was "vary
arena_version". A component version is recorded so a comparison can be
trusted, not chosen so something can be learned -- and with information
1.0 (never varied) and a low price it beat every real axis.

    controllable  corpus_games, depth, feature_count, corpus_source
    contextual    eval_version, arena_version

The record does not distinguish them and cannot: which is which is a
judgement about the domain. So it is supplied by the caller. An axis
nobody declared controllable is still reported -- dropping it would let a
reader think it was never considered -- and cannot be chosen. When no
declaration is made at all, every axis stays selectable and the report
says plainly that none were declared, so the omission is visible instead
of silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .experiments import MIN_EXPERIMENT_VALUE, VALUE_WEIGHTS, select
from .series import Series

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

#: Information in an axis nobody has varied. The most that can be said
#: for an experiment: its outcome is entirely unknown.
NEVER_VARIED = 1.0

#: An axis implicated in a flip that could not be attributed. Isolating it
#: resolves an ambiguity that already exists in the record, which is worth
#: nearly as much as a fresh axis and is more likely to conclude.
IN_CONFOUNDED_FLIP = 0.9

#: An axis where a flip has already been isolated. A boundary is visible;
#: more points refine it rather than discovering it.
FLIP_FOUND = 0.3

#: An axis carrying a standing law whose limits were never tested. Above
#: FLIP_FOUND and below a fresh axis: overturning a claim already made is
#: worth more than refining a boundary and less than the first look at
#: something nobody has claimed anything about.
#:
#: This is what a PROPOSED law is licensed to do -- `laws.py` calls it
#: "suggestive, nothing more", and pointing at the next experiment is
#: exactly that. It may not justify a behaviour change, and nothing here
#: lets it.
LAW_AT_RISK = 0.8

#: How the cost term is scaled before the weight is applied. Mirrors
#: `experiments.plan`, so a probe and a pipeline experiment priced at the
#: same cost score the same penalty.
COST_SCALE = 500.0


def _axes_at_risk(book: Any, domain: str) -> Dict[str, str]:
    """Axes where a standing law has a limit nobody has tested.

    Scoped by domain: a law about chess says nothing about a series that
    is not chess, and letting it speak there is how a law gets applied
    where it was never measured. An empty domain matches nothing rather
    than everything, for the same reason.
    """
    from .laws import REFUTED
    from .lawgiver import axes_of

    if book is None or not domain:
        return {}
    out: Dict[str, str] = {}
    for law in book.all():
        if law.status == REFUTED or not law.exceptions:
            continue
        if law.condition.domain and law.condition.domain != domain:
            continue
        for axis in axes_of(law):
            out.setdefault(axis, law.exceptions[0])
    return out


@dataclass(frozen=True)
class Probe:
    """One axis worth varying next, priced if the caller priced it.

    `estimated_calls` and `value` are named to match `ExperimentPlan`, so
    the existing `experiments.select` ranks these without a second
    selector being written.
    """
    question: str
    condition: str
    tried_values: Tuple[Any, ...]
    information: float
    reason: str
    estimated_calls: int = 0
    priced: bool = False
    #: Whether this is something anybody sets on purpose. A contextual
    #: condition is reported and never chosen.
    controllable: bool = True

    @property
    def value(self) -> float:
        """Information and cost only. Capability is unmeasured here, and
        unmeasured is not zero -- so it is left out rather than counted
        as nothing."""
        cost_term = (VALUE_WEIGHTS["cost"]
                     * min(1.0, self.estimated_calls / COST_SCALE))
        return VALUE_WEIGHTS["information_gain"] * self.information + cost_term

    @property
    def worth_running(self) -> bool:
        return self.controllable and self.value >= MIN_EXPERIMENT_VALUE

    def as_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "condition": self.condition,
                "tried_values": list(self.tried_values),
                "information": round(self.information, 4),
                "value": round(self.value, 4),
                "estimated_calls": self.estimated_calls,
                "priced": self.priced, "controllable": self.controllable,
                "capability_gain": "не измерена — в ценность не входит",
                "reason": self.reason}

    def describe(self) -> str:
        tried = ", ".join(str(v) for v in self.tried_values) or "ничего"
        # No unit is printed: what a probe costs is the caller's fact --
        # seconds for chess, calls for a pipeline experiment -- and this
        # module inventing one would be inventing a measurement.
        price = (f"цена {self.estimated_calls}" if self.priced
                 else "цена не задана")
        head = ("варьировать" if self.controllable
                else "НЕ ось (контекстное условие)")
        return (f"{head} «{self.condition}» "
                f"(уже пробовали: {tried}); "
                f"информативность {self.information:.2f}, {price}, "
                f"ценность {self.value:+.3f} — {self.reason}")


def _distinct(series: Series, condition: str) -> Tuple[Any, ...]:
    seen: List[Any] = []
    for observation in series.observations:
        if condition in observation.conditions:
            value = observation.conditions[condition]
            if value not in seen:
                seen.append(value)
    return tuple(seen)


def probes(series: Series,
           cost: Optional[Callable[[str], int]] = None,
           controllable: Optional[Sequence[str]] = None,
           book: Any = None) -> List[Probe]:
    """Every axis in the record, with what is known about it.

    Includes axes that have been probed and found flat -- their low value
    is the point. Dropping them would hide that they were looked at, and
    a reader could not tell "not worth it" from "never considered". The
    same goes for contextual conditions: reported, marked, not chosen.
    """
    if not series.observations:
        return []

    confounded: Dict[str, bool] = {}
    for comparison in series.confounded_flips:
        for name in comparison.changed:
            confounded[name] = True
    flipped: Dict[str, bool] = {}
    for comparison in series.isolated_flips:
        for name in comparison.changed:
            flipped[name] = True

    at_risk = _axes_at_risk(book, series.domain)

    axes = sorted(set(series.varied) | set(series.constant) | set(series.partial))
    out: List[Probe] = []
    for condition in axes:
        values = _distinct(series, condition)
        if condition in at_risk:
            information = LAW_AT_RISK
            reason = (f"стоящий закон с непроверенным пределом: "
                      f"{at_risk[condition]}")
        elif flipped.get(condition):
            information = FLIP_FOUND
            reason = "переворот на этой оси уже изолирован"
        elif confounded.get(condition):
            information = IN_CONFOUNDED_FLIP
            reason = ("ось участвует в перевороте, который не удалось "
                      "приписать одному условию")
        elif len(values) <= 1:
            information = NEVER_VARIED
            reason = "ось ни разу не варьировалась"
        else:
            information = 1.0 / (1.0 + len(values))
            reason = (f"ось варьировалась ({len(values)} значения), "
                      f"переворота не найдено")

        estimated = 0
        priced = False
        if cost is not None:
            try:
                estimated = int(cost(condition))
                priced = True
            except Exception:
                estimated, priced = 0, False

        out.append(Probe(question=series.question, condition=condition,
                         tried_values=values, information=information,
                         reason=reason, estimated_calls=estimated,
                         priced=priced,
                         controllable=(controllable is None
                                       or condition in controllable)))
    return sorted(out, key=lambda p: p.value, reverse=True)


def choose(series: Series, budget: int,
           cost: Optional[Callable[[str], int]] = None,
           controllable: Optional[Sequence[str]] = None,
           book: Any = None) -> Optional[Probe]:
    """The probe worth running next, or None.

    Delegates to `experiments.select`, which applies the same
    `MIN_EXPERIMENT_VALUE` floor everything else is held to: running a
    worthless experiment because there is budget left is how a research
    loop converts compute into noise. None is a real answer.
    """
    return select([p for p in probes(series, cost, controllable, book)
                   if p.controllable], budget)


def report(series: Series,
           cost: Optional[Callable[[str], int]] = None,
           budget: int = 10 ** 9,
           controllable: Optional[Sequence[str]] = None,
           book: Any = None) -> Dict[str, Any]:
    """What is known per axis and what would be chosen, without choosing."""
    found = probes(series, cost, controllable, book)
    chosen = choose(series, budget, cost, controllable, book)
    notes = ["ценность считается из информативности и цены; прирост "
             "способности здесь не измеряется и в неё не входит"]
    if controllable is None:
        notes.append("управляемые оси не объявлены — выбирается из всех, "
                     "включая контекстные условия вроде версий")
    return {"question": series.question,
            "observations": len(series.observations),
            "controllable_declared": (None if controllable is None
                                      else sorted(controllable)),
            "probes": [p.as_dict() for p in found],
            "chosen": chosen.as_dict() if chosen else None,
            "notes": notes}
