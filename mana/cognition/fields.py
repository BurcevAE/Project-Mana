"""
mana.cognition.fields — making the description space open.

The audit's last unaddressed finding: `FIELD_LIBRARY` is eleven
extractors somebody wrote, and `propose_fields` can only pick from them.
MANA can say "my vocabulary is insufficient" -- and prove it, by counting
task pairs it describes identically that behaved differently -- and then
cannot do anything about it except choose the best of eleven. A twelfth
hand-written field does not change that; it makes the list twelve long.

Two generators here, and one idea deliberately dropped.

**Thresholds chosen from the collisions.** The library's fields bucket a
measurement at cutoffs an author picked: `length_band` splits at fixed
word counts that no observation suggested. A generated field takes a raw
measurement and splits it where the confusion actually is. The parameter
comes from the data, so the space is no longer a list -- it is every
cutoff the observations support.

**What a cheap mechanism can handle.** Since phase 15 there are brains
that answer exactly or refuse. "Does the arithmetic parser accept this?"
is a boolean nothing in the library expresses, it costs microseconds,
and it is the single most useful thing to know about a task in a system
whose whole design is routing to the cheapest sufficient mechanism.

**Dropped: product fields.** Combining two extractors into one axis adds
nothing, because `describe` already emits a tuple of every field -- so a
representation carrying f and g already distinguishes exactly what the
product of f and g would. Building it would have been work that changed
no measurement.

The trap this must not fall into
--------------------------------
A threshold chosen to separate the observed collisions will separate the
observed collisions. That is fitting, and it is the same failure the
gates exist to prevent one level up. So a proposal is derived on one
half of the observations and scored on the other, and a field that
separates only where it was fitted is reported as such rather than
offered.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .self_model import Observation

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.8"

#: Raw measurements, un-bucketed. The library's fields are these with
#: cutoffs already chosen; keeping the raw number is what lets a cutoff
#: be chosen from evidence instead.
Measure = Callable[[str], float]


def _words(text: str) -> float:
    return float(len((text or "").split()))


def _digits(text: str) -> float:
    return float(sum(c.isdigit() for c in text or ""))


def _numbers(text: str) -> float:
    import re
    return float(len(re.findall(r"\d+", text or "")))


def _max_magnitude(text: str) -> float:
    import re
    found = [int(n) for n in re.findall(r"\d+", text or "")]
    return float(max(found)) if found else 0.0


def _operators(text: str) -> float:
    return float(sum((text or "").count(op) for op in "+-*/%"))


def _depth(text: str) -> float:
    depth = best = 0
    for char in text or "":
        if char == "(":
            depth += 1
            best = max(best, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return float(best)


def _distinct_ratio(text: str) -> float:
    words = (text or "").lower().split()
    return len(set(words)) / len(words) if words else 0.0


def _lines(text: str) -> float:
    return float(len([l for l in (text or "").splitlines() if l.strip()]))


def _punctuation(text: str) -> float:
    return float(sum((text or "").count(c) for c in ".,;:!?«»\"'"))


MEASURES: Dict[str, Measure] = {
    "words": _words,
    "digits": _digits,
    "numbers": _numbers,
    "max_magnitude": _max_magnitude,
    "operators": _operators,
    "paren_depth": _depth,
    "distinct_ratio": _distinct_ratio,
    "lines": _lines,
    "punctuation": _punctuation,
}

#: Brains whose acceptance is worth knowing about a task. Asked through
#: `substrates.call`, which either answers exactly or refuses -- and the
#: refusal is the informative half here.
MECHANISM_BRAINS = ("arithmetic", "sequence-solver", "text-ops", "order-logic")


def mechanism_field(brain_id: str) -> Callable[[str], bool]:
    """A field answering "can this cheap mechanism handle the task?"

    Costs microseconds and consults no model. The answer is a fact about
    the task that no hand-written pattern in the library expresses, and
    it is exactly the fact a cascade needs.
    """
    def extract(text: str) -> bool:
        from .. import substrates
        try:
            substrates.call(brain_id, text or "")
            return True
        except Exception:
            # A refusal and a malfunction are both "cannot handle it" for
            # the purpose of describing a task. The distinction matters to
            # the router, not to the vocabulary.
            return False
    return extract


@dataclass(frozen=True)
class GeneratedField:
    """A field derived from evidence rather than written by hand."""
    name: str
    kind: str                      # threshold | mechanism
    source: str                    # measure or brain it came from
    threshold: Optional[float] = None
    #: Contradicting pairs it separated where it was DERIVED.
    separates_fitted: int = 0
    #: Contradicting pairs it separated on observations it never saw.
    separates_held_out: int = 0
    held_out_pairs: int = 0

    @property
    def generalises(self) -> bool:
        """Did it separate anything it was not fitted to?

        The whole question. A cutoff chosen to split the observed
        collisions will split the observed collisions; only the held-out
        half says whether it found something.
        """
        return self.held_out_pairs > 0 and self.separates_held_out > 0

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "source": self.source,
                "threshold": self.threshold,
                "separates_fitted": self.separates_fitted,
                "separates_held_out": self.separates_held_out,
                "held_out_pairs": self.held_out_pairs,
                "generalises": self.generalises}

    def describe(self) -> str:
        where = (f"на отложенных {self.separates_held_out}/{self.held_out_pairs}"
                 if self.held_out_pairs else "отложенных пар нет")
        return (f"{self.name}: разделяет {self.separates_fitted} пар там, где выведено; "
                f"{where}")


def extractor_for(generated: GeneratedField) -> Callable[[str], Any]:
    """The function this generated field describes tasks with."""
    if generated.kind == "mechanism":
        return mechanism_field(generated.source)
    measure = MEASURES[generated.source]
    threshold = float(generated.threshold or 0.0)
    return lambda text: measure(text) >= threshold


def _contradicting_pairs(observations: Sequence[Observation],
                         texts: Dict[str, str]) -> List[Tuple[Observation, Observation]]:
    """Pairs that behaved differently. These are what a field must split.

    Only pairs with opposite outcomes: two failures described identically
    are not a confusion, they are agreement.
    """
    rows = [(o, texts[o.task_id]) for o in observations if o.task_id in texts]
    right = [r for r in rows if r[0].correct]
    wrong = [r for r in rows if not r[0].correct]
    return [(a[0], b[0]) for a in right for b in wrong]


def _splits(extract: Callable[[str], Any], texts: Dict[str, str],
            pairs: Sequence[Tuple[Observation, Observation]]) -> int:
    separated = 0
    for a, b in pairs:
        try:
            if extract(texts[a.task_id]) != extract(texts[b.task_id]):
                separated += 1
        except Exception:                              # pragma: no cover
            continue
    return separated


def generate(observations: Sequence[Observation], texts: Dict[str, str],
             limit: int = 8) -> List[GeneratedField]:
    """Fields derived from these observations, each scored on held-out ones.

    The split is by task id order rather than at random, so the same
    observations always produce the same proposals -- a generator whose
    output moves between runs cannot be audited, and two runs disagreeing
    about what the vocabulary needs is worse than either answer.
    """
    usable = sorted((o for o in observations if o.task_id in texts),
                    key=lambda o: o.task_id)
    if len(usable) < 8:
        return []
    half = len(usable) // 2
    fit, held = usable[:half], usable[half:]
    fit_pairs = _contradicting_pairs(fit, texts)
    held_pairs = _contradicting_pairs(held, texts)
    if not fit_pairs:
        return []

    out: List[GeneratedField] = []

    for name, measure in MEASURES.items():
        values = sorted({measure(texts[o.task_id]) for o in fit})
        if len(values) < 2:
            continue
        # Cutoffs BETWEEN observed values: a threshold equal to a value
        # splits on ties in a way that depends on comparison direction
        # rather than on the data.
        cuts = [(low + high) / 2.0 for low, high in zip(values, values[1:])]
        best_cut, best_score = None, 0
        for cut in cuts:
            score = _splits(lambda t, c=cut, m=measure: m(t) >= c, texts, fit_pairs)
            if score > best_score:
                best_cut, best_score = cut, score
        if best_cut is None or best_score == 0:
            continue
        generated = GeneratedField(
            name=f"{name}>={best_cut:g}", kind="threshold", source=name,
            threshold=best_cut, separates_fitted=best_score,
            held_out_pairs=len(held_pairs))
        out.append(replace(generated, separates_held_out=_splits(
            extractor_for(generated), texts, held_pairs)))

    for brain_id in MECHANISM_BRAINS:
        extract = mechanism_field(brain_id)
        fitted = _splits(extract, texts, fit_pairs)
        if fitted == 0:
            continue
        out.append(GeneratedField(
            name=f"handled_by_{brain_id}", kind="mechanism", source=brain_id,
            separates_fitted=fitted,
            separates_held_out=_splits(extract, texts, held_pairs),
            held_out_pairs=len(held_pairs)))

    # Ranked by what they separated where they were NOT fitted. Ranking by
    # the fitted score would put the best-overfitted field first every
    # time, which is precisely the ordering that makes fitting invisible.
    out.sort(key=lambda g: (-g.separates_held_out, -g.separates_fitted, g.name))
    return out[:limit]
