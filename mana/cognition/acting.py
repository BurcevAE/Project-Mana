"""
mana.cognition.acting — where a law is allowed to change what is done.

The gap this closes
--------------------
`laws.py` derives a status from evidence, `lawgiver.py` puts claims in the
book, `probes.py` lets a standing claim lift the axis it is about. All of
that decides which experiment to run next. Nothing decided anything else:
`LawBook.applicable()` was called from two tests and from no code at all,
which is this project's oldest failure -- a module that exists and is
consumed by nothing.

So this is the one place a law may change a decision, and it is small on
purpose. A caller asks what to set an axis to; the answer is the default
unless a standing law speaks about that axis in that domain.

What may change behaviour, and what may not
--------------------------------------------
Only SUPPORTED and VALIDATED. A PROPOSED law is licensed to point at the
next experiment and nothing more -- that licence is stated in `probes.py`
and this is the other half of it. PROPOSED means one supported result and
no hidden confirmation; acting on it would be acting on a coincidence that
has not yet had the chance to be one.

REFUTED changes nothing and is kept. "We tried this and it did not hold"
is worth having, and a book that deleted its failures would relearn them.

Scope is checked, not assumed
------------------------------
A law carries the domain it was found in, and it speaks only there. A
claim measured on chess says nothing about how long to explore a world,
and letting it speak would be the way a law gets applied where it was
never measured -- the commonest way a folk theory survives contact with
data. An empty domain matches nothing rather than everything, the same
rule `probes._axes_at_risk` already applies.

Both directions are guidance
-----------------------------
A law that says an intervention raises the outcome recommends its `now`
value. A law that says it lowers the outcome recommends staying at `was`.
The second is the more useful half in practice: "beyond here, more of this
buys nothing" is a saving, and it is exactly what a series of diminishing
returns produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .laws import CognitiveLaw, LawBook, SUPPORTED, VALIDATED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Statuses a decision may rest on. PROPOSED is deliberately absent.
STANDING = (SUPPORTED, VALIDATED)


@dataclass(frozen=True)
class Decision:
    """What to do, and what made it that rather than the default."""
    axis: str
    value: Any
    default: Any
    #: Empty when nothing spoke. A reader can then tell "no law applies"
    #: from "a law applies and agrees with the default".
    law_id: str = ""
    status: str = ""
    why: str = ""

    @property
    def changed(self) -> bool:
        return self.law_id != "" and self.value != self.default

    def describe(self) -> str:
        if not self.law_id:
            return f"{self.axis}={self.value!r} (по умолчанию, законов нет)"
        return (f"{self.axis}={self.value!r} вместо {self.default!r} — "
                f"[{self.status}] {self.why}")

    def as_dict(self) -> Dict[str, Any]:
        return {"axis": self.axis, "value": self.value, "default": self.default,
                "law_id": self.law_id, "status": self.status,
                "changed": self.changed, "why": self.why}


def in_scope(law: CognitiveLaw, conditions: Optional[Dict[str, Any]]) -> bool:
    """Are the caller's conditions the ones this was measured under?

    A condition the law held and the caller states differently puts the
    caller outside the evidence. A condition the caller does not mention
    is not a conflict -- it is a caller that has not said, and refusing
    there would make every law unusable by anyone who did not enumerate
    the whole world.
    """
    stated = dict(conditions or {})
    for key, was in (law.scope or {}).items():
        if key in stated and stated[key] != was:
            return False
    return True


def standing_laws(book: Optional[LawBook], domain: str,
                  conditions: Optional[Dict[str, Any]] = None
                  ) -> List[CognitiveLaw]:
    """Laws that have earned the right to change a decision here.

    Not `LawBook.applicable`, which answers "does this claim cover this
    situation" and includes PROPOSED. Two different questions, and merging
    them would let a claim with no hidden confirmation decide something.
    """
    if book is None or not domain:
        return []
    return [law for law in book.all()
            if law.status in STANDING
            and law.condition.domain
            and law.condition.domain == domain
            and in_scope(law, conditions)]


def _speaks_about(law: CognitiveLaw, axis: str) -> Optional[Tuple[Any, Any]]:
    """The (was, now) this law claims about that axis, if it claims one."""
    from .lawgiver import INTERVENTION_FORMAT, axis_of

    separator = INTERVENTION_FORMAT.split("{was}")[0].split("{axis}")[1]
    for intervention in law.intervention:
        if axis_of(intervention) != axis:
            continue
        _, _, rest = intervention.partition(separator)
        was, arrow, now = rest.partition(" -> ")
        if not arrow:
            continue
        return (was.strip(), now.strip())
    return None


def _same_kind(value: str, like: Any) -> Any:
    """Read a value back in the type the caller uses.

    Interventions are written as text because that is what a law book is;
    a caller asking about `steps` wants a number back, and handing it the
    string "2500" would make the comparison against its default silently
    false.
    """
    if isinstance(like, bool):
        return value.strip().lower() in {"true", "1", "yes", "да"}
    if isinstance(like, int):
        try:
            return int(float(value))
        except Exception:
            return value
    if isinstance(like, float):
        try:
            return float(value)
        except Exception:
            return value
    return value


def setting_for(axis: str, default: Any, domain: str,
                book: Optional[LawBook] = None,
                conditions: Optional[Dict[str, Any]] = None) -> Decision:
    """What to set this axis to, given what the book has established.

    The default, unless a standing law in this domain, measured under
    conditions the caller has not contradicted, speaks about this axis.
    Nothing here decides whether a law is true -- that is settled by
    evidence in `laws.py` and cannot be influenced from here.
    """
    for law in standing_laws(book, domain, conditions):
        pair = _speaks_about(law, axis)
        if pair is None:
            continue
        was, now = pair
        raises = law.evidence.mean_effect > 0
        chosen = _same_kind(now if raises else was, default)
        direction = "поднимает исход" if raises else "опускает исход"
        return Decision(
            axis=axis, value=chosen, default=default, law_id=law.law_id,
            status=law.status,
            why=(f"закон: {axis} {was!r} → {now!r} {direction} "
                 f"(эффект {law.evidence.mean_effect:+.3f} по "
                 f"{law.evidence.trials} испытаниям)"))
    return Decision(axis=axis, value=default, default=default)
