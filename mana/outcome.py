"""
mana.outcome — what a tool did, measured against what it was asked to do.

The gap this closes
-------------------
Every tool in this project reports one thing: whether the call raised.
`ToolResult(ok=True)` is written by hand in the wrapper, before anything
has looked at the machine, and `onec_launch.launch` returned

    {"launched": True, "base": base, "mode": ..., "pid": process.pid}

on the line after `Popen`. `base` there is the base that was *asked for*,
echoed back as though it were news; `pid` is a process that may already
have exited. An answer built from that dict says "запустила базу X"
whether 1С opened X, opened the base-selection window instead, or died in
a third of a second on a connection string it did not like.

A failure has to be definable before it can be found, and the definition
used here is: **a failure is a difference between the state an action was
supposed to produce and the state that is actually there afterwards.**
That needs three things recorded where one was recorded before:

    goal      what the person asked for, in their own terms
    expected  the state this action was supposed to leave behind
    observed  the state found on the machine after it ran

`verified` is derived from the last two and can never be set by a caller,
the same way a law's status is derived from its evidence rather than
announced.

Three verdicts, because two would lie
--------------------------------------
CONFIRMED, CONTRADICTED, UNOBSERVED. The third one is the point. An axis
nobody could look at is not an axis that came out right, and folding
"did not check" into "fine" is precisely the defect this module exists to
remove -- it would be strange to rebuild it inside the tool meant to
detect it. This is the same three-state shape the gates already use for
ACCEPTED / REJECTED / NOT_EVALUATED, for the same reason.

Absence of evidence is kept asymmetric, deliberately
-----------------------------------------------------
A key counts as observed only when it is present in `observed`. A key
that could not be read is simply left out, lands in `unobserved`, and
never becomes a mismatch. A 1С window whose title does not yet name the
base is not evidence that the wrong base opened -- it is a window that
has not finished opening. So an observation can only ever *fire* on
evidence, and silence is reported as silence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Everything expected was looked at, and matched.
CONFIRMED = "confirmed"
#: Something expected was looked at, and was not what was expected. This
#: is the only verdict that asserts a failure.
CONTRADICTED = "contradicted"
#: Something expected could not be looked at. Not a failure, and not a
#: success either -- the honest third answer.
UNOBSERVED = "unobserved"

VERDICTS = (CONFIRMED, CONTRADICTED, UNOBSERVED)


def _same(want: Any, got: Any) -> bool:
    """Equality as a person would judge it, for values read off a machine.

    Strings compare stripped and case-folded, because an infobase name
    comes back from a window title the way 1С chose to print it, and
    reporting "УТ11-ER" against "ут11-er " as a failure would be a bug in
    the detector rather than a finding about the world.
    """
    if isinstance(want, str) and isinstance(got, str):
        return want.strip().casefold() == got.strip().casefold()
    return want == got


@dataclass(frozen=True)
class Outcome:
    """One action, and what became of it.

    Frozen because a verdict that can be edited after the fact is not
    evidence. To revise one, observe again and build another.
    """

    #: The tool or operation, as the registry knows it: "onec_launch".
    action: str
    #: What the person asked for, in their words. Kept beside the
    #: machine-level expectation because "запусти конфигуратор УТ" and
    #: {"mode": "конфигуратор"} are not the same statement, and a reader
    #: comparing them later is exactly how a wrong translation is caught.
    goal: str = ""
    #: The state this action was supposed to produce.
    expected: Dict[str, Any] = field(default_factory=dict)
    #: The state actually found afterwards. Keys absent here were not
    #: observed; see the module docstring on why that is not a mismatch.
    observed: Dict[str, Any] = field(default_factory=dict)
    #: What the observation was taken from -- a pid, an exit code, the raw
    #: window title. Never compared; kept so that a verdict can be argued
    #: with, and so the next real run teaches us the format we guessed at.
    evidence: Dict[str, Any] = field(default_factory=dict)
    #: Why an axis could not be observed, when there is something to say.
    note: str = ""

    # ---------- derived, never stored ----------

    @property
    def compared(self) -> List[str]:
        """Axes that were expected and could be looked at."""
        return [k for k in self.expected if k in self.observed]

    @property
    def unobserved(self) -> List[str]:
        """Axes that were expected and could not be looked at."""
        return [k for k in self.expected if k not in self.observed]

    def delta(self) -> Dict[str, Dict[str, Any]]:
        """Where expectation and reality disagree. This is the failure."""
        return {k: {"expected": self.expected[k], "observed": self.observed[k]}
                for k in self.compared
                if not _same(self.expected[k], self.observed[k])}

    @property
    def verified(self) -> str:
        """The verdict, derived from the evidence and nothing else.

        Order matters: a contradiction on one axis is a failure whatever
        happened on the others, and a single unobserved axis is enough to
        stop the whole thing being called confirmed.
        """
        if self.delta():
            return CONTRADICTED
        if self.unobserved or not self.compared:
            return UNOBSERVED
        return CONFIRMED

    def succeeded(self) -> bool:
        return self.verified == CONFIRMED

    def failed(self) -> bool:
        return self.verified == CONTRADICTED

    # ---------- reading it back ----------

    def summary(self) -> str:
        """One line a person can act on, in the language they asked in.

        Written to be said out loud in an answer: the difference between
        "открыла" and "запустила, но не подтвердила" is the whole reason
        this module exists, and burying it in a dict nobody renders would
        waste the observation.
        """
        if self.failed():
            parts = [
                "{}: ожидалось {!r}, на деле {!r}".format(
                    key, diff["expected"], diff["observed"])
                for key, diff in self.delta().items()]
            return "не то, что просили — " + "; ".join(parts)
        if self.verified == UNOBSERVED:
            missing = ", ".join(self.unobserved) or "ничего не проверено"
            tail = " ({})".format(self.note) if self.note else ""
            return "выполнено, но не подтверждено: {}{}".format(missing, tail)
        return "сделано и проверено: " + (self.goal or self.action)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # The derived parts travel with it. A reader of the journal file
        # has no Outcome object to ask, and recomputing the verdict from a
        # JSON line is how two readers end up disagreeing about it.
        d["verified"] = self.verified
        d["delta"] = self.delta()
        d["unobserved"] = self.unobserved
        d["summary"] = self.summary()
        return d

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Outcome":
        """Rebuild from a journal line, ignoring the derived fields.

        `verified` in the file records what was concluded then; the object
        recomputes it from the same evidence, so a change in the rule
        shows up as a disagreement instead of being carried forward
        unnoticed.
        """
        row = row or {}
        return cls(action=str(row.get("action") or ""),
                   goal=str(row.get("goal") or ""),
                   expected=dict(row.get("expected") or {}),
                   observed=dict(row.get("observed") or {}),
                   evidence=dict(row.get("evidence") or {}),
                   note=str(row.get("note") or ""))


def unobserved(action: str, goal: str = "", note: str = "") -> Outcome:
    """An action nobody looked at afterwards.

    For tools that do not observe yet. Saying so costs one line and is
    true; the alternative is the hand-written `ok=True` this module exists
    to replace.
    """
    return Outcome(action=action, goal=goal, note=note)
