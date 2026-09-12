"""
mana.research.contract — the shapes the research loop works with.

    Explanation  one possible answer: a function from a situation to a
                 distribution over outcomes. An explanation that predicts no
                 observation is not admitted -- it could never be told apart
                 from another by acting.
    Probe        an action the world offers: the situation it brings about,
                 its cost, and what it may do to the world
    World        which probes exist for a question, and acting on one
    Stakes       what rides on the answer in the task at hand
    Question     what is unknown, why it is open, the competing explanations

A situation is a dict with `key` (naming it) and `conditions` (what holds in
it, by name). An explanation's claims are defined in `core/standing.py` by
the conditions it reads.

Complete and incomplete worlds
-------------------------------
Where every situation of the world can be listed and brought about -- the
synthetic device -- what an explanation reads is read off its predictions
over them. Where the world shows its situations only by being acted in --
`world/universe.py`, a real program -- the situations known so far are
sparse, and reading "does it turn on c" off them would miss conditions that
no two known situations happen to differ in alone: the boundary would then
reach where the explanation's claims were never true. So there an
explanation declares what it reads (`Explanation.reads`), and one that does
not is taken to read everything -- it then reaches no further than what was
seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from ..core import standing

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.2"

# What a probe may do to the world. A field of the probe, never a decision
# of the chooser: on a person's machine only the first is allowed without
# asking.
READ_ONLY = "read_only"
REVERSIBLE = "reversible"
IRREVERSIBLE = "irreversible"

# Why a probe was taken.
DISTINGUISH = "distinguish"     # which explanation in play is right
CHECK = "check"                 # how far the one left reaches, for the task

#: The absence of an explanation: predicts every outcome equally.
OTHER = "не объясняется ни одной из гипотез"
ALWAYS = "всегда"
NEVER = "никогда"


@dataclass(eq=False)
class Explanation:
    """One possible answer, and what it says a probe would show."""
    name: str
    predict: Callable[[Dict[str, Any]], Dict[Any, float]]
    weight: float = 1.0
    prior: float = 0.0
    #: The conditions its prediction may turn on, when it can say. Needed
    #: in an incomplete world; see the module docstring.
    reads: Optional[Tuple[str, ...]] = None


@dataclass
class Probe:
    """An action the world offers, with what it will make hold and its price.
    `params["key"]` names the situation; `params["conditions"]` is what holds."""
    action: str
    params: Dict[str, Any]
    cost: int = 1
    safety: str = REVERSIBLE


class World(Protocol):
    """Which probes exist for a question, and acting on one.

    Optional, for a world that shows its situations only by being acted in:
    `complete = False`, and `situations(question)` -- the situations known
    so far, which only grow. Without them the probes on offer are taken to
    be every situation there is. `act` may replace `probe.params` with the
    situation actually reached when that differs from the one planned: the
    record is what happened."""
    def probes_for(self, question: "Question") -> List[Probe]: ...
    def act(self, probe: Probe) -> Any: ...


@dataclass
class Stakes:
    """What rides on a question's answer in the task at hand.

    `error_cost`  what a wrong answer costs, in the units of probe cost
    `weight`      how much the task relies on each situation
    `rule`        how far the task accepts an answer beyond what was seen --
                  `observed` or `claims`; `pattern` is refused, because it
                  was measured putting errors inside its own boundary
    `assume_repeatable`
                  whether an outcome seen once in a situation is taken to be
                  the outcome there. False -- it is a claim of the answer,
                  tested by repeating the observation, and a situation is
                  known only once its outcome repeated
    """
    error_cost: float
    weight: Callable[[Dict[str, Any]], float] = field(default=lambda params: 1.0)
    rule: str = standing.CLAIMS
    assume_repeatable: bool = True

    def __post_init__(self) -> None:
        if self.rule not in (standing.OBSERVED, standing.CLAIMS):
            raise ValueError(f"inclusion rule {self.rule!r} is not offered to a task: "
                             f"use {standing.OBSERVED!r} or {standing.CLAIMS!r}")


@dataclass
class Question:
    """What MANA wants to know, why it is still open, and the explanations
    competing to answer it."""
    subject: str
    why_open: str
    explanations: Any
    source: Dict[str, Any] = field(default_factory=dict)
