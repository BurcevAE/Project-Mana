"""
mana.world.schema — the frame a world model is written in, not its contents.

Why a frame and not an encyclopedia
------------------------------------
Loading facts about Windows, 1C and the internet would give MANA an
encyclopedia: a lot of text about the world, and no way to tell a claim
that has been checked from one that was read somewhere. What is missing
underneath is smaller and more basic -- what it means for something to
exist, to have a state, to be changed by an action under conditions, and
to be observed at a time.

Five things, and everything else is built from them:

    ENTITY       something that exists, in some domain
    STATE        what is true of it, at a time, from a source
    ACTION       something that may change a state
    RULE         the conditions an action needs, and what it leaves behind
    OBSERVATION  what was actually seen, and when

Three ways of holding a claim, never collapsed
-----------------------------------------------
    OBSERVED   the sensors reported this, at this time
    KNOWN      established by a discriminating case: the world was seen
               to behave differently when this and only this differed
    BELIEVED   consistent with everything seen so far, and never
               discriminated -- so it may still be a coincidence
    UNKNOWN    not enough evidence to say either way

`mana/agent_parts/context.py` already marks recalled text VERIFIED /
EXPERIENCE / USER CLAIM / INFERENCE, which is the same distinction made
for one narrower purpose. These four are the general form, and the rule
that matters is that BELIEVED never silently becomes KNOWN.

A negative is never KNOWN
--------------------------
"I cannot do this" is not established by failing a hundred times: absence
of evidence is not evidence of absence, and this project has already been
bitten once by reading "no counterexamples in seven episodes" as safety.
So `CANNOT` is recorded as BELIEVED with the evidence beside it, however
many attempts stand behind it.

Memory is not the world
------------------------
Every state carries `observed_at`. A recorded state is an observation
that was true then; whether it is true now is a separate question with a
separate answer, and `StateFact.age` exists so a reader has to think
about it rather than treating the archive as the present.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

# ---------------------------------------------------------------- epistemics

#: The sensors reported it. Says nothing about now.
OBSERVED = "OBSERVED"
#: Established by a discriminating case -- the world behaved differently
#: when this and only this differed.
KNOWN = "KNOWN"
#: Consistent with everything seen, never discriminated. May be a
#: coincidence, and is written down as one.
BELIEVED = "BELIEVED"
#: Not enough evidence either way. A real answer, not a gap.
UNKNOWN = "UNKNOWN"

EPISTEMIC = (OBSERVED, KNOWN, BELIEVED, UNKNOWN)


# ---------------------------------------------------------------- the frame

@dataclass(frozen=True)
class Entity:
    """Something that exists, in some domain."""
    name: str
    domain: str = ""

    def __str__(self) -> str:
        return f"{self.domain}/{self.name}" if self.domain else self.name


@dataclass(frozen=True)
class StateFact:
    """One thing that was true of one entity, at one time, from one source.

    The timestamp is not decoration. A process that was running at 12:30
    may not be running now, and a model that stores "running" without when
    and from where cannot tell the difference between knowing that and
    having known it.
    """
    entity: str
    attribute: str
    value: Any
    observed_at: float = 0.0
    source: str = ""
    confidence: float = 1.0

    @property
    def key(self) -> Tuple[str, str]:
        return (self.entity, self.attribute)

    def age(self, now: Optional[float] = None) -> float:
        """Seconds since this was seen. The question a reader must ask."""
        return max(0.0, (now if now is not None else time.time()) - self.observed_at)

    def as_dict(self) -> Dict[str, Any]:
        return {"entity": self.entity, "attribute": self.attribute,
                "value": self.value, "observed_at": self.observed_at,
                "source": self.source, "confidence": self.confidence}

    def __str__(self) -> str:
        return f"{self.entity}.{self.attribute}={self.value!r}"


#: A state of the world as a set of (entity, attribute, value) triples.
#: Frozen so it can be compared, hashed and used as the "before" of a
#: transition without anybody editing history.
Situation = FrozenSet[Tuple[str, str, Any]]


def situation_of(facts: Iterable[StateFact]) -> Situation:
    return frozenset((f.entity, f.attribute, f.value) for f in facts)


@dataclass(frozen=True)
class Observation:
    """What was seen after something was attempted.

    `succeeded` is what the world reported, not what the actor concluded.
    A failure with no reason given is the normal case -- reasons are what
    the model is for.
    """
    action: str
    target: str
    succeeded: bool
    before: Situation
    after: Situation
    at: float = 0.0
    note: str = ""

    @property
    def changed(self) -> FrozenSet[Tuple[str, str, Any]]:
        """What is true after and was not true before."""
        return self.after - self.before


@dataclass(frozen=True)
class Rule:
    """What an action needs, and what it leaves behind.

    Preconditions and effects are held apart by how well each is
    established: a precondition confirmed by a discriminating failure is
    KNOWN, one that merely held every time something worked is BELIEVED,
    and both are reported rather than merged into a single confident list.
    """
    action: str
    preconditions: Tuple[Tuple[str, str, Any], ...] = ()
    effects: Tuple[Tuple[str, str, Any], ...] = ()
    #: Effects that follow only some of the time, with the share measured.
    #: A physical regularity lives here; a rule of the digital world
    #: normally does not.
    regularities: Tuple[Tuple[Tuple[str, str, Any], float], ...] = ()
    status: str = UNKNOWN
    evidence: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        pre = ", ".join(f"{e}.{a}={v!r}" for e, a, v in self.preconditions) or "—"
        eff = ", ".join(f"{e}.{a}={v!r}" for e, a, v in self.effects) or "—"
        line = f"[{self.status}] {self.action}: если {pre} → {eff}"
        for (e, a, v), share in self.regularities:
            line += f"; иногда {e}.{a}={v!r} ({share:.0%})"
        return line

    def as_dict(self) -> Dict[str, Any]:
        return {"action": self.action,
                "preconditions": [list(p) for p in self.preconditions],
                "effects": [list(e) for e in self.effects],
                "regularities": [[list(f), s] for f, s in self.regularities],
                "status": self.status, "evidence": dict(self.evidence)}


#: What an actor can do, cannot do, and has not found out.
CAN = "CAN"
CANNOT = "CANNOT"


@dataclass(frozen=True)
class Capability:
    """One thing the actor may or may not be able to do, and on what evidence.

    Not a list of functions. "I have onec_launch()" says nothing about the
    world; "in this world I can bring 1C into a running state, when these
    conditions hold, and here is how sure I am" is a claim that can be
    wrong, which is what makes it worth holding.
    """
    action: str
    verdict: str                     # CAN | CANNOT | UNKNOWN
    status: str = UNKNOWN            # how well the verdict is established
    attempts: int = 0
    successes: int = 0
    distinct_situations: int = 0
    note: str = ""

    def describe(self) -> str:
        return (f"[{self.status}] {self.verdict} {self.action} "
                f"({self.successes}/{self.attempts} удач в "
                f"{self.distinct_situations} различных состояниях)"
                + (f" — {self.note}" if self.note else ""))

    def as_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "verdict": self.verdict,
                "status": self.status, "attempts": self.attempts,
                "successes": self.successes,
                "distinct_situations": self.distinct_situations,
                "note": self.note}


@dataclass
class WorldModel:
    """What an actor has worked out about a world, and how sure it is.

    Deliberately not a knowledge base of facts about the world. It holds
    rules of transition, what the actor can do, and what it has not
    established -- the last of which is a first-class part of the model
    rather than everything the other two do not mention.
    """
    domains: Dict[str, List[str]] = field(default_factory=dict)
    rules: Dict[str, Rule] = field(default_factory=dict)
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    #: The last seen value of each fact, with when and from where.
    last_seen: Dict[Tuple[str, str], StateFact] = field(default_factory=dict)
    observations: int = 0

    def know(self, status: str) -> List[Rule]:
        return [r for r in self.rules.values() if r.status == status]

    def can(self) -> List[Capability]:
        return [c for c in self.capabilities.values() if c.verdict == CAN]

    def cannot(self) -> List[Capability]:
        return [c for c in self.capabilities.values() if c.verdict == CANNOT]

    def unknown(self) -> List[Capability]:
        return [c for c in self.capabilities.values() if c.verdict == UNKNOWN]

    def describe(self) -> str:
        lines = [f"наблюдений: {self.observations}",
                 f"доменов: {len(self.domains)}  правил: {len(self.rules)}"]
        for name in sorted(self.domains):
            lines.append(f"  {name}: " + ", ".join(sorted(self.domains[name])))
        for action in sorted(self.rules):
            lines.append("  " + self.rules[action].describe())
        for action in sorted(self.capabilities):
            lines.append("  " + self.capabilities[action].describe())
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {"domains": {k: sorted(v) for k, v in self.domains.items()},
                "rules": {k: r.as_dict() for k, r in self.rules.items()},
                "capabilities": {k: c.as_dict()
                                 for k, c in self.capabilities.items()},
                "observations": self.observations}
