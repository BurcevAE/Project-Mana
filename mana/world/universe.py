"""
mana.world.universe — a small world with layers, whose rules are hidden.

What this is for
-----------------
Not a simulation of the real machine, and not a toy to make a demo look
alive. It is an oracle: a world whose true structure is written down here
so that what an explorer reconstructs can be compared against it and
scored. Without a known truth, "MANA built a model of the world" is a
sentence nobody can check -- and this project has enough of those.

The same shape as the chess capability test: acquire a domain where the
right answer is known independently, measure against it, and report the
number that comes out even when it is not the number anyone wanted.

The layers, and why they are not decoration
--------------------------------------------
    PHYSICAL       a door, a metal rod
    DEVICE         power
    LOCAL_SYSTEM   a file, an application
    NETWORK        a link, a host, a service

Three things are true of this world that make it worth exploring rather
than reading:

  * a cross-layer precondition. Nothing in LOCAL_SYSTEM or NETWORK works
    with the power off, so an explorer that models each layer separately
    finds a rule that only holds sometimes;
  * an action with no actuator. `open_door` never succeeds, and the world
    reports it exactly as it reports a precondition failure -- so "I
    cannot do this" has to be concluded from evidence rather than read off
    an error message;
  * an effect that is not a rule. Heating the rod expands it four times
    out of five, and there is no observable condition separating the fifth
    case. A model that records that as a rule is wrong; one that records
    it as a regularity with a measured share is right.

The contract with the explorer
-------------------------------
`SmallWorld` exposes `observe()` and `act()` and nothing else. The truth
-- `TRUE_RULES`, `NEVER_POSSIBLE`, the domain map -- lives here for
`grade()` to use, and the explorer must never import it. A test asserts
that it does not, because the discipline is the whole experiment: a
learner that peeks scores perfectly and means nothing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .schema import (CAN, CANNOT, Capability, Observation, Rule, Situation,
                     StateFact, UNKNOWN, WorldModel, situation_of)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

PHYSICAL = "PHYSICAL"
DEVICE = "DEVICE"
LOCAL_SYSTEM = "LOCAL_SYSTEM"
NETWORK = "NETWORK"

#: Which entity lives in which layer. The explorer sees this in its
#: observations -- a filesystem sensor and a network sensor really are
#: distinguishable on a real machine, so telling them apart is not a hint,
#: it is the sensor being what it is.
DOMAIN_OF: Dict[str, str] = {
    "door": PHYSICAL, "rod": PHYSICAL,
    "power": DEVICE,
    "file": LOCAL_SYSTEM, "app": LOCAL_SYSTEM,
    "link": NETWORK, "host": NETWORK, "service": NETWORK,
}

#: The world as it starts. Everything an action can reach is false here,
#: so a working sequence has to be discovered rather than stumbled into on
#: the first move.
INITIAL: Dict[Tuple[str, str], Any] = {
    ("power", "on"): True,
    ("door", "open"): False,
    ("rod", "expanded"): False,
    ("file", "exists"): True,
    ("file", "writable"): False,
    ("file", "open"): False,
    ("file", "written"): False,
    ("app", "installed"): True,
    ("app", "running"): False,
    ("link", "up"): False,
    ("host", "reachable"): False,
    ("service", "up"): True,
    ("data", "received"): False,
}

#: THE TRUTH. Preconditions and effects of every action, as the world
#: really works. Read by `grade`, never by an explorer.
TRUE_RULES: Dict[str, Dict[str, Any]] = {
    "cut_power":    {"pre": {("power", "on"): True},
                     "eff": {("power", "on"): False}},
    "restore_power": {"pre": {("power", "on"): False},
                      "eff": {("power", "on"): True}},
    "open_file":    {"pre": {("power", "on"): True, ("file", "exists"): True},
                     "eff": {("file", "open"): True}},
    "grant_write":  {"pre": {("power", "on"): True},
                     "eff": {("file", "writable"): True}},
    "write_file":   {"pre": {("file", "open"): True, ("file", "writable"): True},
                     "eff": {("file", "written"): True}},
    "launch_app":   {"pre": {("power", "on"): True, ("app", "installed"): True},
                     "eff": {("app", "running"): True}},
    "enable_link":  {"pre": {("power", "on"): True},
                     "eff": {("link", "up"): True}},
    "connect":      {"pre": {("link", "up"): True},
                     "eff": {("host", "reachable"): True}},
    "request":      {"pre": {("host", "reachable"): True, ("service", "up"): True},
                     "eff": {("data", "received"): True}},
    "heat_rod":     {"pre": {},
                     "eff": {}, "sometimes": {("rod", "expanded"): 0.8}},
    "open_door":    {"pre": {}, "eff": {("door", "open"): True},
                     "no_actuator": True},
}

#: Actions the actor has no actuator for. They fail exactly the way a
#: precondition failure fails, and the world says nothing about why.
NEVER_POSSIBLE = tuple(sorted(
    name for name, rule in TRUE_RULES.items() if rule.get("no_actuator")))

ACTIONS = tuple(sorted(TRUE_RULES))


class SmallWorld:
    """The world an explorer is dropped into. Answers two questions.

    `observe()` says what is true now, with the layer each fact came from.
    `act()` attempts something and reports whether it worked -- and
    nothing about why, because "why" is the thing being learned.
    """

    #: What can be attempted here. Offered by the world rather than
    #: carried by the explorer: an actor that knew the action vocabulary
    #: from somewhere else would be reading the interface from outside
    #: the world it is meant to be discovering.
    actions = ACTIONS

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._state: Dict[Tuple[str, str], Any] = dict(INITIAL)
        self.clock = 0.0
        self.attempts = 0

    # ---------- sensing ----------

    def reset(self) -> None:
        """Back to the start. Episodes exist so an explorer that painted
        itself into a corner -- power off, nothing works -- can get out
        without that being mistaken for a discovery."""
        self._state = dict(INITIAL)

    def observe(self) -> List[StateFact]:
        self.clock += 1.0
        return [StateFact(entity=entity, attribute=attribute, value=value,
                          observed_at=self.clock,
                          source=DOMAIN_OF.get(entity, LOCAL_SYSTEM))
                for (entity, attribute), value in sorted(self._state.items())]

    def situation(self) -> Situation:
        return situation_of(self.observe())

    # ---------- acting ----------

    def act(self, action: str) -> Observation:
        """Attempt something. The report is what happened, not why.

        An unknown action, an action with no actuator and an action whose
        preconditions do not hold all come back the same way: it did not
        work, and the world looks as it did. Distinguishing them is the
        explorer's problem, which is the point.
        """
        before = self.situation()
        self.clock += 1.0
        self.attempts += 1
        rule = TRUE_RULES.get(action)

        succeeded = False
        if rule is not None and not rule.get("no_actuator"):
            if all(self._state.get(key) == value
                   for key, value in rule["pre"].items()):
                succeeded = True
                for key, value in rule["eff"].items():
                    self._state[key] = value
                for key, chance in (rule.get("sometimes") or {}).items():
                    if self._rng.random() < chance:
                        self._state[key] = True

        return Observation(action=action, target="", succeeded=succeeded,
                           before=before, after=self.situation(),
                           at=self.clock)


# --------------------------------------------------------------------------
# grading: what the explorer got right, and what it made up
# --------------------------------------------------------------------------

@dataclass
class Score:
    """How close a reconstructed model is to the world it came from.

    Precision and recall are kept separate for preconditions because they
    fail differently: a missing precondition makes a model that promises
    too much, and an invented one makes a model that refuses work it
    could do. Averaging them into one number would hide which.
    """
    rules_expected: int = 0
    preconditions_expected: int = 0
    preconditions_found: int = 0
    preconditions_invented: int = 0
    effects_expected: int = 0
    effects_found: int = 0
    effects_invented: int = 0
    capabilities_right: int = 0
    capabilities_wrong: int = 0
    capabilities_unknown: int = 0
    domains_right: int = 0
    domains_expected: int = 0
    regularity_found: bool = False
    regularity_as_rule: bool = False
    wrongly_certain: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def precondition_recall(self) -> float:
        return (self.preconditions_found / self.preconditions_expected
                if self.preconditions_expected else 0.0)

    @property
    def precondition_precision(self) -> float:
        claimed = self.preconditions_found + self.preconditions_invented
        return self.preconditions_found / claimed if claimed else 0.0

    @property
    def effect_recall(self) -> float:
        return (self.effects_found / self.effects_expected
                if self.effects_expected else 0.0)

    def describe(self) -> str:
        return "\n".join([
            f"условия: найдено {self.preconditions_found}/"
            f"{self.preconditions_expected} (полнота "
            f"{self.precondition_recall:.0%}), выдумано "
            f"{self.preconditions_invented} (точность "
            f"{self.precondition_precision:.0%})",
            f"эффекты: найдено {self.effects_found}/{self.effects_expected} "
            f"(полнота {self.effect_recall:.0%}), выдумано "
            f"{self.effects_invented}",
            f"возможности: верно {self.capabilities_right}, неверно "
            f"{self.capabilities_wrong}, не выяснено "
            f"{self.capabilities_unknown}",
            f"слои: {self.domains_right}/{self.domains_expected}",
            f"случайный эффект: замечен как закономерность "
            f"{'да' if self.regularity_found else 'нет'}; "
            f"записан как правило "
            f"{'ДА (ошибка)' if self.regularity_as_rule else 'нет'}",
            f"уверенных ошибок (KNOWN и неверно): "
            f"{len(self.wrongly_certain)}"
            + (": " + ", ".join(self.wrongly_certain) if self.wrongly_certain
               else ""),
        ])

    def as_dict(self) -> Dict[str, Any]:
        return {"precondition_recall": round(self.precondition_recall, 4),
                "precondition_precision": round(self.precondition_precision, 4),
                "effect_recall": round(self.effect_recall, 4),
                "capabilities_right": self.capabilities_right,
                "capabilities_wrong": self.capabilities_wrong,
                "capabilities_unknown": self.capabilities_unknown,
                "domains_right": self.domains_right,
                "domains_expected": self.domains_expected,
                "regularity_found": self.regularity_found,
                "regularity_as_rule": self.regularity_as_rule,
                "wrongly_certain": list(self.wrongly_certain),
                "detail": dict(self.detail)}


def grade(model: WorldModel) -> Score:
    """Compare a reconstructed model against the world as it really is.

    The strictest line here is `wrongly_certain`: a rule the explorer
    marked KNOWN and got wrong. A model may be incomplete without being
    dishonest, and the incompleteness is measured above -- but a confident
    falsehood is a different kind of failure and is counted on its own.
    """
    score = Score(rules_expected=len(TRUE_RULES))
    detail: Dict[str, Any] = {}

    for action, truth in TRUE_RULES.items():
        want_pre = set(
            (entity, attribute, value)
            for (entity, attribute), value in truth["pre"].items())
        want_eff = set(
            (entity, attribute, value)
            for (entity, attribute), value in truth["eff"].items())
        if truth.get("no_actuator"):
            # Effects that can never be produced are not something a model
            # is expected to have found.
            want_eff = set()

        got = model.rules.get(action)
        got_pre = set(got.preconditions) if got else set()
        got_eff = set(got.effects) if got else set()

        score.preconditions_expected += len(want_pre)
        score.preconditions_found += len(want_pre & got_pre)
        score.preconditions_invented += len(got_pre - want_pre)
        score.effects_expected += len(want_eff)
        score.effects_found += len(want_eff & got_eff)
        score.effects_invented += len(got_eff - want_eff)

        if got is not None and got.status == "KNOWN" and (got_pre - want_pre):
            score.wrongly_certain.append(action)

        if action == "heat_rod":
            sometimes = set(
                (entity, attribute, True)
                for (entity, attribute) in (truth.get("sometimes") or {}))
            if got is not None:
                score.regularity_found = bool(
                    sometimes & {fact for fact, _ in got.regularities})
                score.regularity_as_rule = bool(sometimes & got_eff)

        detail[action] = {"pre_want": sorted(map(str, want_pre)),
                          "pre_got": sorted(map(str, got_pre)),
                          "eff_want": sorted(map(str, want_eff)),
                          "eff_got": sorted(map(str, got_eff)),
                          "status": got.status if got else UNKNOWN}

    for action in ACTIONS:
        capability = model.capabilities.get(action)
        truly_possible = action not in NEVER_POSSIBLE
        if capability is None or capability.verdict == UNKNOWN:
            score.capabilities_unknown += 1
        elif (capability.verdict == CAN) == truly_possible:
            score.capabilities_right += 1
        else:
            score.capabilities_wrong += 1

    want_domains = {}
    for entity, domain in DOMAIN_OF.items():
        want_domains.setdefault(domain, set()).add(entity)
    score.domains_expected = len(want_domains)
    for domain, entities in want_domains.items():
        found = set(model.domains.get(domain, ()))
        if entities <= found:
            score.domains_right += 1

    score.detail = detail
    return score
