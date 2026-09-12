"""
mana.research.adapters.universe — `world/universe.SmallWorld` as a research
world.

A world whose situations cannot be brought about directly. Its only
actuators are its actions and `reset()`; which situations exist is not known
in advance and is learned by acting. So:

  * the situations known for a question are the ones the world has shown so
    far. They only grow, and the loop is told the world is not complete, so
    explanations declare what they read (`contract.py`);
  * a probe is an action attempted in a known situation: in the current one
    (cost 1), or after `reset()` and a replay of the shortest action path
    that reached it before (cost: the reset, the path, the attempt);
  * a replay through an effect that happens only sometimes can land
    elsewhere. The probe is then corrected to the situation actually reached
    before the attempt -- the record is what happened, not what was planned.

The questions are whether each action works, and -- named by the task, not
found here -- what some actions leave a fact at. Nothing here reads how the
world really works; the audit does that, in tests and scripts only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...world.universe import SmallWorld
from .. import contract
from ..loop import Explanations, condition_explanations, pairwise_rivals

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

SUCCESS = "успех"
EFFECT = "эффект"


def fact_name(entity: str, attribute: str) -> str:
    return f"{entity}.{attribute}"


def params_of(situation: Any) -> Dict[str, Any]:
    """A `schema.Situation` as a research situation."""
    conditions = {fact_name(e, a): bool(v) for e, a, v in situation}
    return {"key": tuple(sorted(conditions.items())), "conditions": conditions}


def success_subject(action: str) -> str:
    return f"{SUCCESS}: {action}"


def effect_subject(action: str, fact: str) -> str:
    return f"{EFFECT}: {action} → {fact}"


class UniverseWorld:
    """The small world as a `World`: attempt an action where you are, or go
    back to the start and walk a path you have walked before."""

    complete = False

    def __init__(self, world: SmallWorld, effects: Sequence[Tuple[str, str]] = ()) -> None:
        self.world = world
        #: subject -> (action, fact asked about, or None for "does it work")
        self.questions: Dict[str, Tuple[str, Optional[str]]] = {
            success_subject(a): (a, None) for a in world.actions}
        for action, fact in effects:
            self.questions[effect_subject(action, fact)] = (action, fact)
        self.known: List[Dict[str, Any]] = []
        self._index: Dict[Any, int] = {}
        self.paths: Dict[Any, List[str]] = {}
        self.resets = 0
        self.landed_elsewhere = 0
        start = params_of(world.situation())
        self._home = start["key"]
        self._learn(start, [])
        self.current = start["key"]

    @property
    def fact_names(self) -> List[str]:
        return sorted(self.known[0]["conditions"])

    def _learn(self, params: Dict[str, Any], path: List[str]) -> Dict[str, Any]:
        key = params["key"]
        if key not in self._index:
            self._index[key] = len(self.known)
            self.known.append(params)
        if key not in self.paths or len(path) < len(self.paths[key]):
            self.paths[key] = list(path)
        return self.known[self._index[key]]

    def situations(self, question: contract.Question) -> List[Dict[str, Any]]:
        return list(self.known)

    def probes_for(self, question: contract.Question) -> List[contract.Probe]:
        out = []
        for params in self.known:
            key = params["key"]
            cost = 1 if key == self.current else 2 + len(self.paths[key])
            out.append(contract.Probe(action=question.subject, params=params, cost=cost,
                                      safety=contract.REVERSIBLE))
        return out

    def _step(self, action: str) -> Any:
        before = self.current
        observation = self.world.act(action)
        after = params_of(observation.after)
        path = self.paths[before] + [action] if observation.succeeded else self.paths[before]
        self._learn(after, path)
        self.current = after["key"]
        return observation

    def act(self, probe: contract.Probe) -> bool:
        action, fact = self.questions[probe.action]
        target = probe.params["key"]
        if target != self.current:
            self.world.reset()
            self.resets += 1
            self.current = self._learn(params_of(self.world.situation()), [])["key"]
            for step in self.paths[target]:
                self._step(step)
        if self.current != target:
            # The replay went through something that happens only sometimes
            # and landed elsewhere: record where the attempt was really made.
            self.landed_elsewhere += 1
            probe.params = self.known[self._index[self.current]]
        observation = self._step(action)
        if fact is None:
            return bool(observation.succeeded)
        return params_of(observation.after)["conditions"][fact]

    @property
    def actions_taken(self) -> int:
        return self.world.attempts + self.resets


def knowledge_for(world: UniverseWorld) -> Dict[str, Explanations]:
    """For every question, every explanation over the facts equal."""
    return {subject: condition_explanations(world.fact_names) for subject in world.questions}


def challenge_for(world: UniverseWorld) -> Callable[[contract.Explanation], List[contract.Explanation]]:
    return pairwise_rivals(world.fact_names)
