"""
mana.research.adapters.turns — the agent's own turns as a research world.

Step 4 of docs/RESEARCH_CONTRACT.md, first and conservative run: no probes,
no new research. A real turn is an observation --

    question   does capability X serve turns of this kind?  (plan.capability)
    situation  what the turn declared about itself, read off its result --
               the kind of turn, its route, whether and how its answer was
               checked -- as fixed yes/no conditions; nothing guessed
    outcome    served, exactly as phase 3 decides it: a turn with no gap
               (`cognition/gaps.from_turn`) was served

-- and after it `core/standing.py` draws the standing again: status,
boundary, and both assumptions it rests on. Nothing here acts on the
machine and nothing changes the answer.

Conservative on purpose:
  * rule `observed` -- only what was seen, which no interaction of
    conditions can break (step 3b);
  * repeatability checked -- a kind of turn is known only once its outcome
    repeated; real programs are not taken to be deterministic (step 3c);
  * never settled -- the next real turn may refute what the last ones
    showed, so the standing is drawn afresh every time, and while the
    explanations cannot yet be told apart it is OPEN.

The observations persist beside the findings ledger, one line per turn.
The journal of episodes carries neither plan nor verification, so turns
recorded before this cannot be replayed into it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core import standing
from .. import contract
from ..loop import (CONFIDENCE, OTHER, RIVAL_PRIOR, Explanations, condition_explanations,
                    pairwise_rivals)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The explanations cannot be told apart yet.
OPEN = "OPEN"

#: What a turn declares about itself. Fixed, so every situation has the same
#: conditions and a value never seen before maps onto them rather than
#: adding one.
CONDITIONS = ("ход: ответ", "ход: действие программе", "ход: запомнить",
              "маршрут: web", "маршрут: local",
              "проверка: есть", "проверка: арифметика")


def situation_from(conditions: Dict[str, bool]) -> Dict[str, Any]:
    conditions = {name: bool(conditions.get(name, False)) for name in CONDITIONS}
    return {"key": tuple(sorted(conditions.items())), "conditions": conditions}


def situation_of(result: Dict[str, Any]) -> Dict[str, Any]:
    """The situation a turn was in, from what its result says about it."""
    plan = result.get("plan") or {}
    kind = str(plan.get("kind") or "")
    route = str(plan.get("route") or "")
    check = str((result.get("verification") or {}).get("kind") or "none")
    return situation_from({
        "ход: ответ": kind == "answer",
        "ход: действие программе": kind == "app_action",
        "ход: запомнить": kind == "remember",
        "маршрут: web": route in ("web", "mixed"),
        "маршрут: local": route in ("local", "mixed"),
        "проверка: есть": check != "none",
        "проверка: арифметика": check == "arithmetic",
    })


class TurnStandings:
    """Observations of real turns, per capability, and how each stands."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self.stakes = contract.Stakes(error_cost=1.0, rule=standing.OBSERVED,
                                      assume_repeatable=False)
        self._challenge = pairwise_rivals(list(CONDITIONS))
        self._sets: Dict[str, Explanations] = {}
        self._known: Dict[str, List[Dict[str, Any]]] = {}
        if self.path is not None and self.path.exists():
            for row in self._rows():
                self._fold(str(row["capability"]), situation_from(row["conditions"]),
                           bool(row["served"]))

    def _rows(self) -> List[Dict[str, Any]]:
        out = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue            # a torn line is lost, not fatal
        return out

    def _set(self, capability: str) -> Explanations:
        e = self._sets.get(capability)
        if e is None:
            e = self._sets[capability] = condition_explanations(list(CONDITIONS))
            e.stakes = self.stakes
            e.subject = capability
            e.complete = False
            self._known[capability] = []
        return e

    def _fold(self, capability: str, situation: Dict[str, Any], served: bool) -> None:
        e = self._set(capability)
        known = self._known[capability]
        if all(p["key"] != situation["key"] for p in known):
            known.append(situation)
        e.set_space(known)
        e.note(situation, served, check=False)
        e.update(situation, served)

    def observe(self, capability: str, situation: Dict[str, Any], served: bool) -> None:
        """One real turn: which capability, in what situation, served or not."""
        self._fold(capability, situation, served)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"capability": capability,
                                         "conditions": situation["conditions"],
                                         "served": bool(served), "at": time.time()},
                                        ensure_ascii=False) + "\n")

    def explanations(self, capability: str) -> Optional[Explanations]:
        return self._sets.get(capability)

    def standing_of(self, capability: str) -> Dict[str, Any]:
        """How the capability stands now. Drawn afresh, never settled."""
        e = self._sets.get(capability)
        if e is None or not e.history:
            return {"capability": capability, "status": OPEN, "observations": 0}
        names, mass, lead = e.leading_class(e.space)
        if names != (OTHER,) and mass >= CONFIDENCE and lead.name not in e.challenged:
            # The same challenge a leader meets anywhere: rivals that agree
            # with it except in one combination of two conditions, weighed
            # against every turn seen. Nothing is acted on.
            e.challenged.add(lead.name)
            e.add(self._challenge(lead), prior_each=RIVAL_PRIOR * lead.prior)
            names, mass, lead = e.leading_class(e.space)
        if names == (OTHER,):
            answer = standing.unexplained(capability, observed=len(e.history))
        elif mass < CONFIDENCE:
            return {"capability": capability, "status": OPEN, "observations": len(e.history)}
        else:
            answer = e.answer_for(lead)
        row = answer.as_dict()
        row["capability"] = capability
        row["class"] = list(names)
        row["repeatability_meaning"] = standing.REPEATABILITY.get(answer.repeatability, "")
        row["observations"] = len(e.history)
        return row
