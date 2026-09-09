"""
mana.cognition.chess_version — a causal verdict becomes a version of the player.

The contract, stated before the code
--------------------------------------
An agent that changes itself needs five things at once, and dropping any
one of them makes the other four worthless.

**1. Изменение — это данные, не правка исходника.** A version is an
ordered list of adopted changes; version zero is the player as written.
Nothing here edits `chess_arena`. The same rule `policy.adopt` follows,
and for the same reason: a change that lives in source cannot be reverted
by a running system and cannot be told apart from a change somebody made
by hand.

**2. Принятие требует свидетельства, названного заранее.** `REQUIRED`
below. A causal verdict of ACCEPTED, a lever with non-zero reach, enough
decided games, and both halves of the provenance -- the correlation that
suggested the change and the duel that tested it. Anything missing is a
refusal with the reason, never a quiet acceptance.

**3. Провенанс полон или принятия нет.** Every adoption carries the
observational finding, the causal finding, the reach, and the numbers the
duel produced. A change nobody can trace back to the measurement that
justified it is a change somebody made up, and this project has required
that trace since rules were first generated from diagnoses.

**4. Откат всегда возможен, и история не стирается.** `revert` removes
the last adoption from what is in force and appends the reason; the
record of what was believed and when survives. `latest` is the only
reader that has to care.

**5. Наблюдательный корпус не смешивает версии.** Every game is recorded
with the version that played it, and findings are computed inside a
version, never across. Games played by version 0 are evidence about
version 0. Without this the corpus becomes a blend of players and no
later measurement means anything -- the same rule that keeps worlds and
opponent levels apart, one layer up.

**6. Принятие условно, пока не подтверждено новым опытом.** An adoption
is `PROVISIONAL` until it is re-tested on games played *after* it, and
`CONFIRMED` only then. If the effect does not reproduce, it reverts
itself. That is the fresh split in time rather than in data: the duel
that justified the change was run before the change existed, and a result
that only holds on the evidence that suggested it is the oldest failure
in this file's ancestry.

What this module does not do
-----------------------------
It does not play, it does not choose what to adopt, and nothing calls it
yet. Wiring it into the bench means the ladder starts measuring a
different player, which resets what the ladder has measured -- a separate
decision, and not this file's to take.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.gates import ACCEPTED, MIN_PAIRED_TRIALS

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: What an adoption must carry. Named here so a caller can see what to
#: bring, and so a record missing one comes back refused rather than
#: force-fitted.
REQUIRED = ("change", "causal_finding", "observational_finding",
            "reach", "trials", "verdict")

#: An adoption is provisional until this many games played by the new
#: version have been measured against the version it replaced.
CONFIRM_GAMES = MIN_PAIRED_TRIALS

PROVISIONAL, CONFIRMED, REVERTED = "provisional", "confirmed", "reverted"

_lock = threading.Lock()
_cache: Optional[Dict[str, Any]] = None


class Refused(RuntimeError):
    """The evidence did not meet the contract. A state with a reason."""


def path() -> Path:
    from ..paths import resolve_data_path

    root = Path(resolve_data_path("lichess"))
    root.mkdir(parents=True, exist_ok=True)
    return root / "player_versions.json"


@dataclass
class Adoption:
    """One accepted change, and everything that justifies it."""
    property: str
    direction: int
    version: int
    state: str = PROVISIONAL
    causal_finding: str = ""
    observational_finding: str = ""
    reach: float = 0.0
    trials: int = 0
    effect: float = 0.0
    at: float = field(default_factory=time.time)
    note: str = ""

    def describe(self) -> str:
        way = "больше" if self.direction > 0 else "меньше"
        return (f"версия {self.version}: среди равных ходов выбирать тот, "
                f"у которого «{self.property}» {way} [{self.state}]")

    def as_dict(self) -> Dict[str, Any]:
        return {"property": self.property, "direction": self.direction,
                "version": self.version, "state": self.state,
                "causal_finding": self.causal_finding,
                "observational_finding": self.observational_finding,
                "reach": self.reach, "trials": self.trials,
                "effect": self.effect, "at": self.at, "note": self.note}

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Adoption":
        return cls(property=str(row.get("property", "")),
                   direction=int(row.get("direction", 1)),
                   version=int(row.get("version", 0)),
                   state=str(row.get("state", PROVISIONAL)),
                   causal_finding=str(row.get("causal_finding", "")),
                   observational_finding=str(row.get("observational_finding", "")),
                   reach=float(row.get("reach", 0.0)),
                   trials=int(row.get("trials", 0)),
                   effect=float(row.get("effect", 0.0)),
                   at=float(row.get("at", 0.0)),
                   note=str(row.get("note", "")))


def _load() -> Dict[str, Any]:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        try:
            _cache = json.loads(path().read_text(encoding="utf-8"))
        except Exception:
            _cache = {"history": []}
        _cache.setdefault("history", [])
        return _cache


def _save(state: Dict[str, Any]) -> None:
    global _cache
    with _lock:
        _cache = state
        path().write_text(json.dumps(state, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def _reset_for_tests() -> None:
    global _cache
    with _lock:
        _cache = None


def history() -> List[Adoption]:
    """Everything ever adopted, in order, including what was reverted."""
    return [Adoption.from_dict(row) for row in _load().get("history", [])]


def in_force() -> List[Adoption]:
    """The changes the player is made of now, oldest first."""
    return [row for row in history() if row.state != REVERTED]


def version() -> int:
    """How many changes are in force. Zero is the player as written."""
    return len(in_force())


def fingerprint() -> str:
    """A name for this composition, so two versions cannot be confused.

    The number alone is not enough: reverting one change and adopting
    another leaves the count the same and the player different.
    """
    import hashlib

    body = "|".join(f"{row.property}{row.direction:+d}" for row in in_force())
    digest = hashlib.blake2b(body.encode("utf-8"), digest_size=4).hexdigest()
    return f"v{version()}-{digest}" if body else "v0-base"


def check(evidence: Dict[str, Any]) -> str:
    """Why this may not be adopted, or "" when it may.

    Every clause is the contract, and each is checked rather than
    assumed. A refusal names the clause: "не хватает свидетельства" with
    nothing after it is a message nobody can act on.
    """
    # Presence, not truthiness. A reach of zero is a measured fact and a
    # reason to refuse; calling it "missing" hides the reason behind the
    # wrong message, which is the same confusion as reading an unmeasured
    # value as a zero.
    given = evidence or {}
    missing = [key for key in REQUIRED
               if key not in given or given[key] is None]
    # Numbers are checked for presence, identities also for content: a
    # reach of zero is a measured fact worth refusing on its own terms,
    # while an empty provenance is no provenance at all.
    missing += [key for key in ("change", "causal_finding",
                                "observational_finding", "verdict")
                if key not in missing and not given.get(key)]
    if missing:
        return f"нет обязательного: {', '.join(missing)}"
    if str(evidence["verdict"]) != ACCEPTED:
        return (f"вердикт причинного опыта {evidence['verdict']}, "
                f"а принимать можно только {ACCEPTED}")
    if float(evidence["reach"]) <= 0.0:
        return "рычаг ничего не двигает — опыт был не о находке, а о жребии"
    if int(evidence["trials"]) < MIN_PAIRED_TRIALS:
        return (f"решённых партий {evidence['trials']}, нужно "
                f"{MIN_PAIRED_TRIALS}")
    change = evidence["change"]
    already = [row for row in in_force()
               if row.property == getattr(change, "property", "")
               and row.direction == getattr(change, "direction", 0)]
    if already:
        return "это изменение уже в силе"
    return ""


def adopt(evidence: Dict[str, Any]) -> Adoption:
    """Make the change part of the player, as data, provisionally.

    Provisional on purpose: the duel that justified it ran before the
    change existed, and a result that holds only on the evidence which
    suggested it is the oldest failure this project guards against.
    """
    refused = check(evidence)
    if refused:
        raise Refused(refused)
    change = evidence["change"]
    state = _load()
    adoption = Adoption(
        property=str(change.property), direction=int(change.direction),
        version=version() + 1, state=PROVISIONAL,
        causal_finding=str(evidence["causal_finding"]),
        observational_finding=str(evidence["observational_finding"]),
        reach=float(evidence["reach"]), trials=int(evidence["trials"]),
        effect=float(evidence.get("effect", 0.0)),
        note=str(evidence.get("note", "")))
    state["history"] = list(state.get("history", [])) + [adoption.as_dict()]
    _save(state)
    return adoption


def confirm(adoption: Adoption, trials: int, effect: float,
            finding_id: str = "") -> Adoption:
    """Mark an adoption confirmed on experience gathered after it.

    Refuses to confirm on less than a full re-test: an adoption that
    calls itself confirmed on four games is worse than one that stays
    provisional, because the word stops meaning anything.
    """
    if int(trials) < CONFIRM_GAMES:
        raise Refused(f"подтверждать на {trials} партиях нельзя, "
                      f"нужно {CONFIRM_GAMES}")
    return _set_state(adoption, CONFIRMED,
                      f"подтверждено на {trials} новых партиях, "
                      f"эффект {effect:.0%}" + (f", {finding_id}" if finding_id else ""))


def revert(adoption: Adoption, reason: str) -> Adoption:
    """Take a change out of force, keeping the record of it.

    Nothing is deleted. "What was believed and when" is itself worth
    having, and a history that erases its mistakes cannot be used to
    check whether they repeat.
    """
    return _set_state(adoption, REVERTED, reason)


def _set_state(adoption: Adoption, state: str, note: str) -> Adoption:
    whole = _load()
    rows = list(whole.get("history", []))
    for row in rows:
        if (row.get("version") == adoption.version
                and row.get("property") == adoption.property):
            row["state"] = state
            row["note"] = note
            whole["history"] = rows
            _save(whole)
            return Adoption.from_dict(row)
    raise Refused("такой адопции в истории нет")


def player(base: Any) -> Any:
    """The base player with every change in force, in the order adopted.

    Composed rather than rebuilt: the version is the base plus a list, so
    reverting one entry gives back exactly what was there before it.
    """
    from . import chess_action

    out = base
    for row in in_force():
        out = chess_action.Tuned(out, chess_action.Change(
            property=row.property, direction=row.direction,
            from_finding=row.causal_finding))
    return out


def describe() -> str:
    rows = history()
    if not rows:
        return "версия 0: игрок как написан, изменений нет"
    lines = [f"состав: {fingerprint()}"]
    for row in rows:
        lines.append(f"  {row.describe()}"
                     + (f" — {row.note}" if row.note else ""))
    return "\n".join(lines)
