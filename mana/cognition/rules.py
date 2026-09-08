"""
mana.cognition.rules — rules MANA wrote, kept as data with their evidence.

Why not a knob
---------------
A knob is a change somebody thought of, wrote down, and gave two options.
Searching them is choosing among changes MANA did not invent. What this
module holds is the other kind: a rule constructed from a diagnosed
failure, which did not exist in the source before the failure was seen.

The shape is deliberately the smallest thing that can be a rule and not a
parameter: a literal that must appear in the text, the class it decides,
and precedence over the built-in markers. Small because it has to be
inspectable -- a person reading `--rules` must be able to disagree with a
specific claim about specific words -- and because a richer shape is a
search space, and a search space with a language model in it will find
something that wins by accident.

Precedence is the whole point of one of them
---------------------------------------------
The failure that produced the first of these was a conflict between two
surface markers that were both present: "Сколько раз буква «и»
встречается в тексте" carries a maths word and a text word, and the maths
word was consulted first. A rule that fires before the built-in list is
how "when features co-occur, this one decides" becomes something the
program does rather than something a comment says.

Evidence or nothing
--------------------
`install` refuses without evidence naming the experiment, the fresh split
and the verdict, exactly as `policy.adopt` does. A rule MANA wrote about
its own behaviour is a stronger thing than a setting, so the bar is not
lower. `remove` takes one off and the ledger keeps what it was.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: What an installed rule must name before it may be written.
REQUIRED_EVIDENCE = ("experiment", "fresh", "verdict")


class InstallRefused(RuntimeError):
    """A rule without a measurement behind it."""


@dataclass(frozen=True)
class Rule:
    """One literal, one class, and where it sits relative to the built-ins.

    `where` is a domain name, so a rule found while looking at task
    naming cannot start deciding something else. Scope is checked, never
    assumed -- the same rule `acting.py` applies to a law.
    """
    where: str
    marker: str
    decides: str
    before_builtin: bool = True
    #: How the marker did on the half it was mined from. Kept for a
    #: reader, never consulted by the matcher: a rule that scored well is
    #: still just a rule.
    lift: float = 0.0
    support: float = 0.0

    def matches(self, lowered: str) -> bool:
        return self.marker in lowered

    def describe(self) -> str:
        when = "перед встроенными" if self.before_builtin else "после встроенных"
        return (f"«{self.marker}» → {self.decides} ({when}; на половине "
                f"открытия: покрытие {self.support:.0%}, вне класса "
                f"{self.lift:.0%})")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_lock = threading.RLock()
_cache: Optional[Dict[str, Any]] = None


def _path() -> Path:
    from ..paths import resolve_data_path
    return Path(resolve_data_path("policy/rules.json"))


def _load() -> Dict[str, Any]:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        try:
            _cache = json.loads(_path().read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
        return _cache


def installed(where: str = "") -> List[Rule]:
    """Rules in force, optionally for one place only."""
    rows = (_load().get("rules") or [])
    out: List[Rule] = []
    for row in rows:
        try:
            rule = Rule(**{k: v for k, v in row.items()
                           if k in Rule.__dataclass_fields__})
        except Exception:
            continue
        if not where or rule.where == where:
            out.append(rule)
    return out


def evidence_for(marker: str) -> Dict[str, Any]:
    for row in (_load().get("rules") or []):
        if row.get("marker") == marker:
            return dict(row.get("evidence") or {})
    return {}


def install(rule: Rule, evidence: Dict[str, Any]) -> Rule:
    """Put a rule in force, with the measurement that earned it."""
    missing = [key for key in REQUIRED_EVIDENCE if not (evidence or {}).get(key)]
    if missing:
        raise InstallRefused(
            "правило без свидетельства: не названо " + ", ".join(missing))
    global _cache
    with _lock:
        stored = dict(_load())
        rows = [row for row in (stored.get("rules") or [])
                if not (row.get("marker") == rule.marker
                        and row.get("where") == rule.where)]
        rows.append(dict(rule.as_dict(), evidence=dict(evidence),
                         at=time.time()))
        stored["rules"] = rows
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        _cache = stored
    return rule


def remove(marker: str = "", where: str = "") -> int:
    """Take rules off. Everything with no argument."""
    global _cache
    with _lock:
        stored = dict(_load())
        rows = list(stored.get("rules") or [])
        kept = [row for row in rows
                if (marker and row.get("marker") != marker)
                or (where and row.get("where") != where)]
        if not marker and not where:
            kept = []
        stored["rules"] = kept
        try:
            path = _path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(stored, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        except Exception:
            pass
        _cache = stored
        return len(rows) - len(kept)


@dataclass
class Proposed:
    """A rule not yet installed, carried through an experiment."""
    rule: Rule

    def __enter__(self) -> "Proposed":
        _push(self.rule)
        return self

    def __exit__(self, *exc: Any) -> None:
        _pop()


_pending = threading.local()


def _push(rule: Rule) -> None:
    stack = getattr(_pending, "stack", None) or []
    stack.append(rule)
    _pending.stack = stack


def _pop() -> None:
    stack = getattr(_pending, "stack", None) or []
    if stack:
        stack.pop()
    _pending.stack = stack


def in_force(where: str = "") -> List[Rule]:
    """Installed rules plus whatever this thread is trying out.

    A candidate under test wins over an installed rule for the same
    reason an experiment's policy beats an adoption: the arm being
    measured has to be the arm that answers.
    """
    trying = [r for r in (getattr(_pending, "stack", None) or [])
              if not where or r.where == where]
    return trying + installed(where)


def _reset_for_tests() -> None:
    global _cache
    with _lock:
        _cache = None
    _pending.stack = []
