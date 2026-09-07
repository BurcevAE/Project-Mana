"""
mana.cognition.findings — what was tried, what came of it, and don't repeat it.

The gap, in the user's words
-----------------------------
"Провал эксперимента мы тоже не запишем, и этот тест уйдёт в никуда; если
захочу вернуться, начну сначала и потрачу время на повторение."

That was exactly right, and it is this project's recurring failure at a
new level: an apparatus was built to measure whether a change helps, the
measurement ran, and its **output went nowhere a machine could read**. The
chess experiment left a corpus and a weights file on disk; the conclusion
-- fitted evaluation 0.4708 against material over 240 games, 0.5 inside
the interval -- existed only in a chat message and a commit message.

Neither existing store fits, and forcing one would make both worse:

  * `core/transaction.py` is crash recovery. It answers "was a
    self-modification left half-done", not "did we already try this".
  * `cognition/exchange.py` carries genome mutations between installations
    and requires `mutation in MUTATIONS`. A chess evaluation is not a
    genome mutation, and widening that vocabulary to fit would blur a
    boundary that exists for signatures and Sybil resistance.

So this is a third store, deliberately, and its job is narrow.

A negative result is the valuable one
--------------------------------------
"We tried this and it did not work" saves more time than "this worked",
because the second is usually already visible in the behaviour and the
first is invisible by construction. `REJECTED` and `NOT_EVALUATED` are
first-class here, not failures to be tidied away -- the same reasoning as
the three-state verdict in `core/gates.py`.

The part that actually saves the time
--------------------------------------
Recording is half of it. `already_tried()` is the other half: asked
**before** an experiment starts, so a repeat is recognised as a repeat.
A ledger nobody consults is a diary, and this project has enough things
that were built and connected to nothing.

A finding is a prior, not a prohibition
----------------------------------------
Conditions travel with every record -- the version, the size and shape of
the evidence, what the comparison was against. A result measured on a
thousand games at 2.33.0 says something about a thousand games at 2.33.0.
When the conditions have moved, the finding is a reason to expect an
outcome, never a reason to refuse to look. `stale_against()` reports the
difference rather than deciding for the reader.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..core.gates import ACCEPTED, REJECTED, NOT_EVALUATED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The same three states as the acceptance gates, on purpose. "We tested
#: it and it did not hold" and "we could not test it" are different facts
#: here for the same reason they are different there.
VERDICTS = (ACCEPTED, REJECTED, NOT_EVALUATED)

#: Bound on the ledger. Findings are small and rare compared with turns,
#: so this is years of them; it exists so nothing grows without limit.
MAX_FINDINGS = 5000

LEDGER_DIRNAME = "findings"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Finding:
    """One experiment, its verdict, and what would make it stale.

    `question` and `approach` together give the id: two runs of the same
    approach to the same question are the same experiment, whoever ran
    them and whenever. An assigned id would make every repetition look
    like a new result, which is the failure this exists to prevent.
    """
    question: str
    approach: Dict[str, Any]
    verdict: str
    measurement: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    note: str = ""
    created: float = field(default_factory=time.time)
    version: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"неизвестный вердикт {self.verdict!r}; "
                             f"допустимы {VERDICTS}")
        if not str(self.question).strip():
            raise ValueError("у находки должен быть вопрос, на который она отвечает")

    @property
    def finding_id(self) -> str:
        body = _canonical({"question": self.question.strip().lower(),
                           "approach": self.approach})
        return hashlib.blake2b(body.encode("utf-8"), digest_size=8).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {"finding_id": self.finding_id, "question": self.question,
                "approach": self.approach, "verdict": self.verdict,
                "measurement": self.measurement, "conditions": self.conditions,
                "note": self.note, "created": self.created,
                "version": self.version}

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Finding":
        return cls(question=str(row.get("question") or ""),
                   approach=dict(row.get("approach") or {}),
                   verdict=str(row.get("verdict") or NOT_EVALUATED),
                   measurement=dict(row.get("measurement") or {}),
                   conditions=dict(row.get("conditions") or {}),
                   note=str(row.get("note") or ""),
                   created=float(row.get("created") or 0.0),
                   version=str(row.get("version") or ""))

    def stale_against(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """How the world has moved since this was measured.

        Reports the difference and decides nothing. A finding whose
        conditions have changed is a reason to expect an outcome, never a
        reason to refuse to look -- and a ledger that silently suppressed
        a re-run would be worse than no ledger, because it would hide the
        one case where re-running was the right call.
        """
        changed: Dict[str, Any] = {}
        for key, was in sorted(self.conditions.items()):
            now = conditions.get(key)
            if key in conditions and now != was:
                changed[key] = {"was": was, "now": now}
        missing = sorted(set(conditions) - set(self.conditions))
        return {"changed": changed, "not_recorded_then": missing,
                "stale": bool(changed or missing)}

    def describe(self) -> str:
        when = time.strftime("%d.%m.%Y", time.localtime(self.created))
        head = {ACCEPTED: "сработало", REJECTED: "не сработало",
                NOT_EVALUATED: "не удалось оценить"}[self.verdict]
        return (f"[{when}] {head}: {self.question}\n"
                f"  как: {_canonical(self.approach)}\n"
                f"  измерено: {_canonical(self.measurement)}")


def ledger_path() -> Path:
    from ..paths import data_root
    return Path(data_root()) / LEDGER_DIRNAME / "findings.jsonl"


class Ledger:
    """Findings on disk, appended, and looked up before work starts.

    Every write is wrapped: a ledger that cannot be written must not be
    the reason an experiment fails. It is a record about work, not part
    of the work.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else ledger_path()
        self._lock = threading.Lock()

    # ---------- writing ----------

    def record(self, finding: Finding) -> bool:
        """Append a finding. A later one about the same experiment wins.

        Kept as an append rather than a rewrite: the history of what was
        believed and when is itself worth having, and `latest` is the
        only reader that has to care.
        """
        line = json.dumps(finding.as_dict(), ensure_ascii=False)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                return True
            except Exception:
                return False

    # ---------- reading ----------

    def findings(self, limit: int = 0) -> List[Finding]:
        rows: List[Finding] = []
        try:
            handle = self.path.open("r", encoding="utf-8")
        except Exception:
            return rows
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(Finding.from_dict(json.loads(line)))
                except Exception:
                    continue
        rows = rows[-MAX_FINDINGS:]
        return rows[-limit:] if limit and limit > 0 else rows

    def latest(self) -> List[Finding]:
        """The newest finding per experiment, newest first."""
        newest: Dict[str, Finding] = {}
        for finding in self.findings():
            current = newest.get(finding.finding_id)
            if current is None or finding.created >= current.created:
                newest[finding.finding_id] = finding
        return sorted(newest.values(), key=lambda f: -f.created)

    def already_tried(self, question: str, approach: Dict[str, Any],
                      conditions: Optional[Dict[str, Any]] = None
                      ) -> Optional[Dict[str, Any]]:
        """Has exactly this been tried? The half that saves the time.

        Asked BEFORE an experiment runs. A ledger nobody consults is a
        diary.

        Returns the finding together with how the conditions have moved,
        so the caller can tell "we know the answer" from "we knew the
        answer under circumstances that no longer hold". It never says
        "do not run this".
        """
        probe = Finding(question=question, approach=approach,
                        verdict=NOT_EVALUATED)
        for finding in self.latest():
            if finding.finding_id == probe.finding_id:
                return {"finding": finding.as_dict(),
                        "describe": finding.describe(),
                        "staleness": finding.stale_against(conditions or {})}
        return None

    def about(self, question: str) -> List[Finding]:
        """Everything tried against this question, whatever the approach.

        The neighbouring answer to `already_tried`: "this exact thing, no
        -- but here are four other ways somebody went at it" is usually
        more useful than a bare miss.
        """
        wanted = question.strip().lower()
        return [f for f in self.latest() if f.question.strip().lower() == wanted]

    def stats(self) -> Dict[str, Any]:
        latest = self.latest()
        by_verdict: Dict[str, int] = {}
        for finding in latest:
            by_verdict[finding.verdict] = by_verdict.get(finding.verdict, 0) + 1
        return {"path": str(self.path), "exists": self.path.exists(),
                "experiments": len(latest),
                "records": len(self.findings()),
                "by_verdict": by_verdict,
                "questions": sorted({f.question for f in latest})}
