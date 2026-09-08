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

from ..core.gates import (ACCEPTED, REJECTED, NOT_EVALUATED,
                          MIN_PAIRED_TRIALS)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.2"

#: The same three states as the acceptance gates, on purpose. "We tested
#: it and it did not hold" and "we could not test it" are different facts
#: here for the same reason they are different there.
VERDICTS = (ACCEPTED, REJECTED, NOT_EVALUATED)

#: Bound on the ledger. Findings are small and rare compared with turns,
#: so this is years of them; it exists so nothing grows without limit.
MAX_FINDINGS = 5000

LEDGER_DIRNAME = "findings"

# --------------------------------------------------------------------------
# why it failed, derived rather than asserted
#
# `"type": "NO_GENERALIZATION"` written onto the chess result would have
# been an interpretation recorded as a fact. Reasoning built on labels a
# system assigns itself is reasoning built on nothing, and this is exactly
# the place where that matters: a later experiment would take the label as
# a premise.
#
# So a class is COMPUTED from the measurement by a rule that names itself,
# and computed on READ rather than stored -- a stored label can be edited,
# or can drift out of agreement with the numbers printed beside it.
# --------------------------------------------------------------------------

#: The measurement did not carry what the rules need. Said plainly rather
#: than guessed: "we cannot classify this" is a fact, and inventing a
#: class for it is the failure this whole design avoids.
UNCLASSIFIED = "UNCLASSIFIED"

#: Too few trials for the interval to mean anything.
NOT_MEASURED = "NOT_MEASURED"

#: The interval lies entirely below the no-effect value.
WORSE = "WORSE"

#: The interval contains the no-effect value. The chess result is this
#: one: 0.5 inside [0.376, 0.533].
NOT_BETTER = "NOT_BETTER"

#: Better, and the extra cost eats the gain by the rule below.
COSTS_MORE_THAN_IT_GAINS = "COSTS_MORE_THAN_IT_GAINS"

#: Better, and worth it.
BETTER = "BETTER"

FAILURE_CLASSES = (UNCLASSIFIED, NOT_MEASURED, WORSE, NOT_BETTER,
                   COSTS_MORE_THAN_IT_GAINS, BETTER)

#: What `measurement` must carry for a class to be derivable. Named here
#: so a caller can see what to record, and so an old record missing them
#: comes back UNCLASSIFIED instead of being force-fitted.
#:
#:   trials    independent observations (games, paired turns -- not
#:             positions, and not anything correlated)
#:   interval  [low, high] on the effect, at the stated confidence
#:   null      the value that means "no effect": 0.5 for a match score,
#:             0.0 for a margin
#:   cost_ratio  optional. candidate cost / baseline cost, in real units
REQUIRED_MEASUREMENT = ("trials", "interval", "null")

#: How much cost a win has to justify. A candidate that is better by a
#: hair and costs six times as much is not an improvement anybody can
#: spend; the chess evaluation cost 6.2x per position. Stated as a rule so
#: it can be argued with from evidence rather than taste.
COST_TOLERANCE = 1.5


@dataclass(frozen=True)
class Classification:
    """A failure class, the rule that produced it, and what it read."""
    failure: str
    rule: str
    inputs: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"failure": self.failure, "rule": self.rule,
                "inputs": dict(self.inputs)}


def classify(verdict: str, measurement: Dict[str, Any]) -> Classification:
    """Which class the numbers put this in. No opinion enters.

    Every branch names the rule it applied, so a reader can disagree with
    the rule instead of with the label.
    """
    missing = [k for k in REQUIRED_MEASUREMENT if k not in (measurement or {})]
    if missing:
        return Classification(
            UNCLASSIFIED, f"в измерении нет {', '.join(missing)}",
            {"has": sorted(measurement or {})})

    try:
        trials = int(measurement["trials"])
        low, high = (float(x) for x in measurement["interval"])
        null = float(measurement["null"])
    except Exception as exc:
        return Classification(UNCLASSIFIED,
                              f"измерение нечитаемо: {type(exc).__name__}",
                              {"measurement": measurement})

    read = {"trials": trials, "interval": [low, high], "null": null}

    if verdict == NOT_EVALUATED or trials < MIN_PAIRED_TRIALS:
        return Classification(
            NOT_MEASURED,
            f"испытаний {trials} против порога {MIN_PAIRED_TRIALS}", read)

    if high < null:
        return Classification(WORSE, "интервал целиком ниже нуля эффекта", read)

    if low <= null <= high:
        return Classification(NOT_BETTER,
                              "интервал накрывает ноль эффекта", read)

    ratio = measurement.get("cost_ratio")
    if ratio is not None:
        try:
            ratio = float(ratio)
        except Exception:
            ratio = None
    if ratio is not None:
        read["cost_ratio"] = ratio
        if ratio > COST_TOLERANCE:
            return Classification(
                COSTS_MORE_THAN_IT_GAINS,
                f"выигрыш есть, но цена {ratio}x превышает допуск "
                f"{COST_TOLERANCE}x", read)

    return Classification(BETTER, "интервал целиком выше нуля эффекта", read)


def measurement_of(trials: int, interval: Sequence[float], null: float,
                   cost_ratio: Optional[float] = None,
                   **extra: Any) -> Dict[str, Any]:
    """Build a measurement a class can be derived from.

    A helper rather than a convention, because a convention is a thing
    people follow until they are busy.
    """
    row: Dict[str, Any] = {"trials": int(trials),
                           "interval": [float(interval[0]), float(interval[1])],
                           "null": float(null)}
    if cost_ratio is not None:
        row["cost_ratio"] = float(cost_ratio)
    row.update(extra)
    return row



def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def staleness(then: Dict[str, Any], now: Dict[str, Any]) -> Dict[str, Any]:
    """Whether a result measured under `then` still speaks to `now`.

    The one rule, so that every caller asks the same question. A
    condition recorded then and different now has moved; a condition that
    matters now and was never recorded then is worse, because nobody can
    say whether it held -- both count as stale, and the report says which
    of the two it was.
    """
    then, now = dict(then or {}), dict(now or {})
    changed: Dict[str, Any] = {}
    for key, was in sorted(then.items()):
        if key in now and now[key] != was:
            changed[key] = {"was": was, "now": now[key]}
    missing = sorted(set(now) - set(then))
    return {"changed": changed, "not_recorded_then": missing,
            "stale": bool(changed or missing)}


@dataclass(frozen=True)
class Finding:
    """One experiment, its verdict, and what would make it stale.

    Identity is `question` + `approach` + **conditions**. Conditions are
    part of it because a series is "the same approach under different
    conditions": with them outside, corpus=1000 and corpus=5000 collapsed
    to one id, `latest()` kept only the newer, and the series reader
    compared the new point against a stale record while reporting two
    observations. Found by running it.

    `approach_id` -- question and approach alone -- is what groups a
    series and what a lookup asks about. An assigned id would make every
    repetition look like a new result, which is the failure this exists
    to prevent.
    """
    question: str
    approach: Dict[str, Any]
    verdict: str
    measurement: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    #: Guesses about WHY, and nothing else. Kept apart from `failure`,
    #: which is derived, because a guess may seed the next experiment and
    #: may never be a premise in a conclusion. Nothing branches on this;
    #: a test enforces it.
    suspected: Tuple[str, ...] = ()
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
    def failure(self) -> Classification:
        """Derived on read, so it cannot drift from the numbers."""
        return classify(self.verdict, self.measurement)

    @property
    def approach_id(self) -> str:
        """One way of going at one question, across all conditions."""
        body = _canonical({"question": self.question.strip().lower(),
                           "approach": self.approach})
        return hashlib.blake2b(body.encode("utf-8"), digest_size=8).hexdigest()

    @property
    def finding_id(self) -> str:
        """One run: this approach, under these conditions."""
        body = _canonical({"approach_id": self.approach_id,
                           "conditions": self.conditions})
        return hashlib.blake2b(body.encode("utf-8"), digest_size=8).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {"finding_id": self.finding_id,
                "approach_id": self.approach_id, "question": self.question,
                "approach": self.approach, "verdict": self.verdict,
                "measurement": self.measurement, "conditions": self.conditions,
                "failure": self.failure.as_dict(),
                "suspected": list(self.suspected),
                "note": self.note, "created": self.created,
                "version": self.version}

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Finding":
        # `failure` is deliberately NOT read back. It is derived, and
        # accepting it from the file would let a hand-edited record carry
        # a class its numbers do not support.
        return cls(question=str(row.get("question") or ""),
                   approach=dict(row.get("approach") or {}),
                   verdict=str(row.get("verdict") or NOT_EVALUATED),
                   measurement=dict(row.get("measurement") or {}),
                   conditions=dict(row.get("conditions") or {}),
                   suspected=tuple(row.get("suspected") or ()),
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
        return staleness(self.conditions, conditions)

    def describe(self) -> str:
        when = time.strftime("%d.%m.%Y", time.localtime(self.created))
        head = {ACCEPTED: "сработало", REJECTED: "не сработало",
                NOT_EVALUATED: "не удалось оценить"}[self.verdict]
        classified = self.failure
        lines = [f"[{when}] {head}: {self.question}",
                 f"  как: {_canonical(self.approach)}",
                 f"  класс: {classified.failure} — {classified.rule}",
                 f"  измерено: {_canonical(self.measurement)}"]
        if self.suspected:
            lines.append("  предполагаемая причина (догадка, не измерение): "
                         + "; ".join(self.suspected))
        return "\n".join(lines)


def ledger_path() -> Path:
    """Where the ledger lives when nobody says otherwise.

    Kept in step with `Config.findings_path`, which is what an agent
    actually passes; this default is for readers that show the record
    without constructing one.
    """
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
                        verdict=NOT_EVALUATED, conditions=dict(conditions or {}))
        same_approach = [f for f in self.latest()
                         if f.approach_id == probe.approach_id]
        if not same_approach:
            return None

        # An exact condition match first: that is "we have run precisely
        # this". Anything else on the same approach is "we have run this
        # approach, elsewhere in the condition space", which is a
        # different and weaker thing -- and saying so is the point.
        for finding in same_approach:
            if finding.finding_id == probe.finding_id:
                return {"finding": finding.as_dict(),
                        "describe": finding.describe(),
                        "match": "exact",
                        "staleness": finding.stale_against(conditions or {})}

        nearest = max(same_approach, key=lambda f: f.created)
        return {"finding": nearest.as_dict(), "describe": nearest.describe(),
                "match": "same_approach_other_conditions",
                "other_points": len(same_approach),
                "staleness": nearest.stale_against(conditions or {})}

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
