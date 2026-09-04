"""
mana.cognition.novelty — is this actually new, or a rediscovery wearing a
different name?

Why text comparison is not enough
---------------------------------
Two obvious failures, and they point in opposite directions:

  * **Same idea, different spelling.** A chain written as
    GENERATE→CRITIQUE→REPAIR and one composed into
    COUNTERFACTUAL_REFINEMENT→ANSWER are the same cognitive program. A
    comparison by name or by text calls them different, and the search
    then "discovers" its own previous discovery repeatedly, each time
    spending a full experiment budget.
  * **Same shape, different behaviour.** Two chains differing by one
    operator can succeed and fail on completely different tasks. A
    comparison by structure calls them nearly identical, and the search
    stops exploring a direction that was actually productive.

So novelty is measured on four channels, and behaviour dominates.

The channels
------------
  1. **Behaviour** (weight 0.5) -- do they get the same tasks right and
     wrong? Two programs with identical outcome vectors are the same
     program however they are written. This is the only channel that can
     see "same idea, different spelling", and it is why it weighs most.
  2. **Structure** (0.2) -- the operator chain after expansion, compared
     by edit distance. Expanded, so a composite and its chain are not
     mistaken for different things.
  3. **Failure profile** (0.2) -- *how* they fail, not just how often. Two
     programs at the same accuracy that fail on format versus on
     arithmetic are solving the task differently.
  4. **Resource profile** (0.1) -- calls and latency. Weakest, because a
     cheaper way of doing the same thing is a real difference but a small
     one.

Novelty as distance to the nearest thing already known
------------------------------------------------------
A candidate's novelty is its distance to its *closest* neighbour in the
archive, not the average. Averaging makes anything different from most of
the archive look novel even when it duplicates one member exactly, which
is the one case that matters.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

#: Declared, like the other weight sets in this package. Behaviour
#: dominates on purpose: it is the only channel that recognises the same
#: idea written two ways.
CHANNEL_WEIGHTS = {
    "behaviour": 0.5,
    "structure": 0.2,
    "failure_profile": 0.2,
    "resources": 0.1,
}

#: Below this distance a candidate is treated as already known. Not a
#: claim about cognition -- a threshold, and one that has to be exceeded
#: on the weighted combination, so being slightly different on every
#: channel is not enough.
NOVELTY_THRESHOLD = 0.25


@dataclass(frozen=True)
class Behaviour:
    """What a candidate did, in the form novelty can compare.

    `outcomes` is a per-task correctness vector over a fixed probe set --
    the same tasks in the same order for every candidate, or the vectors
    are not comparable and the whole channel is noise.
    """
    candidate_id: str
    steps: Tuple[str, ...]
    outcomes: Tuple[bool, ...]
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    mean_calls: float = 0.0
    mean_latency: float = 0.0
    label: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = list(self.steps)
        payload["outcomes"] = list(self.outcomes)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Behaviour":
        return cls(candidate_id=data["candidate_id"], steps=tuple(data.get("steps") or ()),
                   outcomes=tuple(bool(x) for x in data.get("outcomes") or ()),
                   failure_reasons=dict(data.get("failure_reasons") or {}),
                   mean_calls=float(data.get("mean_calls") or 0.0),
                   mean_latency=float(data.get("mean_latency") or 0.0),
                   label=data.get("label", ""))


# ---------------------------------------------------------------------------
# per-channel distances, each in [0, 1]
# ---------------------------------------------------------------------------

def behaviour_distance(a: Behaviour, b: Behaviour) -> Optional[float]:
    """Fraction of probe tasks where the two disagreed.

    Returns None when the vectors are not comparable rather than padding
    them to a common length: a padded comparison silently invents
    agreement on tasks one candidate never attempted, which is the exact
    direction that would make everything look identical.
    """
    n = min(len(a.outcomes), len(b.outcomes))
    if n == 0 or len(a.outcomes) != len(b.outcomes):
        return None
    return sum(1 for x, y in zip(a.outcomes, b.outcomes) if x != y) / n


def structure_distance(a: Behaviour, b: Behaviour) -> float:
    """Normalised edit distance over the expanded operator chains."""
    return _levenshtein(a.steps, b.steps) / max(1, max(len(a.steps), len(b.steps)))


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def failure_distance(a: Behaviour, b: Behaviour) -> float:
    """How differently they fail, as distance between the failure mixes.

    Two programs at the same accuracy that fail on format versus on
    arithmetic are solving the task differently, and accuracy alone
    cannot see it.
    """
    keys = set(a.failure_reasons) | set(b.failure_reasons)
    if not keys:
        return 0.0
    total_a = sum(a.failure_reasons.values()) or 1
    total_b = sum(b.failure_reasons.values()) or 1
    return sum(abs(a.failure_reasons.get(k, 0) / total_a -
                   b.failure_reasons.get(k, 0) / total_b) for k in keys) / 2.0


def resource_distance(a: Behaviour, b: Behaviour) -> float:
    """Relative difference in cost, saturating at 1."""
    def relative(x: float, y: float) -> float:
        denominator = max(x, y)
        return abs(x - y) / denominator if denominator > 0 else 0.0
    return min(1.0, 0.5 * relative(a.mean_calls, b.mean_calls) +
               0.5 * relative(a.mean_latency, b.mean_latency))


def distance(a: Behaviour, b: Behaviour) -> Dict[str, Any]:
    """Weighted distance with the per-channel breakdown kept.

    The breakdown is returned because "novel because it behaves
    differently" and "novel because it is written differently" are
    different findings, and only the first is worth an experiment.

    When behaviour cannot be compared, its weight is redistributed over
    the remaining channels rather than treated as zero distance --
    scoring an unmeasurable channel as "identical" would make every
    unmeasured candidate look like a duplicate.
    """
    channels: Dict[str, float] = {
        "structure": structure_distance(a, b),
        "failure_profile": failure_distance(a, b),
        "resources": resource_distance(a, b),
    }
    behaviour = behaviour_distance(a, b)
    if behaviour is not None:
        channels["behaviour"] = behaviour

    weight = sum(CHANNEL_WEIGHTS[k] for k in channels)
    total = sum(CHANNEL_WEIGHTS[k] * v for k, v in channels.items()) / weight
    return {"distance": total, "channels": channels,
            "behaviour_comparable": behaviour is not None}


# ---------------------------------------------------------------------------
# the archive
# ---------------------------------------------------------------------------

@dataclass
class NoveltyVerdict:
    candidate_id: str
    novelty: float
    is_novel: bool
    nearest: str = ""
    nearest_label: str = ""
    channels: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NoveltyArchive:
    """Everything already tried, so nothing is discovered twice.

    Holds behaviours, not verdicts. A candidate rejected as duplicate
    today may be the nearest neighbour that makes tomorrow's candidate
    look familiar, and keeping only the accepted ones would lose exactly
    that.
    """

    def __init__(self, behaviours: Optional[Sequence[Behaviour]] = None) -> None:
        self._items: List[Behaviour] = list(behaviours or ())

    def __len__(self) -> int:
        return len(self._items)

    def add(self, behaviour: Behaviour) -> None:
        self._items.append(behaviour)

    def all(self) -> List[Behaviour]:
        return list(self._items)

    def assess(self, candidate: Behaviour,
               threshold: float = NOVELTY_THRESHOLD) -> NoveltyVerdict:
        """Distance to the CLOSEST known thing, not the average.

        Averaging makes a candidate that duplicates one archive member
        exactly look novel as long as it differs from the rest -- and that
        is the one case this function exists to catch.
        """
        if not self._items:
            return NoveltyVerdict(candidate.candidate_id, 1.0, True, reason="архив пуст")

        best: Optional[Tuple[float, Behaviour, Dict[str, float]]] = None
        for known in self._items:
            if known.candidate_id == candidate.candidate_id:
                continue
            result = distance(candidate, known)
            if best is None or result["distance"] < best[0]:
                best = (result["distance"], known, result["channels"])
        if best is None:
            return NoveltyVerdict(candidate.candidate_id, 1.0, True, reason="архив пуст")

        novelty, nearest, channels = best
        is_novel = novelty >= threshold
        if is_novel:
            dominant = max(channels, key=lambda k: channels[k] * CHANNEL_WEIGHTS[k])
            reason = f"отличается прежде всего по каналу «{dominant}»"
        else:
            reason = f"близко к {nearest.label or nearest.candidate_id}"
        return NoveltyVerdict(candidate.candidate_id, novelty, is_novel,
                              nearest=nearest.candidate_id,
                              nearest_label=nearest.label,
                              channels=channels, reason=reason)

    def most_similar_pairs(self, limit: int = 3) -> List[Dict[str, Any]]:
        """The closest pairs in the archive -- where the search is
        repeating itself."""
        pairs: List[Dict[str, Any]] = []
        for i, a in enumerate(self._items):
            for b in self._items[i + 1:]:
                result = distance(a, b)
                pairs.append({"a": a.label or a.candidate_id,
                              "b": b.label or b.candidate_id,
                              "distance": round(result["distance"], 4)})
        pairs.sort(key=lambda p: p["distance"])
        return pairs[:limit]

    # ---------- persistence ----------

    def to_dict(self) -> Dict[str, Any]:
        return {"behaviours": [b.as_dict() for b in self._items]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoveltyArchive":
        return cls([Behaviour.from_dict(row) for row in (data.get("behaviours") or [])])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "NoveltyArchive":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return cls()


def behaviour_from_run(candidate_id: str, steps: Sequence[str], grades: Sequence[Any],
                       calls: Sequence[int] = (), latencies: Sequence[float] = (),
                       label: str = "") -> Behaviour:
    """Build a comparable behaviour from a run over the probe set.

    Takes grades from `core.oracle` rather than raw answers, so the
    outcome vector carries the same verdicts everything else in the system
    is measured by -- a novelty channel judged by a different standard
    from the acceptance gate would let a candidate be novel and useless at
    the same time without either noticing.
    """
    outcomes = tuple(bool(getattr(g, "correct", False)) for g in grades)
    failures: Dict[str, int] = {}
    for g in grades:
        if not getattr(g, "correct", False):
            reason = getattr(g, "reason", "") or "wrong"
            failures[reason] = failures.get(reason, 0) + 1
    return Behaviour(
        candidate_id=candidate_id, steps=tuple(steps), outcomes=outcomes,
        failure_reasons=failures,
        mean_calls=sum(calls) / len(calls) if calls else 0.0,
        mean_latency=sum(latencies) / len(latencies) if latencies else 0.0,
        label=label)
