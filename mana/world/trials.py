"""
mana.world.trials — three splits, and a fresh one that cannot be peeked at.

Why this is not a list of seeds in a script
--------------------------------------------
The last experiment kept its split in a module-level tuple and its
discipline in my head: discovery 1..40, validation 101..140, and a
promise not to look at the second until the strategy was fixed. The
promise held, and a promise is not a mechanism. `mana/core/splits.py`
solved the same problem years of this project ago and its answer is the
one used here: **a hidden set is never returned as data.** There is no
`fresh_seeds()`. The only way to interact with it is to hand in a
strategy and get back numbers.

Three roles, because two were not enough
-----------------------------------------
    DISCOVERY    visible, read as often as you like: this is where a
                 grid is searched and a hypothesis is formed
    VALIDATION   scored, never returned, small budget: does the change
                 survive contact with seeds it was not chosen on
    FRESH        scored once, and only after the strategy is sealed

The third exists because one good validation can be luck. A strategy that
was adjusted after seeing a validation score has been fitted to it, and
the only way to tell that from a real effect is a split that did not
exist for the adjuster -- which is what sealing enforces: after `seal`,
the strategy's identity is fixed and `fresh_score` will answer for that
identity once and refuse for any other.

Budget, not honour system
--------------------------
Every read is counted, capped and written to an audit log, the same
reasoning `core/splits.py` gives: unlimited access to a hidden set is the
same leak as reading it, only slower. Enough queries and a searcher fits
the set through the score alone. Exceeding the budget raises rather than
degrading quietly, and `audit()` shows what was read and when, so a claim
about a holdout can be checked instead of believed.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

DISCOVERY = "discovery"
VALIDATION = "validation"
FRESH = "fresh"

#: Which seeds play which role. Declared here rather than in whatever
#: script runs next, so the split is a property of the experiment and not
#: of the person writing it that day.
_SEEDS: Dict[str, Tuple[int, ...]] = {
    DISCOVERY: tuple(range(1, 41)),
    VALIDATION: tuple(range(101, 141)),
    FRESH: tuple(range(201, 241)),
}

#: How many times each split may be scored. Discovery is unlimited: it is
#: the set you are allowed to fit. Validation gets a few, because a
#: handful of candidates is a search and a hundred is fitting.
_BUDGET: Dict[str, Optional[int]] = {
    DISCOVERY: None,
    VALIDATION: 8,
    FRESH: 3,
}

#: And the fresh split answers once per sealed strategy, which is the
#: rule that makes it fresh at all.
#:
#: The two limits are different questions. The per-strategy one stops a
#: caller asking the same question until the noise falls their way. The
#: overall one is smaller than validation's because every answer leaks:
#: seal a second strategy after seeing the first result and that second
#: strategy was chosen knowing something about this split. It is not
#: forbidden -- sometimes the first candidate genuinely fails and a
#: second is genuinely different -- but each answer is a weaker claim
#: than the one before, and `read_number` in the result says which one
#: this is so a reader can discount it.
_PER_STRATEGY = 1

_lock = threading.RLock()
_reads: Dict[str, int] = {DISCOVERY: 0, VALIDATION: 0, FRESH: 0}
_audit: List[Dict[str, Any]] = []
_sealed: Optional[str] = None
_fresh_answered: Dict[str, int] = {}


class BudgetExceeded(RuntimeError):
    """The split has been read as often as it may be."""


class NotSealed(RuntimeError):
    """The fresh split was asked about a strategy nobody committed to."""


def strategy_id(strategy: Dict[str, Any]) -> str:
    """A name for a strategy that changes when the strategy does."""
    body = json.dumps(strategy, sort_keys=True, ensure_ascii=False,
                      default=str)
    return hashlib.blake2b(body.encode("utf-8"), digest_size=8).hexdigest()


def seal(strategy: Dict[str, Any]) -> str:
    """Commit to a strategy. Nothing may be adjusted after this.

    Returns its id. `fresh_score` answers for this id and no other, which
    is the difference between "measured on unseen data" and "measured on
    data that became seen while we were still choosing".
    """
    global _sealed
    with _lock:
        _sealed = strategy_id(strategy)
        _audit.append({"at": time.time(), "event": "seal",
                       "strategy": _sealed, "settings": dict(strategy)})
        return _sealed


def sealed() -> str:
    return _sealed or ""


def size(split: str) -> int:
    """How many seeds a split holds. A count is not the seeds."""
    return len(_SEEDS[split])


def score(split: str, run: Callable[[int], float],
          strategy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a strategy over a split and report numbers, never the seeds.

    `run` is handed one seed at a time and returns a score for it. The
    seeds exist only inside this call: a caller cannot keep them, cannot
    look at which ones went badly, and therefore cannot fit them.
    """
    if split not in _SEEDS:
        raise KeyError(split)
    with _lock:
        if split == FRESH:
            if not _sealed:
                raise NotSealed(
                    "скрытая свежая выборка отвечает только про "
                    "запечатанную стратегию: сначала seal()")
            if strategy is not None and strategy_id(strategy) != _sealed:
                raise NotSealed(
                    "спрашивают про стратегию, отличную от запечатанной")
            if _fresh_answered.get(_sealed, 0) >= _PER_STRATEGY:
                raise BudgetExceeded(
                    f"свежая выборка уже отвечала про {_sealed}; "
                    f"второй ответ сделал бы её набором для подбора")
        allowed = _BUDGET[split]
        if allowed is not None and _reads[split] >= allowed:
            raise BudgetExceeded(
                f"выборка «{split}» прочитана {_reads[split]} раз при "
                f"квоте {allowed}")

    values = [float(run(seed)) for seed in _SEEDS[split]]
    mean = sum(values) / len(values)
    with _lock:
        _reads[split] += 1
        if split == FRESH:
            _fresh_answered[_sealed] = _fresh_answered.get(_sealed, 0) + 1
        _audit.append({"at": time.time(), "event": "score", "split": split,
                       "n": len(values), "mean": round(mean, 6),
                       "strategy": strategy_id(strategy) if strategy else "",
                       "read_number": _reads[split]})
    return {"split": split, "n": len(values), "mean": mean,
            "read_number": _reads[split],
            "values": values if split == DISCOVERY else None}


def paired(split: str, a: Callable[[int], float], b: Callable[[int], float],
           strategy: Optional[Dict[str, Any]] = None,
           rounds: int = 5000, boot_seed: int = 20260908) -> Dict[str, Any]:
    """B minus A over a split, paired by seed, with a bootstrap interval.

    Counts as one read: it is one question asked once, and charging it
    twice would make the honest way of asking cost more than the sloppy
    one.
    """
    import random
    import statistics

    if split not in _SEEDS:
        raise KeyError(split)
    with _lock:
        if split == FRESH:
            if not _sealed:
                raise NotSealed("сначала seal()")
            if strategy is not None and strategy_id(strategy) != _sealed:
                raise NotSealed("не та стратегия, что запечатана")
            if _fresh_answered.get(_sealed, 0) >= _PER_STRATEGY:
                raise BudgetExceeded("свежая выборка уже отвечала")
        allowed = _BUDGET[split]
        if allowed is not None and _reads[split] >= allowed:
            raise BudgetExceeded(f"квота выборки «{split}» исчерпана")

    diffs = [float(b(seed)) - float(a(seed)) for seed in _SEEDS[split]]
    rng = random.Random(boot_seed)
    boot = sorted(statistics.mean([rng.choice(diffs) for _ in diffs])
                  for _ in range(rounds))
    low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    out = {"split": split, "n": len(diffs),
           "mean": statistics.mean(diffs), "low": low, "high": high,
           "better": sum(1 for d in diffs if d > 0),
           "worse": sum(1 for d in diffs if d < 0),
           "same": sum(1 for d in diffs if d == 0),
           "improved": low > 0.0}
    out["read_number"] = _reads[split] + 1
    with _lock:
        _reads[split] += 1
        if split == FRESH:
            _fresh_answered[_sealed] = _fresh_answered.get(_sealed, 0) + 1
        _audit.append({"at": time.time(), "event": "paired", "split": split,
                       "n": out["n"], "mean": round(out["mean"], 6),
                       "low": round(low, 6), "high": round(high, 6),
                       "strategy": strategy_id(strategy) if strategy else "",
                       "read_number": _reads[split]})
    return out


def reads() -> Dict[str, int]:
    with _lock:
        return dict(_reads)


def audit() -> List[Dict[str, Any]]:
    """Every read, in order. A claim about a holdout can be checked."""
    with _lock:
        return [dict(row) for row in _audit]


def describe() -> str:
    lines = ["выборки:"]
    for split in (DISCOVERY, VALIDATION, FRESH):
        allowed = _BUDGET[split]
        lines.append(f"  {split:<11} {size(split):>3} посевов, "
                     f"прочитана {_reads[split]} раз"
                     + (f" из {allowed}" if allowed is not None
                        else " (без квоты — на ней и подбирают)"))
    lines.append(f"  запечатано: {_sealed or '—'}")
    return "\n".join(lines)


def _reset_for_tests() -> None:
    """Only tests call this. A run that reset its own audit log would be
    a run whose holdout claims mean nothing."""
    global _sealed
    with _lock:
        for split in _reads:
            _reads[split] = 0
        _audit.clear()
        _fresh_answered.clear()
        _sealed = None
