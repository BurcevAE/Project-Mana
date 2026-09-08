"""
mana.cognition.trials — three splits, and a fresh one that cannot be peeked at.

Why this is not a list of seeds in a script
--------------------------------------------
The first experiment that needed a holdout kept its split in a module
tuple and its discipline in my head: discovery 1..40, validation 101..140,
and a promise not to look at the second until the strategy was fixed. The
promise held, and a promise is not a mechanism. `mana/core/splits.py`
solved the same problem long before and its answer is the one used here:
**a hidden set is never returned as data.** There is no `fresh_seeds()`.
The only way to interact with it is to hand in a strategy and get numbers.

Three roles, because two were not enough
-----------------------------------------
    DISCOVERY    visible, read as often as you like: this is where a grid
                 is searched and a hypothesis is formed
    VALIDATION   scored, never returned, small budget: does the change
                 survive contact with seeds it was not chosen on
    FRESH        scored once per sealed strategy, and only after sealing

The third exists because one good validation can be luck. A strategy
adjusted after seeing a validation score has been fitted to it, and the
only way to tell that from a real effect is a split that did not exist for
the adjuster -- which is what sealing enforces.

One discipline, several experiments
------------------------------------
Every experiment names itself and carries its own seeds, budgets, seal and
audit. Budgets used to be global, which would have handed a second
experiment a fresh split already spent by the first: a quota shared
between unrelated questions is a race, not a quota. Asking about a name
nobody registered raises, because a split that appears when you ask for it
is one nobody agreed to.

Budget, not honour system
--------------------------
Every read is counted, capped and written to an audit log, the same
reasoning `core/splits.py` gives: unlimited access to a hidden set is the
same leak as reading it, only slower -- enough queries and a searcher fits
the set through the score alone. Exceeding a budget raises rather than
degrading quietly, and `audit()` shows what was read and when, so a claim
about a holdout can be checked instead of believed.

And it survives a restart, which it did not at first. The naming
experiment read its fresh split, the rule was corrected, the script ran
again in a new process, and the counter said "read once" both times. It
had been read twice. A budget a restart resets is a promise with extra
steps, so the counts, the seals and the audit live on disk from here --
what was read and when, never the seeds.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

DISCOVERY = "discovery"
VALIDATION = "validation"
FRESH = "fresh"
SPLITS = (DISCOVERY, VALIDATION, FRESH)

#: The seeds each role gets by default. Declared here rather than in
#: whatever script runs next, so a split is a property of the experiment
#: and not of the person writing it that day.
DEFAULT_SEEDS: Dict[str, Tuple[int, ...]] = {
    DISCOVERY: tuple(range(1, 41)),
    VALIDATION: tuple(range(101, 141)),
    FRESH: tuple(range(201, 241)),
}

#: How many times each split may be scored. Discovery is unlimited: it is
#: the set you are allowed to fit. Validation gets a few, because a
#: handful of candidates is a search and a hundred is fitting.
DEFAULT_BUDGET: Dict[str, Optional[int]] = {
    DISCOVERY: None,
    VALIDATION: 8,
    FRESH: 3,
}

#: And the fresh split answers once per sealed strategy, which is the rule
#: that makes it fresh at all.
#:
#: The two limits are different questions. The per-strategy one stops a
#: caller asking the same question until the noise falls their way. The
#: overall one is smaller than validation's because every answer leaks:
#: seal a second strategy after seeing the first result and that second
#: strategy was chosen knowing something about this split. Not forbidden
#: -- sometimes the first candidate genuinely fails -- but each answer is
#: a weaker claim than the one before, and `read_number` says which one
#: this is so a reader can discount it.
PER_STRATEGY = 1

#: The experiments that have run through here. Named so that a reader of
#: the audit log knows which question spent which budget.
WORLD_MODEL = "world_model"
TASK_NAMING = "task_naming"

_lock = threading.RLock()


def _store() -> Path:
    """Beside the rest of the user's state, like the journal and the book.

    Evidence about an experiment kept inside the thing being experimented
    on disappears with it on reinstall -- and a holdout audit that can be
    lost by reinstalling is one nobody can rely on.
    """
    from ..paths import resolve_data_path
    return Path(resolve_data_path("trials"))


@dataclass
class Experiment:
    """One question, with its own splits and its own quotas."""
    name: str
    seeds: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: dict(DEFAULT_SEEDS))
    budget: Dict[str, Optional[int]] = field(
        default_factory=lambda: dict(DEFAULT_BUDGET))
    reads: Dict[str, int] = field(
        default_factory=lambda: {DISCOVERY: 0, VALIDATION: 0, FRESH: 0})
    sealed: str = ""
    fresh_answered: Dict[str, int] = field(default_factory=dict)
    audit: List[Dict[str, Any]] = field(default_factory=list)
    #: Whether the counts made it to disk. False means this run is
    #: protected and the next one will not be, which is a different thing
    #: from being protected and should not look the same.
    persisted: bool = True


#: Explicit rather than created on demand: a split nobody agreed to is not
#: a holdout.
_EXPERIMENTS: Dict[str, Experiment] = {}


class BudgetExceeded(RuntimeError):
    """The split has been read as often as it may be."""


class NotSealed(RuntimeError):
    """The fresh split was asked about a strategy nobody committed to."""


def _load(name: str) -> Optional[Dict[str, Any]]:
    path = _store() / f"{name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save(found: Experiment) -> None:
    """Never the reason an experiment fails, and never silent about it.

    A write that cannot happen leaves the in-process counts standing, so
    the run is still protected; what is lost is protection across a
    restart, and `persisted` says which of the two a reader is looking at.
    """
    try:
        directory = _store()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{found.name}.json").write_text(
            json.dumps({"name": found.name, "reads": found.reads,
                        "sealed": found.sealed,
                        "fresh_answered": found.fresh_answered,
                        "audit": found.audit[-500:]},
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
        found.persisted = True
    except Exception:
        found.persisted = False


def register(name: str, seeds: Optional[Dict[str, Tuple[int, ...]]] = None,
             budget: Optional[Dict[str, Optional[int]]] = None) -> Experiment:
    """Declare an experiment, picking up what it has already spent.

    Re-registering an existing one returns it rather than resetting: a
    second call that wiped the read counts would be a way to get a fresh
    split back by asking twice, and so would a restart before the counts
    were written down.
    """
    with _lock:
        found = _EXPERIMENTS.get(name)
        if found is not None:
            return found
        made = Experiment(name=name, seeds=dict(seeds or DEFAULT_SEEDS),
                          budget=dict(budget or DEFAULT_BUDGET))
        stored = _load(name)
        if stored:
            made.reads.update({k: int(v) for k, v in
                               (stored.get("reads") or {}).items()})
            made.sealed = str(stored.get("sealed") or "")
            made.fresh_answered = {k: int(v) for k, v in
                                   (stored.get("fresh_answered") or {}).items()}
            made.audit = list(stored.get("audit") or [])
        _EXPERIMENTS[name] = made
        return made


def experiment(name: str) -> Experiment:
    with _lock:
        found = _EXPERIMENTS.get(name)
    if found is None:
        raise KeyError(
            f"опыт «{name}» не объявлен: сначала register(), иначе выборка "
            f"появляется по запросу и держать её никто не обещал")
    return found


def strategy_id(strategy: Dict[str, Any]) -> str:
    """A name for a strategy that changes when the strategy does."""
    body = json.dumps(strategy, sort_keys=True, ensure_ascii=False,
                      default=str)
    return hashlib.blake2b(body.encode("utf-8"), digest_size=8).hexdigest()


def seal(name: str, strategy: Dict[str, Any]) -> str:
    """Commit to a strategy. Nothing may be adjusted after this.

    Returns its id. The fresh split answers for this id and no other,
    which is the difference between "measured on unseen data" and
    "measured on data that became seen while we were still choosing".
    """
    found = experiment(name)
    with _lock:
        found.sealed = strategy_id(strategy)
        found.audit.append({"at": time.time(), "event": "seal",
                            "strategy": found.sealed,
                            "settings": dict(strategy)})
    _save(found)
    return found.sealed


def size(name: str, split: str) -> int:
    """How many seeds a split holds. A count is not the seeds."""
    return len(experiment(name).seeds[split])


def _admit(found: Experiment, split: str,
           strategy: Optional[Dict[str, Any]]) -> None:
    """Everything that must hold before a split may be read."""
    if split not in found.seeds:
        raise KeyError(split)
    with _lock:
        if split == FRESH:
            if not found.sealed:
                raise NotSealed(
                    "свежая выборка отвечает только про запечатанную "
                    "стратегию: сначала seal()")
            if strategy is not None and strategy_id(strategy) != found.sealed:
                raise NotSealed(
                    "спрашивают про стратегию, отличную от запечатанной")
            if found.fresh_answered.get(found.sealed, 0) >= PER_STRATEGY:
                raise BudgetExceeded(
                    f"свежая выборка уже отвечала про {found.sealed}; "
                    f"второй ответ сделал бы её набором для подбора")
        allowed = found.budget[split]
        if allowed is not None and found.reads[split] >= allowed:
            raise BudgetExceeded(
                f"выборка «{split}» опыта «{found.name}» прочитана "
                f"{found.reads[split]} раз при квоте {allowed}")


def _charge(found: Experiment, split: str, event: Dict[str, Any]) -> int:
    with _lock:
        found.reads[split] += 1
        if split == FRESH:
            found.fresh_answered[found.sealed] = (
                found.fresh_answered.get(found.sealed, 0) + 1)
        event["read_number"] = found.reads[split]
        found.audit.append(event)
        read = found.reads[split]
    _save(found)
    return read


def score(name: str, split: str, run: Callable[[int], float],
          strategy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a strategy over a split and report numbers, never the seeds.

    `run` is handed one seed at a time and returns a score for it. The
    seeds exist only inside this call: a caller cannot keep them, cannot
    see which ones went badly, and therefore cannot fit them.
    """
    found = experiment(name)
    _admit(found, split, strategy)
    values = [float(run(seed)) for seed in found.seeds[split]]
    mean = statistics.mean(values)
    read = _charge(found, split, {
        "at": time.time(), "event": "score", "split": split,
        "n": len(values), "mean": round(mean, 6),
        "strategy": strategy_id(strategy) if strategy else ""})
    return {"experiment": name, "split": split, "n": len(values),
            "mean": mean, "read_number": read,
            "values": values if split == DISCOVERY else None}


def paired(name: str, split: str, a: Callable[[int], float],
           b: Callable[[int], float],
           strategy: Optional[Dict[str, Any]] = None,
           rounds: int = 5000, boot_seed: int = 20260908) -> Dict[str, Any]:
    """B minus A over a split, paired by seed, with a bootstrap interval.

    Counts as one read: it is one question asked once, and charging it
    twice would make the honest way of asking cost more than the sloppy
    one.
    """
    found = experiment(name)
    _admit(found, split, strategy)
    diffs = [float(b(seed)) - float(a(seed)) for seed in found.seeds[split]]
    rng = random.Random(boot_seed)
    boot = sorted(statistics.mean([rng.choice(diffs) for _ in diffs])
                  for _ in range(rounds))
    low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    out = {"experiment": name, "split": split, "n": len(diffs),
           "mean": statistics.mean(diffs), "low": low, "high": high,
           "better": sum(1 for d in diffs if d > 0),
           "worse": sum(1 for d in diffs if d < 0),
           "same": sum(1 for d in diffs if d == 0),
           "improved": low > 0.0}
    out["read_number"] = _charge(found, split, {
        "at": time.time(), "event": "paired", "split": split,
        "n": out["n"], "mean": round(out["mean"], 6),
        "low": round(low, 6), "high": round(high, 6),
        "strategy": strategy_id(strategy) if strategy else ""})
    return out


def reads(name: str) -> Dict[str, int]:
    return dict(experiment(name).reads)


def sealed(name: str) -> str:
    return experiment(name).sealed


def audit(name: str) -> List[Dict[str, Any]]:
    """Every read of this experiment, in order."""
    return [dict(row) for row in experiment(name).audit]


def describe(name: str) -> str:
    found = experiment(name)
    lines = [f"опыт «{found.name}», выборки:"]
    for split in SPLITS:
        allowed = found.budget[split]
        lines.append(f"  {split:<11} {len(found.seeds[split]):>3} посевов, "
                     f"прочитана {found.reads[split]} раз"
                     + (f" из {allowed}" if allowed is not None
                        else " (без квоты — на ней и подбирают)"))
    lines.append(f"  запечатано: {found.sealed or '—'}")
    if not found.persisted:
        lines.append("  ВНИМАНИЕ: счётчики не пишутся на диск — "
                     "после перезапуска квота начнётся заново")
    return "\n".join(lines)


def _reset_for_tests() -> None:
    """Only tests call this. A run that reset its own audit log would be
    a run whose holdout claims mean nothing.

    In-memory only: it deliberately does not delete what is on disk, so a
    test cannot quietly hand a spent holdout back to the next run.
    """
    with _lock:
        _EXPERIMENTS.clear()
