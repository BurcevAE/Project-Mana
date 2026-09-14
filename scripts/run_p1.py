"""N3, probe P1: can the selection language say a selection of another kind,
and does one written by hand do better?

A diagnostic, like the kept state of step 6's control: the two programs of
another kind are written by hand (selection.py), and nothing of them enters
an N3 experiment.

Part 1  of what kind: on T4 seeds 0..2, the first rounds of each arm's run,
        each pool chosen from again with one kept state removed -- a
        reversal (another kept state lost) is impossible for any score of a
        state on its own.
Part 2  does it do better: the seven questions x seeds 0..9, 800k, recall
        by budget, and the seconds the selection itself spent.

    python scripts/run_p1.py [workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import policy as P  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery import selection  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

ALL = {**WORLDS, **FAMILY}
TASKS = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")
BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000, 800000)
ARMS = {"R1": selection.R1, "покрытие": selection.COVER, "разнообразие": selection.DIVERSE}


def label(b):
    return f"{b // 1000}k"


def kind_job(job):
    arm, seed = job
    world = ALL["T4"]
    split = world.split(200, 300, seed)
    policy = P.with_budget(P.with_selection(P.CURRENT, ARMS[arm]), 60000)
    rounds = []
    P.run(policy, split.train, split.train_outcomes, log_rounds=rounds)
    ctx = P.selection_context(policy, split.train, split.train_outcomes)
    tested = [selection.reversals(ARMS[arm], pool, ctx) for pool, _ in rounds[1:4]]
    return arm, seed, sum(tested), len(tested)


def run_job(job):
    arm, name, seed = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    policy = P.with_budget(P.with_selection(P.CURRENT, ARMS[arm]), max(BUDGETS))
    found = P.run(policy, split.train, split.train_outcomes, profile=True)
    solved = []
    for b in BUDGETS:
        program, _, _ = discovery.at_budget(found, b)
        solved.append(world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0)
    return arm, name, seed, solved, found.seconds, found.selection_seconds


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    for arm, program in ARMS.items():
        print(f"  {arm}: {selection.size_of(program)} узлов")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        kinds = list(pool.map(kind_job, [(arm, s) for arm in ARMS for s in range(3)]))
        runs = list(pool.map(run_job, [(arm, n, s) for arm in ARMS for n in TASKS
                                       for s in range(10)]))
    print(f"{time.time() - started:.0f}с")
    print("\n=== часть 1: какого рода отбор (T4, раунды 1-3, убираем по одному удержанному)")
    for arm in ARMS:
        mine = [r for r in kinds if r[0] == arm]
        print(f"  {arm:13s} раундов с разворотом {sum(r[2] for r in mine)}/{sum(r[3] for r in mine)}")
    print("\n=== часть 2: решено при бюджете")
    print("  бюджет              " + "".join(f"{label(b):>7s}" for b in BUDGETS)
          + "   доля времени на отбор")
    for title, tasks in (("T4 (удержание: 10/10)", ("T4",)), ("остальные шесть", ("W0", "W3", "T1", "T2", "T3", "W4"))):
        print(f"  {title}")
        for arm in ARMS:
            mine = [r for r in runs if r[0] == arm and r[1] in tasks]
            line = "".join(f"{sum(r[3][i] for r in mine):>7d}" for i in range(len(BUDGETS)))
            share = sum(r[5] for r in mine) / max(sum(r[4] for r in mine), 1e-9)
            print(f"    {arm:15s} {line}   {share:.0%}")


if __name__ == "__main__":
    main()
