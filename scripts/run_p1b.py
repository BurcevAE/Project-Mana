"""N3, probe P1b: P1 again, with the operation it left out -- a state's successors.

A selection that looks at what a state becomes, written by hand as a
diagnostic: keep the best by score; then, three times, of the first M by
score, the state whose best successor is best (M = 32 and 256). Every
successor it looks at is evaluated in the search's own budget.

Part 1  of what kind: reversals on T4's first rounds, as in P1.
Part 2  does it do better: seven questions x seeds 0..9, 800k, recall by
        budget; the share of the budget the looking took; and on T4,
        whether the state step 6's control kept, if(x < 3, z, y), was kept
        in round 1 -- and where it ranked by score.

    python scripts/run_p1b.py [workers]
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
from mana.discovery.language import LESS, cmp, const, get, if_  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

ALL = {**WORLDS, **FAMILY}
TASKS = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")
BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000, 800000)
ARMS = {"R1": selection.R1, "взгляд 32": selection.look(32), "взгляд 256": selection.look(256)}
STONE = if_(cmp(LESS, get("x"), const(3)), get("z"), get("y"))


def label(b):
    return f"{b // 1000}k"


def kind_job(job):
    arm, seed = job
    split = ALL["T4"].split(200, 300, seed)
    policy = P.with_budget(P.with_selection(P.CURRENT, ARMS[arm]), 60000)
    rounds = []
    P.run(policy, split.train, split.train_outcomes, log_rounds=rounds)
    ctx = P.selection_context(policy, split.train, split.train_outcomes)
    tested = [selection.reversals(ARMS[arm], pool, ctx) for pool, _ in rounds[1:3]]
    return arm, seed, sum(tested), len(tested)


def run_job(job):
    arm, name, seed = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    policy = P.with_budget(P.with_selection(P.CURRENT, ARMS[arm]), max(BUDGETS))
    rounds = [] if name == "T4" else None
    found = P.run(policy, split.train, split.train_outcomes, profile=True, log_rounds=rounds)
    solved = []
    for b in BUDGETS:
        program, _, _ = discovery.at_budget(found, b)
        solved.append(world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0)
    stone = None
    if rounds and len(rounds) > 1:
        pool, kept = rounds[1]
        ctx = P.selection_context(policy, split.train, split.train_outcomes)
        if STONE in pool:
            mine = ctx.score(STONE)
            stone = (STONE in kept, 1 + sum(1 for p in pool if ctx.score(p) < mine), len(pool))
    return (arm, name, seed, solved, found.seconds, found.selection_seconds,
            found.looked, found.evaluations, found.rounds, stone)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        kinds = list(pool.map(kind_job, [(arm, s) for arm in ARMS for s in range(3)]))
        runs = list(pool.map(run_job, [(arm, n, s) for arm in ARMS for n in TASKS
                                       for s in range(10)]))
    print(f"{time.time() - started:.0f}с")
    print("\n=== часть 1: какого рода отбор (T4, раунды 1-2, убираем по одному удержанному)")
    for arm in ARMS:
        mine = [r for r in kinds if r[0] == arm]
        print(f"  {arm:11s} раундов с разворотом {sum(r[2] for r in mine)}/{sum(r[3] for r in mine)}")
    print("\n=== часть 2: решено при бюджете")
    print("  бюджет              " + "".join(f"{label(b):>7s}" for b in BUDGETS)
          + "   доля бюджета на взгляд   раундов (мед)")
    for title, tasks in (("T4 (удержание: 10/10)", ("T4",)),
                         ("остальные шесть", ("W0", "W3", "T1", "T2", "T3", "W4"))):
        print(f"  {title}")
        for arm in ARMS:
            mine = [r for r in runs if r[0] == arm and r[1] in tasks]
            line = "".join(f"{sum(r[3][i] for r in mine):>7d}" for i in range(len(BUDGETS)))
            share = sum(r[6] for r in mine) / max(sum(r[7] for r in mine), 1)
            rounds = sorted(r[8] for r in mine)[len(mine) // 2]
            print(f"    {arm:15s} {line}   {share:>18.0%}   {rounds:>10d}")
    print("\n=== состояние из контроля шага 6 в раунде 1 T4: удержано? место по оценке")
    for arm in ARMS:
        cells = []
        for r in sorted((r for r in runs if r[0] == arm and r[1] == "T4"), key=lambda r: r[2]):
            stone = r[9]
            cells.append("—" if stone is None else f"{'да' if stone[0] else 'нет'} {stone[1]}")
        print(f"  {arm:11s} " + ", ".join(cells))


if __name__ == "__main__":
    main()
