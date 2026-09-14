"""Discovery, cells: experiment B (adaptive cells) against control A (a grid).

The search inside every cell is the search as it stands -- beam 4, its
own edits -- given a local budget. What changes is only where it looks:
at one region at a time.

Arms:
    B@25k, B@50k, B@100k   adaptive cells, the local budget of each search
    A2, A3                 a grid of 2 and 3 equal bins on every variable
    A10x10                 10 bins on the first two variables: 100 cells

Per arm:
    полнота    the global hypothesis is the rule itself, over every state
    в пределах N   the same, counting a run as solved at N only if it spent
               no more than N programs in all -- every local search, the
               refused splits' included. This is the row to read against
               the zero law of step 5.0 at the same N.
    клетки     leaves and depth of the tree; how often one cell's model
               turned out to be the whole rule
    параллель  total time over time along the longest path of the tree --
               a separate axis, never folded into the count

    python scripts/run_cells.py [seeds] [workers] [out.json]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import cells  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import show  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

TASKS = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")
BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000, 800000, 1200000, 2000000)
#: Step 5.0: the best known search at each budget (%).
ZERO_LAW = (29, 29, 29, 39, 44, 63, 63, 84, 94)
ARMS = {
    "B@25k": ("adaptive", 25000, None),
    "B@50k": ("adaptive", 50000, None),
    "B@100k": ("adaptive", 100000, None),
    "A2": ("grid", 25000, 2),
    "A3": ("grid", 25000, 3),
    "A10x10": ("grid", 25000, 10),
}


def label(budget: int) -> str:
    return f"{budget / 1e6:g}M" if budget >= 1000000 else f"{budget // 1000}k"


def one(job):
    name, seed, arm = job
    world = {**WORLDS, **FAMILY}[name]
    split = world.split(200, 300, seed)
    kind, local, bins = ARMS[arm]
    if kind == "adaptive":
        grown = cells.adaptive(split.train, split.train_outcomes, local_budget=local)
    else:
        over = sorted(split.train)[:2] if bins == 10 else None
        grown = cells.grid(split.train, split.train_outcomes, bins, over=over,
                           local_budget=local)
    program = grown.program
    rule = world.grade(lambda cols: discovery.predict(program, cols)) == 1.0
    unseen = float(np.mean(discovery.predict(program, split.test) != split.test_truth))
    local_right = [cell.found.train_errors == 0 for cell in grown.tree.leaves()
                   if cell.found is not None]
    return {"task": name, "seed": seed, "arm": arm, "rule": rule, "unseen_error": unseen,
            "evaluations": grown.evaluations, "searches": grown.searches,
            "leaves": len(grown.tree.leaves()), "depth": grown.tree.depth(),
            "exact_leaves": float(np.mean(local_right)) if local_right else 0.0,
            "source": grown.source, "bits": grown.bits, "parallel": grown.parallel,
            "seconds": grown.seconds, "program": show(program)}


def report(results) -> None:
    print("\n  нулевой закон    " + "".join(f"{z:>7d}%" for z in ZERO_LAW))
    print("  бюджет          " + "".join(f"{label(b):>8s}" for b in BUDGETS))
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        within = "".join(
            f"{sum(r['rule'] and r['evaluations'] <= b for r in rows) / len(rows):>7.0%} "
            for b in BUDGETS)
        print(f"  {arm:8s} в пред. {within}")
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        solved = [r for r in rows if r["rule"]]
        print(f"\n=== {arm}: {len(rows)} прогонов, полнота {len(solved)}/{len(rows)} "
              f"= {len(solved) / len(rows):.0%}; программ медиана "
              f"{int(np.median([r['evaluations'] for r in rows]))}, "
              f"ошибка на новых {np.mean([r['unseen_error'] for r in rows]):.3f}")
        print("    мир   найдено  программ(мед)  поисков  листьев  глубина  "
              "точн.листья  одна клетка  параллель  ошибка")
        for name in TASKS:
            mine = [r for r in rows if r["task"] == name]
            if not mine:
                continue
            print(f"    {name:4s}  {sum(r['rule'] for r in mine):>3d}/{len(mine):<3d}"
                  f"  {int(np.median([r['evaluations'] for r in mine])):>12d}"
                  f"  {np.mean([r['searches'] for r in mine]):>7.1f}"
                  f"  {np.mean([r['leaves'] for r in mine]):>7.1f}"
                  f"  {np.mean([r['depth'] for r in mine]):>7.1f}"
                  f"  {np.mean([r['exact_leaves'] for r in mine]):>11.2f}"
                  f"  {sum(r['source'] != 'tree' for r in mine):>11d}"
                  f"  {np.mean([r['parallel'] for r in mine]):>9.2f}"
                  f"  {np.mean([r['unseen_error'] for r in mine]):>6.3f}")


def main() -> None:
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    arms = sys.argv[4].split(",") if len(sys.argv) > 4 else list(ARMS)
    jobs = [(name, seed, arm) for arm in arms for name in TASKS for seed in range(seeds)]
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results.append(row)
            print(f"  [{done}/{len(jobs)}] {row['arm']} {row['task']} сид {row['seed']}: "
                  f"{'правило' if row['rule'] else 'нет'}, {row['evaluations']} программ, "
                  f"{row['leaves']} кл., {row['seconds']:.0f}с", flush=True)
    print(f"\nвсего {time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сырые данные: {out}")
    report(results)


if __name__ == "__main__":
    main()
