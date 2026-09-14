"""Experiment M, step M0: the order of the methods, world by world, and the
strength of the controls that choose without knowing anything.

For every structure, n and seed: all four methods on the same box, each
judged by the box on inputs it never asked about. Then, from the same
attempts:
    oracle          the cheapest method the verdict says is right
    cascade         cheapest announced plan first, stop at the first own
                    check that passes; pays the union of what it asked
    random order    the same cascade over all 24 orders, averaged: what
                    choosing with no rule at all costs

    python scripts/run_methods.py [seeds] [workers] [out.json]
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.methods import portfolio, solvers  # noqa: E402
from mana.methods.world import LEVELS, STRUCTURES, BlackBox  # noqa: E402

NS = (2, 3, 4, 5, 6, 8)
LETTER = {"exhaust": "A", "calculate": "B", "decompose": "C", "experiment": "D"}


def one(job):
    structure, n, seed = job
    box = BlackBox(n, structure, seed)
    attempts = {name: solvers.attempt(name, box.answer, n, LEVELS, seed)
                for name in solvers.METHODS}
    verdicts = {name: box.verdict(a.model) for name, a in attempts.items()}
    right = {name: v == 1.0 for name, v in verdicts.items()}
    fixed = portfolio.cascade(attempts)
    orders = [portfolio.cascade(attempts, order)
              for order in itertools.permutations(portfolio.ORDER)]
    best = portfolio.oracle(attempts, verdicts)
    return {"structure": structure, "n": n, "seed": seed,
            "methods": {name: {"right": right[name], "believed": a.believed,
                               "queries": a.queries, "planned": a.planned,
                               "declined": a.declined, "seconds": a.seconds}
                        for name, a in attempts.items()},
            "oracle": {"method": best.method, "cost": best.cost},
            "cascade": {"method": fixed.method, "cost": fixed.cost,
                        "right": bool(fixed.method and right[fixed.method]),
                        "tried": list(fixed.tried)},
            "random": {"cost": float(np.mean([c.cost for c in orders])),
                       "right": float(np.mean([bool(c.method and right[c.method])
                                               for c in orders]))}}


def report(results) -> None:
    for structure in STRUCTURES:
        rows = [r for r in results if r["structure"] == structure]
        print(f"\n=== {structure}   (верно/сидов, медиана вопросов)")
        print("               " + "".join(f"{'n=' + str(n):>16s}" for n in NS))
        for name in portfolio.ORDER:
            line = f"  {LETTER[name]} {name:10s} "
            for n in NS:
                mine = [r for r in rows if r["n"] == n]
                ok = [r for r in mine if r["methods"][name]["right"]]
                cost = int(np.median([r["methods"][name]["queries"] for r in ok])) if ok else 0
                line += f"{len(ok):>5d}/{len(mine):<2d}{cost:>8d} " if ok else f"{0:>5d}/{len(mine):<2d}{'—':>8s} "
            print(line)
        for label, key in (("оракул", "oracle"), ("каскад", "cascade"), ("случ.порядок", "random")):
            line = f"  {label:12s} "
            for n in NS:
                mine = [r for r in rows if r["n"] == n]
                if key == "oracle":
                    ok = sum(r["oracle"]["method"] is not None for r in mine)
                    cost = np.median([r["oracle"]["cost"] for r in mine])
                elif key == "cascade":
                    ok = sum(r["cascade"]["right"] for r in mine)
                    cost = np.median([r["cascade"]["cost"] for r in mine])
                else:
                    ok = sum(r["random"]["right"] for r in mine)
                    cost = np.median([r["random"]["cost"] for r in mine])
                line += f"{ok:>5.0f}/{len(mine):<2d}{int(cost):>8d} "
            print(line)
        line = "  каскад/оракул "
        order_line = "  порядок      "
        for n in NS:
            mine = [r for r in rows if r["n"] == n and r["oracle"]["cost"] > 0]
            ratio = np.median([r["cascade"]["cost"] / r["oracle"]["cost"] for r in mine]) if mine else float("nan")
            line += f"{ratio:>16.2f}"
            everyone = [r for r in rows if r["n"] == n]
            good = []
            for name in solvers.METHODS:
                ok = [r["methods"][name]["queries"] for r in everyone if r["methods"][name]["right"]]
                if len(ok) >= 0.8 * len(everyone):
                    good.append((np.median(ok), LETTER[name]))
            order_line += f"{'<'.join(letter for _, letter in sorted(good)) or 'никто':>16s}"
        print(line)
        print(order_line)
    attempts = [m for r in results for m in r["methods"].values() if m["queries"] > 0]
    agree = sum(m["believed"] == m["right"] for m in attempts)
    fooled = sum(m["believed"] and not m["right"] for m in attempts)
    print(f"\nпроверка метода согласна с вердиктом: {agree}/{len(attempts)}; "
          f"поверил себе и ошибся: {fooled}")


def main() -> None:
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    jobs = [(s, n, seed) for s in STRUCTURES for n in NS for seed in range(seeds)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, jobs))
    print(f"{len(results)} ящиков, {time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сырые данные: {out}")
    report(results)


if __name__ == "__main__":
    main()
