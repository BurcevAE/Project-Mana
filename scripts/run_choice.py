"""Experiment M, step M1: choosing a method from experience, when no method
says what it will cost.

A stream: 40 training boxes of n = 2..4 (10 of each structure, shuffled),
the chooser learning as it goes; then 36 test boxes of n = 5, 6, 8 (3 of
each structure and n) with what it learnt frozen. On the same boxes:
    каскад         the writer's order, calculate -> decompose -> experiment
                   -> exhaust, in one session, stopping at the first own
                   check that passes
    случайный      one random order per box: no knowledge, no memory
    оракул         the cheapest method the verdict says is right, alone
Every figure is questions asked, the failed methods' included; solved is
the world's verdict on the model accepted.

    python scripts/run_choice.py [streams] [workers] [out.json]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.methods import choice, portfolio, solvers  # noqa: E402
from mana.methods.world import LEVELS, STRUCTURES, BlackBox  # noqa: E402

TRAIN_N = (2, 3, 4)
TEST_N = (5, 6, 8)
TRAIN_PER = 10
TEST_PER = 3
LETTER = {"exhaust": "A", "calculate": "B", "decompose": "C", "experiment": "D"}


def stream(seed):
    rng = np.random.default_rng([seed, 11])
    train = [(s, int(rng.choice(TRAIN_N))) for s in STRUCTURES for _ in range(TRAIN_PER)]
    rng.shuffle(train)
    test = [(s, n) for s in STRUCTURES for n in TEST_N for _ in range(TEST_PER)]
    chooser = choice.Chooser(choice.Experience(), seed=seed)
    rows = []
    for phase, boxes in (("train", train), ("test", test)):
        for i, (structure, n) in enumerate(boxes):
            box_seed = seed * 1000 + (i if phase == "train" else 500 + i)
            box = BlackBox(n, structure, box_seed)
            picked, model = chooser.solve(box.answer, n, LEVELS, box_seed,
                                          learn=phase == "train")
            mana = {"method": picked.method, "cost": picked.cost,
                    "right": box.verdict(model) == 1.0,
                    "tried": "".join(LETTER[m] for m in picked.tried)}
            box = BlackBox(n, structure, box_seed)
            fixed, model, _ = portfolio.run_order(box.answer, n, LEVELS, box_seed)
            cascade = {"cost": fixed.cost, "right": box.verdict(model) == 1.0,
                       "tried": "".join(LETTER[m] for m in fixed.tried)}
            order = list(portfolio.ORDER)
            rng.shuffle(order)
            box = BlackBox(n, structure, box_seed)
            blind, model, _ = portfolio.run_order(box.answer, n, LEVELS, box_seed, order)
            rand = {"cost": blind.cost, "right": box.verdict(model) == 1.0}
            box = BlackBox(n, structure, box_seed)
            attempts = {m: solvers.attempt(m, box.answer, n, LEVELS, box_seed, announce=False)
                        for m in solvers.METHODS}
            verdicts = {m: box.verdict(a.model) for m, a in attempts.items()}
            best = portfolio.oracle(attempts, verdicts)
            rows.append({"stream": seed, "phase": phase, "index": i, "structure": structure,
                         "n": n, "mana": mana, "cascade": cascade, "random": rand,
                         "oracle": {"method": best.method, "cost": best.cost}})
    return rows


def report(results) -> None:
    arms = (("MANA", "mana"), ("каскад", "cascade"), ("случайный", "random"))
    train = [r for r in results if r["phase"] == "train"]
    print("\n=== обучение, n = 2..4: вопросы на ящик, сумма по потокам / сумма оракула")
    print("  ящики     " + "".join(f"{label:>12s}" for label, _ in arms) + "     решено MANA")
    for start in range(0, 40, 10):
        block = [r for r in train if start <= r["index"] < start + 10]
        oracle = sum(r["oracle"]["cost"] for r in block)
        line = f"  {start + 1:>2d}-{start + 10:<2d}     "
        for _, key in arms:
            line += f"{sum(r[key]['cost'] for r in block) / oracle:>12.2f}"
        line += f"     {sum(r['mana']['right'] for r in block)}/{len(block)}"
        print(line)
    test = [r for r in results if r["phase"] == "test"]
    print("\n=== проверка, n = 5, 6, 8, опыт заморожен: решено / медиана вопросов")
    print("  мир       n " + "".join(f"{label:>18s}" for label, _ in arms) + f"{'оракул':>18s}"
          + "   порядок MANA")
    for structure in STRUCTURES:
        for n in TEST_N:
            mine = [r for r in test if r["structure"] == structure and r["n"] == n]
            line = f"  {structure:8s} {n:>2d} "
            for _, key in arms:
                line += (f"{sum(r[key]['right'] for r in mine):>8d}/{len(mine):<2d}"
                         f"{int(np.median([r[key]['cost'] for r in mine])):>8d}")
            line += (f"{sum(r['oracle']['method'] is not None for r in mine):>8d}/{len(mine):<2d}"
                     f"{int(np.median([r['oracle']['cost'] for r in mine])):>8d}")
            tried = {}
            for r in mine:
                tried[r["mana"]["tried"] or "—"] = tried.get(r["mana"]["tried"] or "—", 0) + 1
            line += "   " + " ".join(f"{k}:{v}" for k, v in sorted(tried.items(), key=lambda kv: -kv[1]))
            print(line)
    print("\n  всего на проверке:")
    for label, key in arms + (("оракул", "oracle"),):
        solved = sum((r[key]["right"] if key != "oracle" else r["oracle"]["method"] is not None)
                     for r in test)
        print(f"    {label:10s} решено {solved:>3d}/{len(test)}, вопросов {sum(r[key]['cost'] for r in test):>10d}")


def main() -> None:
    streams = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = [row for rows in pool.map(stream, range(streams)) for row in rows]
    print(f"{streams} потоков, {len(results)} ящиков, {time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сырые данные: {out}")
    report(results)


if __name__ == "__main__":
    main()
