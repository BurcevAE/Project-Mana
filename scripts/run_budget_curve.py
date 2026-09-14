"""Discovery, step 5.0: the search's budget curve -- the zero law.

Every later change to the search is measured against this: at N programs
evaluated, does it find more existing solutions than the search as it is?
Finding more with more programs is buying completeness with CPU, not with
a better way of looking.

One profiled run per (question, seed, beam) with the largest budget; every
smaller budget is read off it -- search.at_budget, exact, tested against
separate runs. So every budget sees the same seeds and the same data.

Per budget and beam:
    полнота    P(found | a solution exists): the rule itself, over every
               state (a solution exists in every run: step 4's certificate)
    стоимость  programs evaluated before the answer was first seen, among
               the solved
    качество   bits of what was returned, and its error on unseen states
    тупики     misses that stopped at a local end rather than at the budget
And per question and seed, the smallest budget at which it is solved.

    python scripts/run_budget_curve.py [seeds] [workers] [out.json]
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

from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import show  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000, 800000, 1200000, 2000000)
BEAMS = (4, 16)
TASKS = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")


def label(budget: int) -> str:
    return f"{budget / 1e6:g}M" if budget >= 1000000 else f"{budget // 1000}k"


def one(job):
    name, seed, beam = job
    world = {**WORLDS, **FAMILY}[name]
    split = world.split(200, 300, seed)
    started = time.time()
    found = discovery.search(split.train, split.train_outcomes, beam_width=beam,
                             budget=max(BUDGETS), profile=True)
    graded = {}
    points = []
    for budget in BUDGETS:
        program, found_at, bits = discovery.at_budget(found, budget)
        if program not in graded:
            rule = world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0
            unseen = float(np.mean(discovery.predict(program, split.test) != split.test_truth))
            graded[program] = (rule, unseen)
        rule, unseen = graded[program]
        points.append({"budget": budget, "rule": rule, "found_at": found_at,
                       "bits": bits, "unseen_error": unseen,
                       "termination": discovery.termination_at(found, budget),
                       "program": show(program)})
    return {"task": name, "seed": seed, "beam": beam, "evaluations": found.evaluations,
            "termination": found.termination, "seconds": time.time() - started,
            "points": points}


def report(results) -> None:
    for beam in BEAMS:
        rows = [r for r in results if r["beam"] == beam]
        print(f"\n=== луч {beam}: {len(rows)} прогонов")
        print("  бюджет          " + "".join(f"{label(b):>8s}" for b in BUDGETS))

        def at(r, i):
            return r["points"][i]

        line = "  полнота         "
        for i, _ in enumerate(BUDGETS):
            solved = sum(at(r, i)["rule"] for r in rows)
            line += f"{solved / len(rows):>7.0%} "
        print(line)
        for name in TASKS:
            mine = [r for r in rows if r["task"] == name]
            print(f"    {name:4s}          " + "".join(
                f"{sum(at(r, i)['rule'] for r in mine):>6d}/{len(mine)}"
                for i, _ in enumerate(BUDGETS)))
        cost, bits, unseen, stuck = [], [], [], []
        for i, _ in enumerate(BUDGETS):
            solved = [at(r, i)["found_at"] for r in rows if at(r, i)["rule"]]
            cost.append(f"{int(np.median(solved)) if solved else 0:>8d}")
            bits.append(f"{np.mean([at(r, i)['bits'] for r in rows]):>8.1f}")
            unseen.append(f"{np.mean([at(r, i)['unseen_error'] for r in rows]):>8.3f}")
            missed = [r for r in rows if not at(r, i)["rule"]]
            ends = sum(at(r, i)["termination"] == discovery.SEARCH_EXHAUSTED for r in missed)
            stuck.append(f"{ends:>4d}/{len(missed):<3d}")
        print("  стоимость (мед) " + "".join(cost))
        print("  биты (ср)       " + "".join(bits))
        print("  ошибка на новых " + "".join(unseen))
        print("  тупики/провалы  " + "".join(stuck))

    print("\n=== наименьший бюджет решения, по сидам (луч 4 | луч 16; — не решено до 2M)")
    for name in TASKS:
        cells = []
        for beam in BEAMS:
            row = []
            for r in sorted((r for r in results if r["beam"] == beam and r["task"] == name),
                            key=lambda r: r["seed"]):
                first = next((p["budget"] for p in r["points"] if p["rule"]), None)
                row.append(label(first) if first else "—")
            cells.append(" ".join(f"{c:>5s}" for c in row))
        print(f"  {name:4s} {cells[0]}  |  {cells[1]}")


def main() -> None:
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    jobs = [(name, seed, beam) for beam in BEAMS for name in TASKS for seed in range(seeds)]
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results.append(row)
            print(f"  [{done}/{len(jobs)}] {row['task']} сид {row['seed']} луч {row['beam']}: "
                  f"{row['evaluations']} программ, {row['termination']}, "
                  f"{row['seconds']:.0f}с", flush=True)
    print(f"\nвсего {time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сырые данные: {out}")
    report(results)


if __name__ == "__main__":
    main()
