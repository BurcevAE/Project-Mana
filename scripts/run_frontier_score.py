"""Discovery, step 7: which states the beam keeps, learnt from the search's history.

Stage 1  T1, T2, T3, seeds 0..9: an expensive search (beam 16, 2M) with its
         derivation kept. T4 takes no part in anything before stage 4.
Stage 2  the frontier's decisions along every solved derivation: the state
         that led on, and the programs made from the same parent (frontier.py).
Stage 3  a score fitted to them; two controls of the same size -- its
         weights permuted between features, and random weights of its norm.
Stage 4  the search as it stands (beam 4, 800k), with each score deciding
         only which states the beam keeps; recall by budget, as in 5.0:
             T4 seeds 0..9 (transfer; kept by hand: 10/10)
             W0, W3, W4 seeds 0..9, T1..T3 seeds 10..19

    python scripts/run_frontier_score.py [workers] [out.json]
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

from mana.discovery import frontier  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import show  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

TRAIN = ("T1", "T2", "T3")
BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000, 800000)
TESTS = [("T4", s) for s in range(10)] + [(n, s) for n in ("W0", "W3", "W4") for s in range(10)] \
    + [(n, s) for n in TRAIN for s in range(10, 20)]
ALL = {**WORLDS, **FAMILY}


def label(budget):
    return f"{budget // 1000}k"


def trace_job(job):
    name, seed = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    found = discovery.search(split.train, split.train_outcomes, beam_width=16,
                             budget=2000000, trace=True)
    solved = world.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    return name, seed, solved, found.derivation


def steps_job(job):
    name, seed, derivation = job
    split = ALL[name].split(200, 300, seed)
    return frontier.steps(derivation, split.train, split.train_outcomes, seed=seed)


def test_job(job):
    name, seed, arm, score = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    found = discovery.search(split.train, split.train_outcomes, budget=max(BUDGETS),
                             profile=True, frontier=score)
    points = []
    for budget in BUDGETS:
        program, found_at, _ = discovery.at_budget(found, budget)
        rule = world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0
        points.append({"rule": rule, "found_at": found_at})
    return {"task": name, "seed": seed, "arm": arm, "points": points,
            "program": show(found.program), "bits": found.bits}


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        traces = list(pool.map(trace_job, [(n, s) for n in TRAIN for s in range(10)]))
        solved = [(n, s, d) for n, s, ok, d in traces if ok]
        data = [step for part in pool.map(steps_job, solved) for step in part]
        print(f"стадия 1-2: решено {len(solved)}/{len(traces)}, решений фронта {len(data)}, "
              f"{time.time() - started:.0f}с", flush=True)
        learnt = frontier.fit(data)
        stand = frontier.base(learnt.scale)
        controls = [frontier.permuted(learnt, 0), frontier.random_like(learnt, 0)]
        ranks = np.array([s.base_rank for s in data])
        print(f"  где мера как есть ставила состояние, которое вело к ответу, среди его братьев: "
              f"медиана {int(np.median(ranks))}-е из {int(np.median([s.made for s in data]))}; "
              f"вне первых четырёх в {np.mean(ranks > 4):.0%} решений")
        print("  веса (на величинах как есть: биты программы, биты ошибок, размер, ошибок):")
        for score in [stand, learnt] + controls:
            print(f"    {score.name:13s} {np.array2string(score.raw(), precision=3)}   "
                  f"верный порядок в истории {frontier.accuracy(score, data):.1%}")
        arms = {"как есть": None, "выучена": learnt, "переставлена": controls[0],
                "случайная": controls[1]}
        results = list(pool.map(test_job, [(n, s, arm, score) for arm, score in arms.items()
                                           for n, s in TESTS]))
    print(f"\nвсего {time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сырые данные: {out}")
    report(results, list(arms))


def report(results, arms):
    groups = [("T4 — перенос (удержанием вручную: 10 из 10)", ("T4",)),
              ("W0, W3, W4", ("W0", "W3", "W4")),
              ("T1-T3 на новых данных", TRAIN),
              ("все семь", ("T4", "W0", "W3", "W4") + TRAIN)]
    for title, tasks in groups:
        rows = [r for r in results if r["task"] in tasks]
        per = len(rows) // len(arms)
        print(f"\n=== {title}: решено при бюджете (из {per})")
        print("  бюджет          " + "".join(f"{label(b):>7s}" for b in BUDGETS)
              + "   программ до ответа (мед)")
        for arm in arms:
            mine = [r for r in rows if r["arm"] == arm]
            line = "".join(f"{sum(r['points'][i]['rule'] for r in mine):>7d}"
                           for i in range(len(BUDGETS)))
            last = [r["points"][-1]["found_at"] for r in mine if r["points"][-1]["rule"]]
            print(f"  {arm:14s} {line}   {int(np.median(last)) if last else 0:>18d}")


if __name__ == "__main__":
    main()
