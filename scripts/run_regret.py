"""Discovery, step 7b: a frontier learnt from regret, the last control.

The question, fixed with the user: can MANA distil a dearer search of its
own into a cheaper frontier, from a comparison of the two -- within the
space of scores already given? Not: can it grow a new measure. T4 cannot
be won in this space: the state its answer is reached from is worse than
a bare leaf in all four quantities a score reads.

Stage 1  T1, T2, T3, seeds 0..9: a cheap search (beam 4, 400k) logging
         what its frontier made and kept, and a dear one (beam 16, 2M)
         keeping its derivation. T4 takes no part before stage 3.
Stage 2  regret: states of the dear derivation the cheap frontier made and
         never kept, against what it kept that round; a correction to the
         bits fitted to them (frontier.fit_correction); its weights
         permuted, and random weights of its norm, as controls.
Stage 3  recall by budget on all seven questions: the search as it stands,
         with each score, and the dear policy itself (beam 16).

    python scripts/run_regret.py [workers] [out.json]
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
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

TRAIN = ("T1", "T2", "T3")
BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000, 800000)
TESTS = [("T4", s) for s in range(10)] + [(n, s) for n in ("W0", "W3", "W4") for s in range(10)] \
    + [(n, s) for n in TRAIN for s in range(10, 20)]
ALL = {**WORLDS, **FAMILY}


def label(budget):
    return f"{budget // 1000}k"


def regret_job(job):
    name, seed = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    rounds = []
    cheap = discovery.search(split.train, split.train_outcomes, budget=400000, log_rounds=rounds)
    dear = discovery.search(split.train, split.train_outcomes, beam_width=16, budget=2000000,
                            trace=True)
    grade = lambda p: world.grade(lambda cols: discovery.predict(p, cols)) == 1.0  # noqa: E731
    cheap_ok, dear_ok = grade(cheap.program), grade(dear.program)
    found, counts = (frontier.regrets(dear.derivation, rounds, split.train, split.train_outcomes)
                     if dear_ok else ([], {}))
    return name, seed, cheap_ok, dear_ok, found, counts


def test_job(job):
    name, seed, arm, score = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    kwargs = {"budget": max(BUDGETS), "profile": True}
    if arm == "широкий луч":
        kwargs["beam_width"] = 16
    elif score is not None:
        kwargs["frontier"] = score
    found = discovery.search(split.train, split.train_outcomes, **kwargs)
    points = []
    for budget in BUDGETS:
        program, found_at, _ = discovery.at_budget(found, budget)
        rule = world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0
        points.append({"rule": rule, "found_at": found_at})
    return {"task": name, "seed": seed, "arm": arm, "points": points}


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(regret_job, [(n, s) for n in TRAIN for s in range(10)]))
        data = [r for *_, found, _ in runs for r in found]
        failed_cheap = [(n, s) for n, s, cheap_ok, dear_ok, _, _ in runs if not cheap_ok and dear_ok]
        total = {key: sum(c.get(key, 0) for *_, c in runs) for key in ("path", "kept", "dropped", "never made")}
        print(f"стадия 1: дешёвый поиск решил {sum(r[2] for r in runs)}/30, дорогой {sum(r[3] for r in runs)}/30; "
              f"дешёвый провалил, дорогой решил: {len(failed_cheap)}; {time.time() - started:.0f}с")
        print(f"  состояния дорогих выводов: {total['path']}; дешёвый держал {total['kept']}, "
              f"сделал и выбросил {total['dropped']}, не сделал вовсе {total['never made']}")
        if not data:
            print("  сожаления нет: учить нечему")
            return
        dominated = np.array([r.dominated for r in data])
        print(f"  сожалений {len(data)}; доля удержанных соперников, не хуже выброшенного по всем четырём "
              f"величинам: медиана {np.median(dominated):.0%}; сожалений, где так со всеми: "
              f"{np.mean(dominated >= 1.0):.0%}")
        learnt = frontier.fit_correction(data)
        stand = frontier.base(learnt.scale)
        controls = [frontier.permuted(learnt, 0), frontier.random_like(learnt, 0)]
        print("  веса (биты программы, биты ошибок, размер, ошибок), и верный порядок пар сожаления:")
        for score in [stand, learnt] + controls:
            print(f"    {score.name:13s} {np.array2string(score.raw(), precision=3)}   "
                  f"{frontier.pair_accuracy(score, data):.1%}")
        arms = {"как есть": None, "по сожалению": learnt, "переставлена": controls[0],
                "случайная": controls[1], "широкий луч": None}
        results = list(pool.map(test_job, [(n, s, arm, score) for arm, score in arms.items()
                                           for n, s in TESTS]))
    print(f"\nвсего {time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сырые данные: {out}")
    for title, tasks in (("T4 — перенос (потолок пространства: не выигрывается)", ("T4",)),
                         ("W0, W3, W4", ("W0", "W3", "W4")),
                         ("T1-T3 на новых данных", TRAIN),
                         ("все семь", ("T4", "W0", "W3", "W4") + TRAIN)):
        rows = [r for r in results if r["task"] in tasks]
        print(f"\n=== {title}: решено при бюджете (из {len(rows) // len(arms)})")
        print("  бюджет          " + "".join(f"{label(b):>7s}" for b in BUDGETS))
        for arm in arms:
            mine = [r for r in rows if r["arm"] == arm]
            print(f"  {arm:14s} " + "".join(f"{sum(r['points'][i]['rule'] for r in mine):>7d}"
                                          for i in range(len(BUDGETS))))


if __name__ == "__main__":
    main()
