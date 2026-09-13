"""Discovery, step 4: how reliable is the search everything else stands on?

Nothing about the search is changed here; it is measured.

For each question of a fixed set:
    certificate   the world's rule, written in the same language: does it
                  reproduce every state, and does it fit within the size the
                  search explores? If so, a solution exists in the searched
                  space, and any miss is the search's.
    ceiling       bottom-up enumeration on the training data: the smallest
                  exact program, or a lower bound where the enumeration stops
For each seed:
    the search as it is, and -- for every miss -- a strong search (a wider
    beam, a larger budget) to tell a budget failure from a structural one.

Misses are sorted into:
    FITS_NOT_RULE  exact on the training points, wrong elsewhere
    BUDGET_FAIL    stopped by the budget with its neighbourhood unexhausted
    FALSE_FAIL     stopped at a local end although a solution exists

    python scripts/run_search_reliability.py [seeds]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import ceiling  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import show, size  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TASKS = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")
WORLD = {**WORLDS, **FAMILY}
STRONG = {"beam_width": 16, "budget": 2000000, "patience": 3, "max_rounds": 60}


def classify(row):
    if row["rule"]:
        return "SUCCESS"
    if row["found"].status == discovery.FOUND_EXACT:
        return "FITS_NOT_RULE"
    if row["found"].termination == discovery.SEARCH_LIMIT:
        return "BUDGET_FAIL"
    return "FALSE_FAIL"


def attempt(world, split, **options):
    found = discovery.search(split.train, split.train_outcomes, **options)
    rule = world.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    unseen = float(np.mean(discovery.predict(found.program, split.test) == split.test_truth))
    return {"found": found, "rule": rule, "unseen": unseen}


def main() -> None:
    overall = {"runs": 0, "success": 0, "rescued": 0}
    summary = []
    for name in TASKS:
        world = WORLD[name]
        reproduces = world.grade(lambda cols: discovery.predict(world.truth, cols)) == 1.0
        fits = size(world.truth) <= discovery.MAX_SIZE
        first = world.split(200, 300, 0)
        clean = world.rule_on(first.train)
        leaves, _ = discovery.vocabulary(first.train, clean)
        limit = ceiling.smallest_fit(first.train, clean, leaves)
        print(f"\n=== {name}: {world.note}")
        print(f"  сертификат: правило в том же языке, {size(world.truth)} узлов, "
              f"воспроизводит все состояния: {'да' if reproduces else 'НЕТ'}; "
              f"в пределах размера поиска ({discovery.MAX_SIZE}): {'да' if fits else 'НЕТ'}")
        print(f"  потолок: {limit.describe()} ({limit.seconds:.0f}с)")
        rows = []
        for seed in range(SEEDS):
            split = world.split(200, 300, seed)
            row = attempt(world, split)
            row["class"] = classify(row)
            if row["class"] != "SUCCESS":
                strong = attempt(world, split, **STRONG)
                row["strong"] = strong
            rows.append(row)
            found = row["found"]
            tail = ""
            if "strong" in row:
                s = row["strong"]
                tail = (f" | сильный: {'спас' if s['rule'] else 'не спас'} "
                        f"({s['found'].termination}, {s['found'].evaluations}, "
                        f"{s['found'].seconds:.0f}с)")
            print(f"  сид {seed}: {row['class']:13s} {found.termination:16s} "
                  f"{found.status:12s} ошибок на обучении {found.train_errors:3d}, "
                  f"на новых {1 - row['unseen']:.3f}, {found.bits:6.1f} бит, "
                  f"найдено на {found.found_at:>6d} из {found.evaluations:>6d}, "
                  f"{found.seconds:4.1f}с{tail}")
            print(f"         {show(found.program)[:110]}")
        success = sum(r["class"] == "SUCCESS" for r in rows)
        rescued = sum(1 for r in rows if "strong" in r and r["strong"]["rule"])
        classes = {c: sum(r["class"] == c for r in rows)
                   for c in ("FITS_NOT_RULE", "BUDGET_FAIL", "FALSE_FAIL")}
        unseen = [r["unseen"] for r in rows]
        found_at = [r["found"].found_at for r in rows if r["rule"]]
        overall["runs"] += len(rows)
        overall["success"] += success
        overall["rescued"] += rescued
        summary.append((name, success, rescued, classes, np.mean(unseen), np.std(unseen),
                        int(np.median(found_at)) if found_at else None, limit))
    print("\n=== итог")
    print(f"  {'вопрос':6s} {'найдено':>8s} {'+сильный':>9s}  "
          f"{'FITS_NOT_RULE':>13s} {'BUDGET':>7s} {'FALSE':>6s}  "
          f"{'на новых (ср ± откл)':>22s}  {'найдено на (мед.)':>17s}")
    for name, success, rescued, classes, mean, std, median, limit in summary:
        print(f"  {name:6s} {success:>5d}/{SEEDS} {rescued:>9d}  {classes['FITS_NOT_RULE']:>13d} "
              f"{classes['BUDGET_FAIL']:>7d} {classes['FALSE_FAIL']:>6d}  "
              f"{mean:>14.3f} ± {std:.3f}  {str(median):>17s}")
    runs = overall["runs"]
    print(f"\n  решение существует во всех {runs} прогонах (сертификат в пределах размера поиска)")
    print(f"  P(найдено | существует): поиск как есть {overall['success']}/{runs} = "
          f"{overall['success'] / runs:.0%}; с сильным повтором провалов "
          f"{overall['success'] + overall['rescued']}/{runs} = "
          f"{(overall['success'] + overall['rescued']) / runs:.0%}")


if __name__ == "__main__":
    main()
