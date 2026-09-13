"""Discovery, step 3: solve what you can, grow a word, retry what you could not.

For each seed:
    A   every question of the family searched in the starting language
    C   every program found for T1..T4 compressed: a word is kept only if
        it makes all of them, with its definition, shorter. Nothing tells
        the compression which programs are right -- the bits decide.
    B   every question searched again with the grown language

W4 asks the same kind of question over variables with other names and is
kept out of the compression: it is the transfer. W0 needs no word at all:
it is the control for what a larger language costs.

    python scripts/run_library.py [seeds]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import library as words  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import show  # noqa: E402
from mana.discovery.worlds import FAMILY, W0  # noqa: E402

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
QUESTIONS = ("T1", "T2", "T3", "T4", "W4", "W0")
COMPRESSED = ("T1", "T2", "T3", "T4")
WORLD = {**FAMILY, "W0": W0}


def attempt(name, seed, library=None):
    world = WORLD[name]
    split = world.split(200, 300, seed)
    started = time.time()
    found = discovery.search(split.train, split.train_outcomes, library=library)
    graded = world.grade(lambda cols: discovery.predict(found.program, cols, library))
    return {"found": found, "graded": graded, "seconds": time.time() - started}


def main() -> None:
    totals = {name: {"A": [], "B": []} for name in QUESTIONS}
    for seed in range(SEEDS):
        print(f"\n=== сид {seed}")
        first = {name: attempt(name, seed) for name in QUESTIONS}
        grown = words.compress([first[name]["found"].program for name in COMPRESSED], 3)
        print(f"  слова: {grown.kept or 'ни одно не окупилось'}; "
              f"биты программ {grown.bits_before:.1f} -> {grown.bits_after:.1f}")
        second = {name: attempt(name, seed, grown.library) for name in QUESTIONS}
        for name in QUESTIONS:
            a, b = first[name], second[name]
            totals[name]["A"].append(a)
            totals[name]["B"].append(b)
            print(f"  {name:3s} без: {show(a['found'].program)[:48]:48s} {a['graded']:.3f} "
                  f"на {a['found'].found_at:>6d} | с: {show(b['found'].program)[:34]:34s} "
                  f"{b['graded']:.3f} на {b['found'].found_at:>6d}")

    print("\n=== итог по сидам")
    print(f"  {'вопрос':6s} | {'решено без':>10s} {'решено с':>9s} | "
          f"{'найдено на (медиана) без':>24s} {'с':>8s}")
    for name in QUESTIONS:
        a, b = totals[name]["A"], totals[name]["B"]
        solved_a = sum(row["graded"] == 1.0 for row in a)
        solved_b = sum(row["graded"] == 1.0 for row in b)
        cost_a = int(np.median([row["found"].found_at for row in a]))
        cost_b = int(np.median([row["found"].found_at for row in b]))
        print(f"  {name:6s} | {solved_a:>7d}/{len(a)} {solved_b:>6d}/{len(b)} | "
              f"{cost_a:>24d} {cost_b:>8d}")


if __name__ == "__main__":
    main()
