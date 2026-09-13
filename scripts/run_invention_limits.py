"""Where step 2's invention breaks: three worlds beyond the one it was built on.

    W2n   W2 with one outcome in ten random
    W2k   a hidden counter of three states choosing among x, y and z --
          one binary variable cannot hold it
    W2p   a hidden number, presses mod 4, added to x -- a quantity that
          acts by arithmetic, not by choosing

The operation is not changed for them. What is measured is whether the
language changes, and what that buys on new episodes against the rule
itself (the clean outcome), the same search without the variable, a
decision tree and a table.

    python scripts/run_invention_limits.py [W2n,W2k,W2p] [seeds]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import baselines, invent  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.worlds import SEQUENCE_WORLDS  # noqa: E402

NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["W2n", "W2k", "W2p"]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def flat(columns):
    return {name: np.asarray(values).reshape(-1) for name, values in columns.items()}


def run(world, seed):
    train, test = world.split(20, 50, seed)
    started = time.time()
    found = invent.invent(train.columns, train.outcomes)
    truth = test.clean.reshape(-1)
    base = float(np.mean(discovery.predict(found.base.program, flat(test.columns)) == truth))
    new = float(np.mean(invent.predict(found, test.columns).reshape(-1) == truth))
    tree, _ = baselines.tree_predict(flat(train.columns), train.outcomes.reshape(-1),
                                     flat(test.columns), seed)
    table = baselines.table_predict(flat(train.columns), train.outcomes.reshape(-1),
                                    flat(test.columns))
    return {"found": found, "base": base, "new": new,
            "tree": float(np.mean(tree == truth)), "table": float(np.mean(table == truth)),
            "seconds": time.time() - started}


def main() -> None:
    for name in NAMES:
        world = SEQUENCE_WORLDS[name]
        print(f"\n=== {world.name}: {world.note}")
        rows = [run(world, seed) for seed in range(SEEDS)]
        for seed, row in enumerate(rows):
            print(f"  сид {seed}: {row['found'].describe()}")
            print(f"         без переменной {row['base']:.3f} -> {row['new']:.3f}; "
                  f"размечено {row['found'].labelled:.0%}; {row['seconds']:.1f}с")

        def mean(key):
            return float(np.mean([row[key] for row in rows]))

        changed = sum(row["found"].accepted for row in rows)
        print(f"  язык изменён: {changed} из {len(rows)}")
        print(f"  точность на новых эпизодах против правды: без переменной "
              f"{mean('base'):.3f}, итог {mean('new'):.3f}, дерево {mean('tree'):.3f}, "
              f"таблица {mean('table'):.3f}")


if __name__ == "__main__":
    main()
