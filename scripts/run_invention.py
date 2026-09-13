"""Discovery, step 2: does changing the language find what is hidden?

W2 hides a switch: every a = 1 flips it, and the outcome is x while it is
on, y while it is off. The same search as step 1 cannot write that; the
question is whether inventing a variable can -- and whether the same
currency refuses a variable where there is nothing hidden (W0, W3).

For each seed on W2:
    без переменной   the step-1 search on the same data
    с переменной     invent(): v1's definition, the outcome program, bits
    переключатель    agreement of v1 with the hidden switch on new episodes
                     (or with its negation: which way round is a name)
    дерево, таблица  on what is seen at a step

    python scripts/run_invention.py [seeds] [train_episodes] [test_episodes]
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
from mana.discovery.language import show  # noqa: E402
from mana.discovery.worlds import SEQUENCE_WORLDS, WORLDS  # noqa: E402

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 50


def flat(columns):
    return {name: np.asarray(values).reshape(-1) for name, values in columns.items()}


def w2(seed):
    world = SEQUENCE_WORLDS["W2"]
    train, test = world.split(TRAIN, TEST, seed)
    started = time.time()
    found = invent.invent(train.columns, train.outcomes)
    seconds = time.time() - started
    truth = test.clean.reshape(-1)
    base = float(np.mean(discovery.predict(found.base.program, flat(test.columns)) == truth))
    new = float(np.mean(invent.predict(found, test.columns).reshape(-1) == truth))
    value = invent.hidden(found, test.columns)
    switch = (max(float(np.mean(value == test.hidden)), float(np.mean(value != test.hidden)))
              if value is not None else 0.0)
    tree, _ = baselines.tree_predict(flat(train.columns), train.outcomes.reshape(-1),
                                     flat(test.columns), seed)
    table = baselines.table_predict(flat(train.columns), train.outcomes.reshape(-1),
                                    flat(test.columns))
    return {"found": found, "base": base, "new": new, "switch": switch,
            "tree": float(np.mean(tree == truth)), "table": float(np.mean(table == truth)),
            "seconds": seconds}


def control(name, seed):
    world = WORLDS[name]
    split = world.split(200, 50, seed)
    shaped = {key: values.reshape(10, 20) for key, values in split.train.items()}
    return invent.invent(shaped, split.train_outcomes.reshape(10, 20))


def main() -> None:
    print(f"W2: обучение {TRAIN} эпизодов по 20 шагов, проверка на {TEST} новых; сидов {SEEDS}")
    rows = [w2(seed) for seed in range(SEEDS)]
    for seed, row in enumerate(rows):
        found = row["found"]
        print(f"  сид {seed}: {found.describe()}")
        print(f"         без переменной {row['base']:.3f} -> с переменной {row['new']:.3f}; "
              f"переключатель восстановлен {row['switch']:.3f}; размечено "
              f"{found.labelled:.0%}; {row['seconds']:.1f}с")

    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    accepted = sum(row["found"].accepted for row in rows)
    print(f"\n  язык изменён: {accepted} из {len(rows)}")
    print(f"  точность на новых эпизодах: без переменной {mean('base'):.3f}, "
          f"с переменной {mean('new'):.3f}, дерево {mean('tree'):.3f}, "
          f"таблица {mean('table'):.3f}")
    print(f"  переключатель восстановлен: {mean('switch'):.3f}")
    print(f"  биты: без переменной {np.mean([r['found'].base.bits for r in rows]):.1f}, "
          f"с переменной {np.mean([r['found'].bits for r in rows if r['found'].accepted] or [0]):.1f}")
    print(f"  время: {mean('seconds'):.1f}с на сид")

    print("\nКонтроль: где скрытого нет, язык меняться не должен")
    for name in ("W0", "W3"):
        results = [control(name, seed) for seed in range(SEEDS)]
        changed = sum(r.accepted for r in results)
        print(f"  {name}: язык изменён {changed} из {len(results)}; {results[0].describe()}")


if __name__ == "__main__":
    main()
