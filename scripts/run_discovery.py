"""Discovery, step 1: does the search find the rule, and does it resist noise?

For each world and seed: a split into training states and held-out states
the search never saw; the search's program; and the floor it has to clear.

    найдено      the search's program, its bits, whether it IS the rule
                 (agreement with the truth over every state of the world),
                 accuracy on held-out states
    таблица      remember the training states; unseen -> most common outcome
    дерево       a decision tree on the same variables
    правда       the world's own rule, written in the same language

    python scripts/run_discovery.py [W0,W3] [seeds] [n_train] [n_test]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import baselines, description  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import Evaluator, show, size  # noqa: E402
from mana.discovery.worlds import WORLDS  # noqa: E402

NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["W0", "W3"]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_TRAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_TEST = int(sys.argv[4]) if len(sys.argv) > 4 else 300


def run(world, seed):
    split = world.split(N_TRAIN, N_TEST, seed)
    started = time.time()
    found = discovery.search(split.train, split.train_outcomes)
    seconds = time.time() - started

    def accuracy(predicted):
        return float(np.mean(predicted == split.test_outcomes))

    alphabet = description.alphabet_of(split.train_outcomes)
    on_train = Evaluator(split.train)
    variables = len(world.variables)
    tree, tree_nodes = baselines.tree_predict(split.train, split.train_outcomes,
                                              split.test, seed)
    return {
        "seed": seed, "program": show(found.program), "size": size(found.program),
        "bits": found.bits, "evaluations": found.evaluations, "seconds": seconds,
        "is_rule": world.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0,
        "graded": world.grade(lambda cols: discovery.predict(found.program, cols)),
        "found": accuracy(discovery.predict(found.program, split.test)),
        "truth_bits": (description.program_bits(world.truth, variables)
                       + description.error_bits(on_train(world.truth),
                                                split.train_outcomes, alphabet)),
        "truth": accuracy(discovery.predict(world.truth, split.test)),
        "table_bits": description.table_bits(N_TRAIN, variables, world.values, alphabet),
        "table": accuracy(baselines.table_predict(split.train, split.train_outcomes,
                                                  split.test)),
        "tree": accuracy(tree), "tree_nodes": tree_nodes,
    }


def main() -> None:
    print(f"обучение {N_TRAIN} состояний, отложено {N_TEST} других, сидов {SEEDS}")
    for name in NAMES:
        world = WORLDS[name]
        print(f"\n=== {world.name}: {world.note}")
        rows = [run(world, seed) for seed in range(SEEDS)]
        for row in rows:
            print(f"  сид {row['seed']}: {row['program']}  {row['bits']:.1f} бит, "
                  f"правило {'да' if row['is_rule'] else 'нет'} ({row['graded']:.3f}), "
                  f"отложенные {row['found']:.3f}, перебрано {row['evaluations']}, "
                  f"{row['seconds']:.1f}с")

        def mean(key):
            return float(np.mean([row[key] for row in rows]))

        print(f"  нашла само правило: {sum(r['is_rule'] for r in rows)} из {len(rows)}")
        print(f"  биты: найдено {mean('bits'):.1f}, правда {mean('truth_bits'):.1f}, "
              f"таблица {mean('table_bits'):.1f}")
        print(f"  точность на отложенных: найдено {mean('found'):.3f}, "
              f"дерево {mean('tree'):.3f} ({mean('tree_nodes'):.0f} узлов), "
              f"таблица {mean('table'):.3f}, правда {mean('truth'):.3f}")
        print(f"  поиск: {mean('evaluations'):.0f} программ, {mean('seconds'):.1f}с")


if __name__ == "__main__":
    main()
