"""H3, D1a: with no question operators, a problem is the flat search.

docs/ГЛУБИНА_D1.md. One hypothesis only -- the new object is transparent
when it asks nothing. Every instance is run twice: the flat search as it
stands (policy.run), and problems.solve with an empty catalogue on the
whole problem. Compared field by field: the answer (graded on the world),
program, bits (program, errors), evaluations, rounds, history, where it
was first seen, how it stopped, training errors, the anytime curve; and
the ledger -- every evaluation under "search", the other articles 0.

    ladder     R, S, M rungs 1..5 and C1, seeds 0..9: size limit the
               truth's size + 2, budget 400k -- as calibrated
    classic    W0, W3, T1..T4, W4, seeds 0..9: the search as it stands

    python -X utf8 scripts/run_d1a.py [workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import ladder, problems  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import size  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

BUDGET = 400000
SEEDS = range(10)
CLASSIC = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")
FIELDS = ("program", "bits", "program_bits", "error_bits", "evaluations", "rounds",
          "history", "found_at", "termination", "train_errors", "anytime")


def instance(job):
    group, name, rung, seed = job
    if group == "ladder":
        world = ladder.world(name, rung, seed)
        policy = replace(P.CURRENT, max_size=size(world.truth) + 2)
    else:
        world = {**WORLDS, **FAMILY}[name]
        policy = P.CURRENT
    return world, world.split(200, 300, seed), policy


def job(args):
    group, name, rung, seed = args
    world, split, policy = instance(args)
    began = time.time()
    flat = P.run(P.with_budget(policy, BUDGET), split.train, split.train_outcomes, profile=True)
    flat_seconds = time.time() - began
    began = time.time()
    solved = problems.solve(problems.Problem.whole(split.train, split.train_outcomes),
                            policy, BUDGET, profile=True)
    solved_seconds = time.time() - began
    differ = [f for f in FIELDS if getattr(flat, f) != getattr(solved.found, f)]
    if solved.program != flat.program:
        differ.append("answer")
    grade = lambda p: world.grade(lambda cols: discovery.predict(p, cols)) == 1.0  # noqa: E731
    if grade(flat.program) != grade(solved.program):
        differ.append("graded answer")
    spent = solved.ledger.spent
    if spent[problems.SEARCH] != flat.evaluations or any(
            spent[a] for a in problems.ARTICLES if a != problems.SEARCH):
        differ.append("ledger")
    return group, name, rung, seed, differ, flat_seconds, solved_seconds


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    jobs = [("ladder", f, k, s) for f in ladder.FAMILIES for k in ladder.RUNGS for s in SEEDS]
    jobs += [("ladder", "C1", 1, s) for s in SEEDS]
    jobs += [("classic", n, 0, s) for n in CLASSIC for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(job, jobs))
    print(f"всего {time.time() - started:.0f}с\n")
    for group in ("ladder", "classic"):
        mine = [r for r in runs if r[0] == group]
        same = [r for r in mine if not r[4]]
        ratio = sorted(r[6] / max(r[5], 1e-9) for r in mine)[len(mine) // 2]
        print(f"  {group}: совпали во всех полях {len(same)} из {len(mine)}; "
              f"время интерпретатора / плоского (мед) {ratio:.2f}")
        for r in mine:
            if r[4]:
                print(f"    {r[1]}{r[2] or ''}/{r[3]}: расходится {r[4]}")


if __name__ == "__main__":
    main()
