"""The research-question contract's gate, before any probe
(docs/ГЛУБИНА_ВОПРОС_ИССЛЕДОВАНИЯ.md, 7). Checks 1, 2 and 4 are tests
(tests/test_questions.py); this is check 3 at full size.

    zero   no plan: questions.solve against the flat search (policy.run),
           on the ladder and its controls (R, S, M rungs 1..5 and C1, seeds
           0..9, size limit truth + 2) and the classic questions (W0, W3,
           T1..T4, W4, seeds 0..9), 400k -- every field of the flat search,
           the whole ledger under "search", one node, no record
    d1b    D1b's plans through the contract against problems.solve (D1b), on
           the same ladder instances at flat shares 1, 1/2, 1/4, 400k --
           answer, every field of the root's search, the ledger, the whole
           tree (operator, parent, dependencies, budget, cost, program,
           chosen, used, children) and the depth used

    python -X utf8 scripts/run_questions_gate.py [workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import ladder, plans, problems, questions  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery.language import show, size  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

BUDGET = 400000
SEEDS = range(10)
CLASSIC = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")
SHARES = (1.0, 0.5, 0.25)
FIELDS = ("program", "bits", "program_bits", "error_bits", "evaluations", "rounds",
          "history", "found_at", "termination", "train_errors", "anytime")


def _instance(group, name, rung, seed):
    if group == "ladder":
        world = ladder.world(name, rung, seed)
        return world.split(200, 300, seed), replace(P.CURRENT, max_size=size(world.truth) + 2)
    return {**WORLDS, **FAMILY}[name].split(200, 300, seed), P.CURRENT


def _tree(solution):
    return [(n.problem.operator, n.problem.parent, n.problem.depends_on, n.budget,
             dict(n.cost), show(n.program), n.chosen, n.used, tuple(n.children))
            for n in solution.nodes]


def zero_job(args):
    split, policy = _instance(*args)
    flat = P.run(P.with_budget(policy, BUDGET), split.train, split.train_outcomes, profile=True)
    mine = questions.solve(problems.Problem.whole(split.train, split.train_outcomes), policy,
                           BUDGET, profile=True)
    differ = [f for f in FIELDS if getattr(flat, f) != getattr(mine.found, f)]
    if mine.program != flat.program:
        differ.append("answer")
    if mine.ledger.spent[problems.SEARCH] != flat.evaluations or \
            sum(mine.ledger.spent.values()) != flat.evaluations:
        differ.append("ledger")
    if len(mine.nodes) != 1 or mine.records:
        differ.append("tree")
    return args, differ


def d1b_job(args):
    *inst, share = args
    split, policy = _instance(*inst)
    whole = problems.Problem.whole(split.train, split.train_outcomes)
    d1b = problems.solve(whole, policy, BUDGET, catalogue=problems.BRANCHES, flat_share=share,
                         profile=True)
    whole = problems.Problem.whole(split.train, split.train_outcomes)
    mine = questions.solve(whole, policy, BUDGET, plans=plans.D1B, flat_share=share,
                           profile=True)
    differ = [f for f in FIELDS if getattr(d1b.found, f) != getattr(mine.found, f)]
    if mine.program != d1b.program:
        differ.append("answer")
    if mine.ledger.spent != d1b.ledger.spent:
        differ.append("ledger")
    if _tree(mine) != _tree(d1b):
        differ.append("tree")
    if mine.used_depth() != d1b.used_depth():
        differ.append("depth")
    return args, differ, len(d1b.nodes)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    lad = [("ladder", f, k, s) for f in ladder.FAMILIES for k in ladder.RUNGS for s in SEEDS]
    lad += [("ladder", "C1", 1, s) for s in SEEDS]
    classic = [("classic", n, 0, s) for n in CLASSIC for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        zero = list(pool.map(zero_job, lad + classic))
        d1b = list(pool.map(d1b_job, [i + (sh,) for i in lad for sh in SHARES]))
    print(f"всего {time.time() - started:.0f}с\n")
    print(f"  нулевой случай (нет планов = плоский поиск): совпали во всех полях "
          f"{sum(1 for _, d in zero if not d)} из {len(zero)}")
    for args, d in zero:
        if d:
            print(f"    {args}: расходится {d}")
    for share in SHARES:
        mine = [r for r in d1b if r[0][-1] == share]
        trees = sorted(r[2] for r in mine)
        print(f"  D1b через контракт, доля {share}: совпали во всех полях "
              f"{sum(1 for _, d, _ in mine if not d)} из {len(mine)}; узлов в дереве "
              f"мед {trees[len(trees) // 2]}, макс {trees[-1]}")
        for args, d, _ in mine:
            if d:
                print(f"    {args}: расходится {d}")


if __name__ == "__main__":
    main()
