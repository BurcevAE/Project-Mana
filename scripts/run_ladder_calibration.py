"""H3, D0: calibrating the ladder of necessary depth, before D1.

docs/ГЛУБИНА_D0.md, 4.3. Deterministic worlds only -- noise is the second
wave. Families R, S, M, rungs 1..5, seeds 0..9; rung 1 is the flat control
C0; C1 is the out-of-catalogue control.

    flat     the search as it stands, its size limit the truth's size + 2,
             budget 400k: solved at each budget
    oracle   each level's question built from the true intermediate result,
             with the catalogue of D1 -- a residual for a sum, the misses and
             then the cases for a condition -- and every piece left solved by
             the flat search, within an equal share of the same 400k. The
             assembled program must be exact on the training points and right
             on every state of the world
    C1       the flat search only: the catalogue has no question for it

What the oracle shows, said plainly: that a tree of pieces the flat search
can solve exists, given the right intermediate results. Not that anything
can find those results -- that is D1's question. For R it shows little: the
addends are given, and only the last one is searched for.

    python -X utf8 scripts/run_ladder_calibration.py [workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import ladder  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import ADD, CONST, GET, IF, Evaluator, add, if_, size  # noqa: E402

BUDGET = 400000
BUDGETS = (25000, 50000, 100000, 200000, 400000)
SEEDS = range(10)


def _leaf(p) -> bool:
    return p[0] in (GET, CONST)


def _case_level(p) -> bool:
    """A level of cases, if(c, leaf, rest); an if of two leaves is a piece."""
    return p[0] == IF and not (_leaf(p[2]) and _leaf(p[3]))


def pieces(truth) -> int:
    """How many problems the oracle leaves to the flat search."""
    if truth[0] == ADD:
        return pieces(truth[2])
    if _case_level(truth):
        return 1 + pieces(truth[3])
    return 1


def levels(truth) -> int:
    if truth[0] == ADD or _case_level(truth):
        return 1 + levels(truth[2] if truth[0] == ADD else truth[3])
    return 1


class Oracle:
    """Decomposes by the truth's own construction, with D1's catalogue."""

    def __init__(self, columns, share: int) -> None:
        self.columns = columns
        self.evaluator = Evaluator(columns)
        self.share = share
        self.spent = 0
        self.failure = ""

    def flat(self, rows: np.ndarray, target: np.ndarray, condition: bool = False):
        """A piece for the flat search. Not exact on its own points: the
        piece failed, and the tree with it. A condition is judged by what
        if() reads of it, whether it is 0 -- not by its value (second
        calibration: the first judged a condition by value)."""
        cols = {v: c[rows] for v, c in self.columns.items()}
        found = P.run(replace(P.CURRENT, budget=self.share), cols, target[rows])
        self.spent += found.evaluations
        out = Evaluator(cols)(found.program)
        if condition:
            out = (out != 0).astype(np.int64)
        wrong = int(np.count_nonzero(out != target[rows]))
        if wrong:
            values = len(set(target[rows].tolist()))
            # On two values a program wrong on most points can be a short
            # description (a wrong guess inside the alphabet names the right
            # one for free): reported apart from a piece merely not found.
            kind = (" — двузначная цель, ответ против цели"
                    if values == 2 and 2 * wrong > int(rows.sum()) else "")
            self.failure = (f"плоский поиск не решил кусок: {wrong} из {int(rows.sum())} "
                            f"точек{kind}")
            return None
        return found.program

    def solve(self, truth, rows: np.ndarray, target: np.ndarray):
        if truth[0] == ADD:                          # residual: the first addend is given
            first = truth[1]
            rest = self.solve(truth[2], rows, target - self.evaluator(first))
            return None if rest is None else add(first, rest)
        if _case_level(truth):                       # misses of the leaf, then the cases
            leaf = truth[2]
            ours = self.evaluator(leaf)
            misses = rows & (target != ours)
            if not misses.any():
                # The leaf is right on every point of this problem: the level is
                # solved by it (the first calibration gave up here).
                return leaf
            rest = self.solve(truth[3], misses, target)
            if rest is None:
                return None
            theirs = self.evaluator(rest)
            neither = rows & (target != ours) & (target != theirs)
            if neither.any():
                self.failure = f"ни одна ветвь не права в {int(neither.sum())} точках"
                return None
            told = rows & (ours != theirs)
            condition = self.flat(told, (target == ours).astype(np.int64), condition=True)
            return None if condition is None else if_(condition, leaf, rest)
        return self.flat(rows, target)


def flat_job(job):
    family, rung, seed = job
    world = ladder.world(family, rung, seed)
    split = world.split(200, 300, seed)
    policy = replace(P.CURRENT, max_size=size(world.truth) + 2, budget=BUDGET)
    found = P.run(policy, split.train, split.train_outcomes, profile=True)
    solved = []
    for b in BUDGETS:
        program, _, _ = discovery.at_budget(found, b)
        solved.append(world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0)
    return family, rung, seed, solved


def oracle_job(job):
    family, rung, seed = job
    world = ladder.world(family, rung, seed)
    split = world.split(200, 300, seed)
    target = np.asarray(split.train_outcomes, dtype=np.int64)
    oracle = Oracle(split.train, BUDGET // pieces(world.truth))
    program = oracle.solve(world.truth, np.ones(len(target), dtype=bool), target)
    exact = program is not None and bool(np.array_equal(Evaluator(split.train)(program), target))
    right = exact and world.grade(lambda cols: discovery.predict(program, cols)) == 1.0
    return family, rung, seed, right, exact, oracle.spent, pieces(world.truth), oracle.failure


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    ladder_jobs = [(f, k, s) for f in ladder.FAMILIES for k in ladder.RUNGS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        flats = list(pool.map(flat_job, ladder_jobs + [("C1", 1, s) for s in SEEDS]))
        oracles = list(pool.map(oracle_job, ladder_jobs))
    print(f"всего {time.time() - started:.0f}с\n")
    print("  ступень  размер  уровней  куски   плоский: решено при бюджете (из 10)"
          "                оракул (из 10)   вычислений оракула (мед)")
    print("                                    " + "".join(f"{b // 1000:>6d}k" for b in BUDGETS))
    for family in ladder.FAMILIES:
        role = "лестница" if family in ladder.LADDER else "контроль: разложение, родное для плоского поиска"
        print(f"  --- {family}: {role}")
        for rung in ladder.RUNGS:
            truth = ladder.truth(family, rung, 0)
            flat = [r for r in flats if r[:2] == (family, rung)]
            orc = [r for r in oracles if r[:2] == (family, rung)]
            line = "".join(f"{sum(r[3][i] for r in flat):>7d}" for i in range(len(BUDGETS)))
            spent = sorted(r[5] for r in orc)[len(orc) // 2]
            sizes = sorted(size(ladder.truth(family, rung, s)) for s in SEEDS)
            print(f"  {family}{rung}      {sizes[0]:>2d}-{sizes[-1]:<2d}  {levels(truth):>5d}  "
                  f"{pieces(truth):>5d}  {line}   {sum(r[3] for r in orc):>10d}   {spent:>12d}")
            for r in orc:
                if not r[3]:
                    print(f"      оракул не решил: сид {r[2]}, точен на обучении {r[4]}; {r[7]}")
    c1 = [r for r in flats if r[0] == "C1"]
    print("\n  C1 (вне каталога), плоский: " +
          "".join(f"{sum(r[3][i] for r in c1):>7d}" for i in range(len(BUDGETS))))


if __name__ == "__main__":
    main()
