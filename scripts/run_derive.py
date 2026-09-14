"""Discovery, step 6: moves of the search, learnt from how answers were reached.

Stage 1  family A (T1, T2, T3), seeds 0..9, a wide beam (16) and 2M: every
         solved answer's derivation.
Stage 2  moves compressed from those derivations (derive.py); words
         compressed from the same answers (library.py, step 3); and a
         control: random moves of the same size and arity as the learnt
         ones, to tell the use of a particular move from a wider
         neighbourhood.
Stage 3  the search as it stands (beam 4) with each, at equal programs
         evaluated, read off one profiled run per (question, seed, arm):
             B, transfer    T4, seeds 0..9 -- another surface, the
                            search's structural limit
             A, in-family   T1, T2, T3, seeds 10..19 -- unseen data
         For each solve, whether a move took a step of its derivation.

    python scripts/run_derive.py [workers] [out.json]
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

from mana.discovery import derive, library  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import (EQUAL, LESS, add, cmp, hole, if_, nodes,  # noqa: E402
                                     show, size, sub)
from mana.discovery.worlds import FAMILY  # noqa: E402

TRAIN = ("T1", "T2", "T3")
BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000, 800000)
TEST_BUDGET = max(BUDGETS)
#: Step 5.0, T4 alone: solved of 10 at each budget, beam 4 and beam 16.
T4_BEAM4 = (0, 0, 0, 0, 4, 4, 4)
T4_BEAM16 = (0, 0, 0, 0, 0, 0, 6)


def label(budget):
    return f"{budget // 1000}k"


def learn_job(job):
    name, seed = job
    world = FAMILY[name]
    split = world.split(200, 300, seed)
    found = discovery.search(split.train, split.train_outcomes, beam_width=16,
                             budget=2000000, trace=True)
    solved = world.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    leaves, conditions = discovery.vocabulary(split.train, split.train_outcomes)
    return {"task": name, "seed": seed, "solved": solved, "program": found.program,
            "derivation": found.derivation, "leaves": leaves, "conditions": conditions}


def _random_moves(like, seed):
    """Moves of the same size and arity as the learnt ones, of no origin."""
    rng = np.random.default_rng([seed, 23])
    leaves = [discovery.SELF, hole(0)]

    def grow(depth):
        if depth == 0 or rng.random() < 0.3:
            return leaves[int(rng.integers(0, 2))]
        kind = int(rng.integers(0, 4))
        if kind == 0:
            return add(grow(depth - 1), grow(depth - 1))
        if kind == 1:
            return sub(grow(depth - 1), grow(depth - 1))
        if kind == 2:
            return cmp(LESS if rng.random() < 0.5 else EQUAL, grow(depth - 1), grow(depth - 1))
        return if_(grow(depth - 1), grow(depth - 1), grow(depth - 1))

    out = []
    for i, move in enumerate(like):
        while True:
            template = grow(3)
            has_self = any(part == discovery.SELF for _, part in nodes(template))
            has_hole = any(part == hole(0) for _, part in nodes(template))
            if (size(template) == size(move.template) and has_self
                    and has_hole == (move.arity == 1)):
                out.append(discovery.Macro(f"r{i + 1}", template, move.arity))
                break
    return out


def test_job(job):
    name, seed, arm, payload = job
    world = FAMILY[name]
    split = world.split(200, 300, seed)
    kwargs = {"budget": TEST_BUDGET, "profile": True, "trace": True}
    if arm in ("moves", "control"):
        kwargs["macros"] = payload
    elif arm == "words":
        kwargs["library"] = payload
    started = time.time()
    found = discovery.search(split.train, split.train_outcomes, **kwargs)
    lib = payload if arm == "words" else None
    points = []
    for budget in BUDGETS:
        program, found_at, _ = discovery.at_budget(found, budget)
        rule = world.grade(lambda cols, p=program: discovery.predict(p, cols, lib)) == 1.0
        points.append({"budget": budget, "rule": rule, "found_at": found_at})
    uses_word = any(part[0] == "prim" for _, part in nodes(found.program))
    return {"task": name, "seed": seed, "arm": arm, "points": points,
            "evaluations": found.evaluations, "macro_steps": found.macro_steps,
            "uses_word": uses_word, "program": show(found.program),
            "seconds": time.time() - started}


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        learnt = list(pool.map(learn_job, [(n, s) for n in TRAIN for s in range(10)]))
        solved = [r for r in learnt if r["solved"]]
        print(f"стадия 1: решено {len(solved)}/{len(learnt)} "
              f"({', '.join(f'{n} {sum(r['solved'] for r in learnt if r['task'] == n)}/10' for n in TRAIN)}), "
              f"шагов в выводе (медиана) {int(np.median([len(r['derivation']) - 1 for r in solved]))}, "
              f"{time.time() - started:.0f}с", flush=True)
        derivations = [derive.Derivation(r["derivation"], r["leaves"], r["conditions"],
                                         discovery.MAX_SIZE, r["task"]) for r in solved]
        moves = derive.compress(derivations, 3)
        words = library.compress([r["program"] for r in solved], 3)
        control = _random_moves(moves.macros, 0)
        print(f"стадия 2: ходов рассмотрено {moves.considered}, принято {len(moves.macros)}; "
              f"выводы {moves.bits_before:.1f} -> {moves.bits_after:.1f} бит")
        for name, template, saved, uses in moves.kept:
            print(f"    ход   {name}: n -> {template.replace('@self', 'n')}  (-{saved:.1f} бит, шагов {uses})")
        for name, template, saved in words.kept:
            print(f"    слово {name}: {template}  (-{saved:.1f} бит)")
        for move in control:
            print(f"    контроль {move.name}: n -> {show(move.template).replace('@self', 'n')}")
        arms = {"base": None, "moves": moves.macros, "words": words.library, "control": control}
        jobs = [("T4", s, arm, payload) for arm, payload in arms.items() for s in range(10)]
        jobs += [(n, s, arm, payload) for arm, payload in arms.items()
                 for n in TRAIN for s in range(10, 20)]
        results = list(pool.map(test_job, jobs))
    print(f"\nвсего {time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps({"moves": moves.kept, "words": words.kept,
                                   "control": [show(m.template) for m in control],
                                   "results": results}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"сырые данные: {out}")
    report(results)


def report(results):
    for title, tasks in (("B, перенос: T4, сиды 0-9", ("T4",)),
                         ("A, своё семейство на новых данных: T1-T3, сиды 10-19", TRAIN)):
        rows = [r for r in results if r["task"] in tasks]
        print(f"\n=== {title}: решено при бюджете (из {len(rows) // 4})")
        print("  бюджет     " + "".join(f"{label(b):>7s}" for b in BUDGETS)
              + "   программ до ответа (мед)   ход в выводе / слово в ответе")
        for arm in ("base", "moves", "words", "control"):
            mine = [r for r in rows if r["arm"] == arm]
            solved = "".join(f"{sum(r['points'][i]['rule'] for r in mine):>7d}"
                             for i in range(len(BUDGETS)))
            last = [r for r in mine if r["points"][-1]["rule"]]
            cost = int(np.median([r["points"][-1]["found_at"] for r in last])) if last else 0
            used = sum(r["macro_steps"] > 0 for r in last) if arm in ("moves", "control") else \
                sum(r["uses_word"] for r in last) if arm == "words" else 0
            print(f"  {arm:9s}  {solved}   {cost:>24d}   {used}/{len(last)}")
        if tasks == ("T4",):
            print("  шаг 5.0   " + "".join(f"{v:>7d}" for v in T4_BEAM4) + "   (луч 4)")
            print("            " + "".join(f"{v:>7d}" for v in T4_BEAM16) + "   (луч 16)")


if __name__ == "__main__":
    main()
