"""Discovery, step 6 control: is it the frontier that throws the path away?

Step 6 found T4's rule with a distance at 66 bits -- expressible, and far
shorter than anything the search reached -- and the learnt move one step
from if(x < 3, z, y), a program the ranking puts below a bare leaf. Two
explanations, and this tells them apart:

    H1  the measure is in the way: the answer is reachable, and the beam,
        ranked by bits, drops the state it is reached from
    H2  the measure is not: T4 needs something the language or the edits
        lack, and no state kept would change that

The measure is left alone. The search is only allowed to keep one given
program in its beam, beside the best -- if(x < 3, z, y), chosen because
the answer is known: a diagnostic, not a method. Four arms on T4, seeds
0..9, beam 4, 800k programs:

    as it stands       nothing kept, no move
    move               the move learnt in step 6
    kept               the program kept, no move
    kept + move        both

and where the kept program stands among everything the first round made.

    python scripts/run_frontier.py [workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import description  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import (LESS, Evaluator, cmp, const, get, hole, if_,  # noqa: E402
                                     show, size, sub)
from mana.discovery.worlds import FAMILY  # noqa: E402

T4 = FAMILY["T4"]
X, Y, Z = get("x"), get("y"), get("z")
KEPT = if_(cmp(LESS, X, const(3)), Z, Y)
RULE = if_(cmp(LESS, if_(cmp(LESS, Y, X), sub(X, Y), sub(Y, X)), const(3)), Z, Y)
#: The move step 6 learnt (derive.json), written out: n -> |n - #0|, both ways round.
SELF = discovery.SELF
MOVES = [discovery.Macro("m13", if_(cmp(LESS, hole(0), SELF), sub(SELF, hole(0)),
                                    sub(hole(0), SELF)), 1),
         discovery.Macro("m14", if_(cmp(LESS, SELF, hole(0)), sub(hole(0), SELF),
                                    sub(SELF, hole(0))), 1)]
ARMS = {"как есть": {}, "ход": {"macros": MOVES}, "удержан": {"pinned": [KEPT]},
        "удержан + ход": {"pinned": [KEPT], "macros": MOVES}}
BUDGET = 800000


def _bits(split, p):
    evaluator = Evaluator(split.train)
    actual = np.asarray(split.train_outcomes)
    return (description.program_bits(p, 3)
            + description.error_bits(evaluator(p), actual, description.alphabet_of(actual),
                                     description.membership(actual)))


def _first_round_rank(split):
    """Where the kept program stands among everything round 1 makes from
    the starting beam, by the search's own ranking."""
    leaves, conditions = discovery.vocabulary(split.train, split.train_outcomes)
    beam = sorted(leaves, key=lambda p: (_bits(split, p), size(p), show(p)))[:discovery.BEAM]
    made = {q for p in beam for q in discovery.neighbours(p, leaves, conditions,
                                                          discovery.MAX_SIZE)}
    scored = sorted((_bits(split, q), size(q), show(q)) for q in made)
    target = (_bits(split, KEPT), size(KEPT), show(KEPT))
    return (sum(1 for row in scored if row < target) + 1, len(scored)) if KEPT in made else (None, len(scored))


def job(args):
    seed, arm = args
    split = T4.split(200, 300, seed)
    started = time.time()
    found = discovery.search(split.train, split.train_outcomes, budget=BUDGET, trace=True,
                             **ARMS[arm])
    rule = T4.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    out = {"seed": seed, "arm": arm, "rule": rule, "found_at": found.found_at,
           "bits": found.bits, "program": show(found.program),
           "macro_steps": found.macro_steps, "evaluations": found.evaluations,
           "derivation": " -> ".join(show(p) for p in found.derivation),
           "seconds": time.time() - started}
    if arm == "как есть":
        out["rank"] = _first_round_rank(split)
        out["kept_bits"], out["rule_bits"], out["leaf_bits"] = (
            _bits(split, KEPT), _bits(split, RULE), _bits(split, Y))
    return out


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(job, [(seed, arm) for arm in ARMS for seed in range(10)]))
    print(f"{time.time() - started:.0f}с")
    print(f"\nправило с расстоянием: {show(RULE)}, {size(RULE)} узлов; удерживается {show(KEPT)}")
    print("\n  плечо            решено   программ до ответа (мед)   ответ через ход   биты ответа (мед)")
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        solved = [r for r in mine if r["rule"]]
        cost = int(np.median([r["found_at"] for r in solved])) if solved else 0
        print(f"  {arm:15s}  {len(solved):>3d}/10   {cost:>24d}   "
              f"{sum(r['macro_steps'] > 0 for r in solved):>8d}/{len(solved):<2d}"
              f"   {np.median([r['bits'] for r in mine]):>14.1f}")
    print("\n  сид  биты: правило  удержанный  лист y   место удержанного в раунде 1   как есть / удержан + ход")
    base = {r["seed"]: r for r in rows if r["arm"] == "как есть"}
    both = {r["seed"]: r for r in rows if r["arm"] == "удержан + ход"}
    for seed in range(10):
        b, k = base[seed], both[seed]
        rank, total = b["rank"]
        place = f"{rank} из {total}" if rank else f"не среди {total}"
        print(f"  {seed:>3d}  {b['rule_bits']:>13.1f}  {b['kept_bits']:>10.1f}  {b['leaf_bits']:>7.1f}"
              f"   {place:>28s}   {'да' if b['rule'] else 'нет'} / {'да' if k['rule'] else 'нет'}")
    print("\n  выводы плеча «удержан + ход» (сиды 0-2):")
    for seed in range(3):
        print(f"    {seed}: {both[seed]['derivation']}")
    kept_only = [r for r in rows if r["arm"] == "удержан"]
    print("\n  плечо «удержан» без хода, ответы (сиды 0-4):")
    for r in kept_only[:5]:
        print(f"    {r['seed']}: {'правило' if r['rule'] else 'нет'}  {r['program']}")


if __name__ == "__main__":
    main()
