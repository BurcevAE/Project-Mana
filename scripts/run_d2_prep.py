"""H3, D2-prep: is there material for subgoals in the flat search's own trace?

docs/ГЛУБИНА_D2.md, declared (commit 60b1034) before this was written. A
diagnostic: nothing is learnt, nothing claimed. The truth is read only to
grade a route -- by the instrument, never by MANA.

    phase 1  the flat search as it stands on R3-R5, M3, M4, seeds 40..49,
             400k, size limit truth + 2, rounds logged. Every program held in
             a beam is an intermediate state X (30 at most, evenly by the
             order first held); its features as known when first held; the
             cost of having it: the search's evaluations to the end of that
             round
    phase 2  from each X, D0's three branches, each subproblem solved by the
             flat search within 100k (the same size limit as the root, as in
             D1b): size after X (bits), cost to the first exact program, the
             route assembled and graded, and a separate 2k probe
    support  a route that solves the root at a cost of at most 400k, on an
             instance the flat search did not solve (on one it did: at most
             half the flat search's cost)
    reduced  size after X at most half the root's size

D2-prep-1b (declared in commit 85df320): the same, with X taken from what
the search evaluated and did not hold -- `pool` as the second argument.
Per round the pool is sorted by the search's own order, every program ever
held in a beam left out; 4 near X at ranks 1, 9, 17, 25 of the rest, 4 far
X at random from the rest, seeded by the instance; at most 60, rounds
taken evenly. "Place in the beam" is then the place in the round's order.

D2-prep-1c (declared in docs/ГЛУБИНА_D2.md before this was written): X from
one exploratory step of P2's class per round -- `look` as the second
argument. The first 256 states of the round's pool, in the search's own
order, are expanded through A' (rebuilt from R2's experience; the run stops
if the change is not R2's); the result of looking at a state is its best
successor. X: those results the flat search never evaluated -- 4 near, at
places 1, 9, 17, 25 by description, 4 far at random; at most 60. The cost
of having X: the flat search to the end of the round plus that round's
step. An instance has a reproducible support when its supports come from
looks at two different states at least.

    python -X utf8 scripts/run_d2_prep.py [workers] [beam | pool | look]
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mana.discovery import description, ladder  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import Evaluator, add, if_, show, size, sub  # noqa: E402

BUDGET, SUB, PROBE, MAX_X = 400000, 100000, 2000, 30
#: D2-prep-1b: X from the pools.
POOL_MAX, NEAR_RANKS, FAR = 60, (0, 8, 16, 24), 4
#: D2-prep-1c: the exploratory step looks at the first 256 states of a round.
SHORTLIST = 256
#: Instances where the trace (beam or pool) already had a support.
TRACE_SUPPORTS = {("R", 3, 44), ("M", 3, 40), ("M", 3, 47), ("M", 4, 48)}
RUNGS = [("R", 3), ("R", 4), ("R", 5), ("M", 3), ("M", 4)]
SEEDS = range(40, 50)
RESIDUAL, RESIDUAL_MINUS, CASES = "остаток", "остаток-", "промахи и случаи"
BRANCHES = (RESIDUAL, RESIDUAL_MINUS, CASES)
#: Free features: smaller first, but for the share of zeros, larger first.
FREE = ("биты X", "биты программы X", "биты ошибок X", "неверных точек X", "размер X",
        "раунд", "место в луче", "значений в цели", "доля нулей в цели", "биты таблицы цели")
LARGER_FIRST = {"доля нулей в цели"}


def _setup(inst):
    family, rung, seed = inst
    world = ladder.world(family, rung, seed)
    split = world.split(200, 300, seed)
    policy = replace(P.CURRENT, max_size=size(world.truth) + 2)
    return world, split.train, np.asarray(split.train_outcomes, dtype=np.int64), policy


def _flat(policy, cols, target, budget, profile=False, log=None):
    return P.run(P.with_budget(policy, budget), cols, target, profile=profile, log_rounds=log)


def _right(world, program) -> bool:
    return world.grade(lambda cols: discovery.predict(program, cols)) == 1.0


def _first_exact(found, cols, target, condition=False):
    """Evaluations to the first exact program of the anytime curve, and it."""
    ev = Evaluator(cols)
    for count, program, _ in found.anytime:
        out = ev(program)
        if condition:
            out = (out != 0).astype(np.int64)
        if np.array_equal(out, target):
            return count, program
    return None, None


def _target_features(cols, target):
    n, variables = len(target), len(cols)
    values = len({int(v) for c in cols.values() for v in c})
    alphabet = description.alphabet_of(target)
    return {"значений в цели": alphabet,
            "доля нулей в цели": float(np.mean(target == 0)) if n else 0.0,
            "биты таблицы цели": description.table_bits(n, variables, values, alphabet)}


def trace_job(inst, source="beam", rules=None):
    world, cols, target, policy = _setup(inst)
    rounds = []
    found = _flat(policy, cols, target, BUDGET, log=rounds)
    ev = Evaluator(cols)
    alphabet, known = description.alphabet_of(target), description.membership(target)
    cumulative, before, total = [], set(), 0
    for r, (pool, beam) in enumerate(rounds):
        total += len(set(pool) - before) if r else len(pool)
        cumulative.append(min(total, found.evaluations))
        before = set(beam)

    def record(x, r, place, stratum):
        out = ev(x)
        pb = description.program_bits(x, len(cols))
        eb = description.error_bits(out, target, alphabet, known)
        return {"program": x, "have": cumulative[r], "stratum": stratum,
                "features": {"биты X": pb + eb, "биты программы X": pb, "биты ошибок X": eb,
                             "неверных точек X": int(np.count_nonzero(out != target)),
                             "размер X": size(x), "раунд": r, "место в луче": place}}

    held, seen = [], set()
    for r, (_, beam) in enumerate(rounds):
        for place, x in enumerate(beam):
            if x in seen:
                continue
            seen.add(x)
            held.append(record(x, r, place, "луч"))
    if source == "beam":
        if len(held) > MAX_X:
            keep = sorted(set(np.linspace(0, len(held) - 1, MAX_X).round().astype(int)))
            held = [held[i] for i in keep]
        chosen = held
    elif source == "look":
        evaluated = set()
        for pool, _ in rounds:
            evaluated.update(pool)
        memo = {}

        def key(p):
            if p not in memo:
                out = ev(p)
                memo[p] = (description.program_bits(p, len(cols))
                           + description.error_bits(out, target, alphabet, known), size(p), show(p))
            return memo[p]

        ahead = replace(policy, rules=tuple(rules))
        leaves, conditions = discovery.vocabulary(cols, target)
        per_round = len(NEAR_RANKS) + FAR
        use = list(range(len(rounds)))
        if len(use) * per_round > POOL_MAX:
            use = sorted(set(np.linspace(0, len(use) - 1, POOL_MAX // per_round)
                             .round().astype(int)))
        chosen = []
        for r in use:
            shortlist = sorted(rounds[r][0], key=key)[:SHORTLIST]
            looked, results = set(), {}
            for state in shortlist:
                made = list(dict.fromkeys(P.successors(ahead, state, leaves, conditions)))
                looked.update(made)
                if made:
                    results.setdefault(min(made, key=key), state)
            cost = len(looked)
            fresh = sorted((x for x in results if x not in evaluated), key=key)
            near = [fresh[i] for i in NEAR_RANKS if i < len(fresh)]
            others = [x for x in fresh if x not in set(near)]
            rng = np.random.default_rng([sum(map(ord, inst[0])), inst[1], inst[2], r, 67])
            far = [others[i] for i in rng.choice(len(others), size=min(FAR, len(others)),
                                                 replace=False)] if others else []
            place = {x: i for i, x in enumerate(fresh)}
            for stratum, xs in (("ближние", near), ("дальние", far)):
                for x in xs:
                    row = record(x, r, place[x], stratum)
                    row["have"] = cumulative[r] + cost
                    row["parent"] = show(results[x])
                    chosen.append(row)
    else:
        beams = seen
        per_round = len(NEAR_RANKS) + FAR
        use = list(range(len(rounds)))
        if len(use) * per_round > POOL_MAX:
            use = sorted(set(np.linspace(0, len(use) - 1, POOL_MAX // per_round)
                             .round().astype(int)))
        chosen = []
        for r in use:
            pool = rounds[r][0]
            keys = {}
            for p in pool:
                out = ev(p)
                keys[p] = (description.program_bits(p, len(cols))
                           + description.error_bits(out, target, alphabet, known),
                           size(p), show(p))
            order = sorted(pool, key=keys.__getitem__)
            place = {p: i for i, p in enumerate(order)}
            rest = [p for p in order if p not in beams]
            near = [rest[i] for i in NEAR_RANKS if i < len(rest)]
            others = [p for p in rest if p not in set(near)]
            rng = np.random.default_rng([sum(map(ord, inst[0])), inst[1], inst[2], r, 61])
            far = [others[i] for i in rng.choice(len(others), size=min(FAR, len(others)),
                                                 replace=False)] if others else []
            chosen += [record(x, r, place[x], "ближние") for x in near]
            chosen += [record(x, r, place[x], "дальние") for x in far]
    return inst, {"solved": _right(world, found.program), "evaluations": found.evaluations,
                  "bits": found.bits, "held": chosen}


def branch_job(args):
    inst, index, x, branch, have = args
    world, cols, target, policy = _setup(inst)
    ev = Evaluator(cols)
    xv = ev(x)
    row = {"inst": inst, "index": index, "branch": branch, "asked": True}
    if branch in (RESIDUAL, RESIDUAL_MINUS):
        asked = target - xv if branch == RESIDUAL else xv - target
        found = _flat(policy, cols, asked, SUB, profile=True)
        cost, answer = _first_exact(found, cols, asked)
        row.update(_target_features(cols, asked))
        row["after"] = found.bits
        row["probe"] = _flat(policy, cols, asked, PROBE).bits
        candidate = None if answer is None else (add(x, answer) if branch == RESIDUAL
                                                 else sub(x, answer))
    else:
        misses = xv != target
        if not misses.any():
            return {**row, "asked": False}
        mcols = {k: v[misses] for k, v in cols.items()}
        found_m = _flat(policy, mcols, target[misses], SUB, profile=True)
        cost_m, rest = _first_exact(found_m, mcols, target[misses])
        row.update(_target_features(mcols, target[misses]))
        row["probe"] = _flat(policy, mcols, target[misses], PROBE).bits
        row["after"] = found_m.bits
        cost, candidate = None, None
        if rest is not None:
            theirs = ev(rest)
            told = (xv == target) != (theirs == target)
            if told.any():
                ccols = {k: v[told] for k, v in cols.items()}
                ctarget = (xv == target)[told].astype(np.int64)
                found_c = _flat(policy, ccols, ctarget, SUB, profile=True)
                cost_c, condition = _first_exact(found_c, ccols, ctarget, condition=True)
                row["after"] += found_c.bits
                if condition is not None:
                    cost, candidate = cost_m + cost_c, if_(condition, x, rest)
    exact = candidate is not None and np.array_equal(ev(candidate), target)
    row["solves"] = bool(exact and _right(world, candidate))
    row["route"] = have + cost + 2 if cost is not None else math.inf
    row["subproblem"] = cost
    row["candidate"] = show(candidate) if candidate is not None else ""
    return row


def auc(pos, neg, larger_first):
    if not pos or not neg:
        return float("nan")
    sign = 1 if larger_first else -1
    wins = sum((sign * p > sign * q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    source = sys.argv[2] if len(sys.argv) > 2 else "beam"
    started = time.time()
    instances = [(f, k, s) for f, k in RUNGS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rules = None
        if source == "look":
            import run_p2 as P2
            import run_reflect as R2
            from mana.discovery import reflect
            rows = list(pool.map(R2.experience_job, [("A", n, s) for n in R2.TRAIN
                                                     for s in range(10)]))
            change = reflect.improve(P.CURRENT, [r[5] for r in rows if r[4]])
            steps = [what.split(":")[0] for what, _ in change.steps]
            print(f"стадия 0: A' пересобрана, изменение {steps}; {time.time() - started:.0f}с")
            if steps != P2.R2_CHANGE:
                print("изменение не совпало с R2 — прогон остановлен")
                return
            rules = change.after.rules
        traces = dict(pool.map(trace_job, instances, [source] * len(instances),
                               [rules] * len(instances)))
        jobs = [(inst, i, x["program"], b, x["have"]) for inst in instances
                for i, x in enumerate(traces[inst]["held"]) for b in BRANCHES]
        rows = list(pool.map(branch_job, jobs))
    print(f"всего {time.time() - started:.0f}с; пар (X, ветвь) {len(rows)}")
    for r in rows:
        t = traces[r["inst"]]
        r["features"] = {**t["held"][r["index"]]["features"],
                         **{k: r[k] for k in FREE if k in r}}
        limit = BUDGET if not t["solved"] else t["evaluations"] / 2
        r["support"] = r["asked"] and r["solves"] and r["route"] <= limit
        r["reduced"] = r["asked"] and r.get("after", math.inf) <= t["bits"] / 2

    print("\n=== D2-prep-1: опоры и сокращения (экземпляров из 10)")
    print("  ступень  плоский решил  X на экз.(мед)  с опорой  опор (мед | макс)  с сокращением")
    for f, k in RUNGS:
        mine = [i for i in instances if i[:2] == (f, k)]
        per = {i: [r for r in rows if r["inst"] == i] for i in mine}
        supports = {i: sum(r["support"] for r in per[i]) for i in mine}
        counts = sorted(supports.values())
        print(f"  {f}{k}      {sum(traces[i]['solved'] for i in mine):>6d}        "
              f"{sorted(len(traces[i]['held']) for i in mine)[5]:>6d}         "
              f"{sum(1 for v in counts if v):>5d}      {counts[5]:>3d} | {counts[-1]:>3d}        "
              f"{sum(1 for i in mine if any(r['reduced'] for r in per[i])):>5d}")
    good = [r for r in rows if r["support"]]
    print(f"\n  опоры по ветвям: {Counter(r['branch'] for r in good).most_common()}")
    for stratum in sorted({x["stratum"] for t in traces.values() for x in t["held"]}):
        mine = [r for r in rows if traces[r["inst"]]["held"][r["index"]]["stratum"] == stratum]
        with_one = sorted({r["inst"] for r in mine if r["support"]})
        reduced = {r["inst"] for r in mine if r["reduced"]}
        print(f"  слой {stratum}: пар {len(mine)}, опор {sum(r['support'] for r in mine)}, "
              f"экземпляров с опорой {len(with_one)} {with_one}, с сокращением {len(reduced)}")
    for r in good[:8]:
        print(f"    {r['inst']}: {r['branch']}, цена пути {r['route']}, подзадача {r['subproblem']}; "
              f"X = {show(traces[r['inst']]['held'][r['index']]['program'])} -> {r['candidate']}")
    if source == "look":
        parents = {i: {traces[i]["held"][r["index"]]["parent"] for r in good if r["inst"] == i}
                   for i in instances}
        repro = sorted(i for i in instances if len(parents[i]) >= 2)
        print(f"\n  воспроизводимая опора (взгляды хотя бы на два разных состояния): "
              f"{len(repro)} экземпляров {repro}")
        deep = [i for i in instances if i[0] == "R" and i[1] in (4, 5)]
        deep_repro = [i for i in deep if i in repro]
        deep_reduced = [i for i in deep if any(r["reduced"] for r in rows if r["inst"] == i)]
        new = [i for i in repro if i not in TRACE_SUPPORTS]
        channel = len(deep_repro) >= 2 or len(deep_reduced) >= 5 or len(new) >= 6
        print(f"  R4, R5: воспроизводимых опор {len(deep_repro)} из {len(deep)}, с сокращением "
              f"{len(deep_reduced)} из {len(deep)}; новых (не из трассы) с воспроизводимой опорой "
              f"{len(new)} {new}")
        print(f"  канал по объявленному правилу: {'ЕСТЬ' if channel else 'нет'}")

    print("\n=== D2-prep-2: распознаваемость (только если опоры есть)")
    if not good:
        print("  опор нет — распознавать нечего")
        return
    with_support = [i for i in instances
                    if any(r["support"] for r in rows if r["inst"] == i)]
    pool_rows = [r for r in rows if r["asked"] and r["inst"] in with_support]
    for name in FREE + ("проба 2k",):
        key = "probe" if name == "проба 2k" else None
        larger = name in LARGER_FIRST

        def value(r):
            return r["probe"] if key else r["features"].get(name, math.nan)

        pos = [value(r) for r in pool_rows if r["support"]]
        neg = [value(r) for r in pool_rows if not r["support"]]
        hits = 0
        for inst in with_support:
            mine = [r for r in pool_rows if r["inst"] == inst]
            best = (max if larger else min)(mine, key=value)
            hits += best["support"]
        kind = "дешёвая проба" if key else "бесплатный"
        print(f"  {name:22s} ({kind:13s}) площадь под кривой {auc(pos, neg, larger):.2f}; "
              f"первая пара — опора в {hits} из {len(with_support)} экземпляров")


if __name__ == "__main__":
    main()
