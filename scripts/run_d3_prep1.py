"""H3, D3-prep-1: is there an experiment program at all?
(docs/ГЛУБИНА_D3.md, 10 -- declared before this code.)

No generator: the space of experiment programs (mana/discovery/expressions.py)
is walked in one fixed order, by size, and every expression is put through
three stages. The catalogue gives the derivation and the rebuild; the
experiment is the only thing that varies. Nothing is learnt here, and the
truth is read only by the instrument, only to grade, only at stage 3.

    1. допустимость   X is not the whole answer of the parent and not the
                      node's own answer; the derived question is built.
                      Node's budget 5k, the residual branch
    2. полезность     a probe at b0 = 20k: the subquestion is shorter than
                      the parent probed the same way, or the composition is
                      shorter than the parent's answer at that b0. One probe
                      per distinct X
    3а. проба         400k on 4 existence instances, the residual branch:
                      right on the world where the flat search is not. The
                      same quota of candidates from every size, after exact
                      repetitions of X are dropped
    3б. подтверждение 10 instances, all three D1b branches. Existence is
                      decided here and nowhere else (10.4)

Resumable: every batch and every run is a line of a JSON-lines file, and a
restart skips what is there.

    python -u -X utf8 scripts/run_d3_prep1.py [workers] [results.jsonl] [most] [скрининг]

The fourth argument, "скрининг", stops after stages 1 and 2 and reports what
they cost -- a measurement of throughput, to fix the quota and the coverage
before the probe itself is run.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import sys
import time
from collections import Counter
from dataclasses import replace
from itertools import islice
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import expressions as X  # noqa: E402
from mana.discovery import ladder, plans, problems, questions  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import Evaluator, add, size  # noqa: E402
from mana.discovery.search import vocabulary  # noqa: E402

SCREEN = (("R", 4, 70), ("R", 5, 71))
EXISTENCE = tuple((f, k, s) for f, k in (("R", 4), ("R", 5)) for s in range(60, 70))
FIRST_FOUR = (("R", 4, 60), ("R", 4, 61), ("R", 5, 60), ("R", 5, 61))

SCREEN_BUDGET, PROBE, FULL = 5000, 20000, 400000
FLAT_SHARE = 0.5
BATCH = 2000
#: Ceilings (10.5), frozen before the run.
CEILING = {"порождено": 15_000_000, "ступень 1": 15_000_000, "ступень 2": 200_000,
           "ступень 3б": 100}
#: Stage 3a takes the same number of candidates from every size, not the
#: first N of the whole order: the size is one of the variables of the
#: experiment, so it must not depend on how well small programs screen
#: (owner's decision, 10.3). Fixed before the run, from measured cost: one
#: run is 10.6 seconds of processor time at every size from 4 to 10, so
#: 8 sizes x 200 x 4 instances is about 3.2 hours on six processes.
QUOTA = 200
#: Of the 12 hours, what screening may take. Measured before the run: 234
#: expressions a second, so every size up to 9 is exhausted in 2.4 hours and
#: the rest of this budget goes into the beginning of size 10, which cannot
#: be exhausted at all (10.5). Without this, screening would eat the whole
#: ceiling and stage 3 would never run.
SCREEN_HOURS = 6.5
HOURS = 12.0
#: What an expression of the wrong shape may raise when it runs.
FAULTS = (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError, OverflowError,
          questions._Short)
CACHE = 2_000_000

_READY: dict = {}


def _tuples(x):
    return tuple(_tuples(i) for i in x) if type(x) is list else x


def _screening(inst):
    """What every expression on this instance is measured against -- computed
    once: the node's own answer at the screening budget, what its search held,
    and the parent probed at b0."""
    if inst in _READY:
        return _READY[inst]
    family, rung, seed = inst
    world = ladder.world(family, rung, seed)
    split = world.split(200, 300, seed)
    policy = replace(P.CURRENT, max_size=size(world.truth) + 2)
    columns = split.train
    target = np.asarray(split.train_outcomes, dtype=np.int64)
    leaves, conditions = vocabulary(columns, target)
    rounds: list = []
    # The contract's own rule: the node's flat search takes at most its share
    # of the budget, and what is left goes to the plan.
    own = P.run(P.with_budget(policy, int(SCREEN_BUDGET * FLAT_SHARE)), columns, target,
                log_rounds=rounds)
    base = P.run(P.with_budget(policy, PROBE), columns, target)
    evaluator = Evaluator(columns)
    _READY[inst] = {
        "problem": problems.Problem.whole(columns, target), "columns": columns, "target": target,
        "policy": policy, "leaves": leaves, "conditions": conditions, "own": own,
        "held": questions._held(rounds), "base": base, "evaluator": evaluator,
        "own_values": evaluator(own.program),
        "base_bits": problems._key(base.program, columns, target, evaluator)[0],
    }
    return _READY[inst]


def _admissible(e, it):
    """Stage 1 on one instance: (why it stopped, X, the values of X)."""
    share = SCREEN_BUDGET - it["own"].evaluations
    if share <= len(it["leaves"]):
        return "нет доли", None, None
    ledger = problems.Ledger(share)
    run = questions._Interpreter(it["policy"], ledger, (), 1.0, False, False, None)
    node = problems.Node(0, it["problem"], None, None, {a: 0 for a in problems.ARTICLES},
                         budget=share)
    run.nodes.append(node)
    scope = questions._Scope(run, node, it["columns"], it["target"], it["evaluator"],
                             it["leaves"], it["conditions"], it["own"].program, it["held"])
    try:
        x = scope.value(e, {"share": share}, True)
        values = np.asarray(it["evaluator"](x), dtype=np.int64)
    except FAULTS:
        return "не исполнилась", None, None
    if values.shape != it["target"].shape:
        return "не исполнилась", None, None
    if np.array_equal(values, it["target"]):
        return "полный ответ", None, None
    if np.array_equal(values, it["own_values"]):
        return "ответ узла", None, None
    return "", x, values


def _useful(x, values, it):
    """Stage 2 on one instance, by the instrument's own probe at b0."""
    sub = P.run(P.with_budget(it["policy"], PROBE), it["columns"], it["target"] - values)
    if sub.bits < it["base"].bits:
        return "подзадача"
    whole = problems._key(add(x, sub.program), it["columns"], it["target"], it["evaluator"])
    return "композиция" if whole[0] < it["base_bits"] else ""


def screen_batch(args):
    """Stages 1 and 2 for a batch of expressions, in the order given."""
    number, most, listed = args
    counts: Counter = Counter()
    passed = []
    seen = _READY.setdefault("seen", {})
    for rank, item in listed:
        e = _tuples(item)
        for inst in SCREEN:
            it = _screening(inst)
            why, x, values = _admissible(e, it)
            counts[why or "допущено"] += 1
            if x is None:
                continue
            mark = (inst, hashlib.blake2b(values.tobytes(), digest_size=16).digest())
            if mark in seen:
                counts["повтор X"] += 1
                good = seen[mark]
            else:
                if len(seen) >= CACHE:
                    seen.clear()
                good = _useful(x, values, it)
                seen[mark] = good
                counts["ступень 2"] += 1
            if good:
                counts["прошло"] += 1
                passed.append([rank, most, item, f"{inst[0]}{inst[1]}/{inst[2]}", good, x])
                break
    return {"kind": "screen", "key": f"screen:{most}:{number}", "size": most, "batch": number,
            "counts": dict(counts), "passed": passed}


# -- stage 3: the real check, through the contract itself ----------------------

def _world(inst):
    key = ("full",) + tuple(inst)
    if key in _READY:
        return _READY[key]
    family, rung, seed = inst
    world = ladder.world(family, rung, seed)
    split = world.split(200, 300, seed)
    policy = replace(P.CURRENT, max_size=size(world.truth) + 2)
    flat = P.run(P.with_budget(policy, FULL), split.train, split.train_outcomes)
    _READY[key] = (world, split, policy,
                   world.grade(lambda cols: discovery.predict(flat.program, cols)) == 1.0)
    return _READY[key]


def _nondegenerate(solution, index):
    node = solution.nodes[index]
    columns, _ = node.problem.data()
    evaluator = Evaluator(columns)
    mine = evaluator(node.program)
    if np.array_equal(mine, evaluator(node.found.program)):
        return False
    return not any(np.array_equal(mine, evaluator(solution.nodes[c].program)) for c in node.used)


def full_job(args):
    rank, item, inst, branch = args
    e = _tuples(item)
    world, split, policy, flat_right = _world(tuple(inst))
    plan = plans.with_experiment(plans.D1B[branch], e)
    whole = problems.Problem.whole(split.train, split.train_outcomes)
    try:
        solved = questions.solve(whole, policy, FULL, plans=(plan,), flat_share=FLAT_SHARE)
    except FAULTS:
        return {"kind": "full", "key": f"full:{rank}:{inst}:{branch}", "rank": rank,
                "item": item, "inst": list(inst), "branch": branch, "right": False,
                "flat_right": flat_right, "at_root": False, "observation": "", "fault": True}
    right = world.grade(lambda cols: discovery.predict(solved.program, cols)) == 1.0
    at_root = solved.nodes[0].chosen == plan.name and _nondegenerate(solved, 0)
    observation = next((r.observation for r in solved.records if r.parent == 0), "")
    return {"kind": "full", "key": f"full:{rank}:{inst}:{branch}", "rank": rank, "item": item,
            "inst": list(inst), "branch": branch, "right": bool(right),
            "flat_right": bool(flat_right), "at_root": bool(at_root),
            "observation": observation, "fault": False,
            "spent": dict(solved.ledger.spent), "nodes": len(solved.nodes)}


# -- the run -------------------------------------------------------------------

def load(path):
    done = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done[row["key"]] = row
    except FileNotFoundError:
        pass
    return done


def _base(most: int) -> int:
    """Where this size starts in the one declared order, counted not built."""
    return sum(X.total(n) for n in range(1, most))


def _batches(most, done):
    """The expressions of one size, numbered, in batches, skipping what is done.
    The rank is the place in the whole declared order, not within the size:
    ranks of different sizes must not collide."""
    stream = X.generate(most)
    number, rank = 0, _base(most)
    while True:
        listed = [[rank + i, e] for i, e in enumerate(islice(stream, BATCH))]
        if not listed:
            return
        rank += len(listed)
        number += 1
        if f"screen:{most}:{number}" not in done:
            yield (number, most, listed)


def _write(fh, done, row):
    done[row["key"]] = row
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def screen(pool, path, done, most, started):
    """Stages 1 and 2 over every size up to `most`, in the declared order."""
    counts: Counter = Counter()
    survivors = []
    stopped = ""
    with open(path, "a", encoding="utf-8") as fh:
        for n in range(1, most + 1):
            whole = X.total(n)
            began = time.time()
            print(f"  размер {n}: выражений {whole}", flush=True)
            for row in pool.imap(screen_batch, _batches(n, done), chunksize=1):
                _write(fh, done, row)
                counts.update(row["counts"])
                survivors += row["passed"]
                if row["batch"] % 20 == 0:
                    print(f"    {n}: батч {row['batch']}, прошло всего {len(survivors)}, "
                          f"{time.time() - started:.0f}с", flush=True)
                spent = sum(v for k, v in counts.items() if k != "повтор X" and k != "прошло")
                if spent >= CEILING["ступень 1"] or counts["ступень 2"] >= CEILING["ступень 2"]:
                    stopped = f"потолок на размере {n}"
                if time.time() - started > SCREEN_HOURS * 3600:
                    stopped = f"время скрининга на размере {n}"
                if stopped:
                    break
            done_here = not stopped
            here = time.time() - began
            # The rate is only the size's own when the size was exhausted: an
            # expression of size 10 is longer than one of size 5, and the
            # whole protocol's arithmetic hangs on whether the rate holds.
            rate = f", {whole / here:.0f} выражений в секунду" if done_here and here else ""
            print(f"  размер {n}: {'исчерпан' if done_here else stopped}; "
                  f"прошло {len(survivors)}; {here:.0f}с{rate}", flush=True)
            if stopped:
                break
    return survivors, counts, stopped


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "d3_prep1_results.jsonl")
    most = int(sys.argv[3]) if len(sys.argv) > 3 else X.MOST
    only = len(sys.argv) > 4 and sys.argv[4] == "скрининг"
    started = time.time()
    done = load(path)
    print(f"результаты: {path}; уже записано {len(done)}; потолки {CEILING}", flush=True)
    with multiprocessing.Pool(workers) as pool:
        survivors, counts, stopped = screen(pool, path, done, most, started)
        spent = time.time() - started
        print(f"\nскрининг: {dict(counts)}; прошло {len(survivors)}; {spent:.0f}с", flush=True)
        if only:
            walked = sum(X.total(n) for n in range(1, most + 1))
            rate = walked / spent if spent else 0
            print(f"  пройдено выражений {walked}, {rate:.0f} в секунду на {workers} процессах")
            for n in (9, 10, 11):
                whole = sum(X.total(k) for k in range(1, n + 1))
                print(f"  весь размер {n} и меньше: {whole} выражений ≈ "
                      f"{whole / rate / 3600:.1f} ч при этой скорости")
            return
        survivors.sort(key=lambda s: s[0])
        chosen, taken = [], {}
        for rank, n, item, _, _, x in survivors:
            mine = taken.setdefault(n, {})
            # Only exact repetitions of X are dropped -- no clustering, no
            # judgement of what counts as the same intermediate object.
            same = json.dumps(x, ensure_ascii=False)
            if same in mine or len(mine) >= QUOTA:
                continue
            mine[same] = rank
            chosen.append((rank, n, item))
        for n in sorted(taken):
            print(f"  ступень 3а, размер {n}: взято {len(taken[n])} из "
                  f"{sum(1 for s in survivors if s[1] == n)} (квота {QUOTA})", flush=True)
        jobs = [(rank, item, inst, 0) for rank, _, item in chosen for inst in FIRST_FOUR]
        rows = []
        with open(path, "a", encoding="utf-8") as fh:
            for row in pool.imap(full_job, [j for j in jobs if f"full:{j[0]}:{j[2]}:0" not in done],
                                 chunksize=1):
                _write(fh, done, row)
                rows.append(row)
        kept = sorted({row["rank"] for row in rows
                       if row["right"] and not row["flat_right"] and row["at_root"]})
        print(f"ступень 3а: прогонов {len(rows)}; выражений с выигрышем {len(kept)}", flush=True)
        by_rank = {rank: item for rank, _, item in chosen}
        jobs = [(rank, by_rank[rank], inst, branch) for rank in kept[:CEILING["ступень 3б"]]
                for inst in EXISTENCE for branch in range(3)]
        confirmed = []
        with open(path, "a", encoding="utf-8") as fh:
            for row in pool.imap(full_job, jobs, chunksize=1):
                _write(fh, done, row)
                confirmed.append(row)
    report(survivors, counts, stopped, kept, confirmed, by_rank, started)


def report(survivors, counts, stopped, kept, confirmed, by_rank, started):
    print(f"\n=== D3-prep-1, {time.time() - started:.0f}с")
    print(f"  скрининг: {dict(counts)}")
    print(f"  выражений, прошедших ступень 2: {len(survivors)}; ступень 3а: {len(kept)}")
    print(f"  остановка скрининга: {stopped or 'нет, пройдено до объявленного размера'}")
    wins = [row for row in confirmed if row["right"] and not row["flat_right"] and row["at_root"]]
    exists = sorted({row["rank"] for row in wins})
    print(f"  существование: {'ЕСТЬ' if exists else 'нет'}; выражений с выигрышем {len(exists)}")
    for rank in exists[:10]:
        mine = [row for row in wins if row["rank"] == rank]
        rungs = {tuple(row["inst"])[:2] for row in mine}
        seen = {row["observation"] for row in mine}
        general = len({tuple(row["inst"]) for row in mine}) >= 3 and len(rungs) >= 2 and len(seen) > 1
        print(f"    ранг {rank}: экземпляров {len({tuple(r['inst']) for r in mine})}, "
              f"ступеней {len(rungs)}, различных X {len(seen)}, "
              f"общность {'да' if general else 'нет'}")
        print(f"      {json.dumps(by_rank[rank], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
