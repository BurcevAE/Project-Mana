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
#: Ceilings (10.5), frozen before the run. What bounds screening is time,
#: not a count: every size gets an equal share of it, and coverage is what
#: the share buys (10.5.1).
CEILING = {"ступень 2": 200_000, "ступень 3б": 100}
#: Of the 12 hours, what stage 3b is held back for.
RESERVE_3B = 1.5
#: Stage 3a takes the same number of candidates from every size, not the
#: first N of the whole order: the size is one of the variables of the
#: experiment, so it must not depend on how well small programs screen
#: (owner's decision, 10.3). The number itself is derived at the boundary of
#: screening, from what is left of the budget and what one run costs in this
#: session; this is the cap it may not exceed.
QUOTA_CAP = 500
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
            "counts": dict(counts), "passed": passed, "visited": len(listed)}


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


def _batches(most, done, limit=0):
    """The expressions of one size, numbered, in batches, skipping what is done.
    The rank is the place in the whole declared order, not within the size:
    ranks of different sizes must not collide. `limit` walks only the first
    so many of the size -- for measuring the rate of a size too large to
    exhaust."""
    stream = islice(X.generate(most), limit) if limit else X.generate(most)
    number, rank = 0, _base(most)
    while True:
        listed = [[rank + i, e] for i, e in enumerate(islice(stream, BATCH))]
        if not listed:
            return
        rank += len(listed)
        number += 1
        if f"screen:{most}:{number}" not in done:
            yield (number, most, listed)


def from_file(done):
    """Survivors and coverage as an earlier session wrote them, for a run
    that takes up stage 3a where screening was stopped by hand (10.9). What
    was screened is read, never screened again: the file is the record."""
    survivors, covered = [], {}
    for row in done.values():
        if row.get("kind") != "screen":
            continue
        n = row["size"]
        visited, whole, here = covered.get(n, (0, X.total(n), 0.0))
        covered[n] = (visited + row.get("visited", 0), whole, here)
        survivors += row.get("passed", [])
    return survivors, covered


def _full_key(job) -> str:
    rank, _, inst, branch = job
    return f"full:{rank}:{inst}:{branch}"


def _by_quota(survivors, quota):
    """The candidates stage 3a takes: the same number from every size, in the
    declared order, dropping only exact repetitions of X -- no clustering and
    no judgement of what counts as the same intermediate object (10.3)."""
    chosen, taken = [], {}
    for rank, n, item, _, _, x in survivors:
        mine = taken.setdefault(n, {})
        same = json.dumps(x, ensure_ascii=False)
        if same in mine or len(mine) >= quota:
            continue
        mine[same] = rank
        chosen.append((rank, n, item))
    return chosen


def _measure(pool, workers, path, done, jobs):
    """What one stage-3a run costs, measured on real runs of this session --
    they count, and nothing about their outcome takes part in the number.

    Four waves, not two: the first run in each process builds its worlds and
    searches them flat at 400k, and on a short sample that one-off lands in
    the mean and halves the quota it derives. The setup is paid in the real
    run too, so it belongs in the number -- it must simply not be counted as
    if every run paid it."""
    mine = [j for j in jobs if _full_key(j) not in done][:workers * 4]
    if not mine:
        return [], 10.6
    began = time.time()
    rows = list(pool.map(full_job, mine))
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            _write(fh, done, row)
    return rows, max(0.1, (time.time() - began) * workers / len(mine))


def _write(fh, done, row):
    done[row["key"]] = row
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def screen(pool, workers, path, done, most, started, limit=0):
    """Stages 1 and 2, size by size, each size on an equal share of the
    screening budget (10.5.1).

    A size exhausted before its share returns what it did not spend, and the
    pool is shared equally among the sizes still to come -- so the last size
    is not left with the structural zero a strict order hands it. Coverage
    is what the share buys, measured and reported; the share itself is
    decided by time and by how many expressions were visited, never by what
    was found."""
    counts = Counter()
    survivors, covered, stopped = [], {}, ""
    sizes = [n for n in range(1, most + 1) if X.total(n)]
    with open(path, "a", encoding="utf-8") as fh:
        for i, n in enumerate(sizes):
            left = SCREEN_HOURS * 3600 - (time.time() - started)
            if left <= 0:
                stopped = f"бюджет скрининга кончился до размера {n}"
                break
            share = left / (len(sizes) - i)
            whole = X.total(n)
            began, visited, here_stopped = time.time(), 0, ""
            print(f"  размер {n}: выражений {whole}, доля {share / 60:.0f} мин", flush=True)
            pending, stream, ended = [], _batches(n, done, limit), False
            while True:
                while len(pending) < workers * 2 and not ended:
                    job = next(stream, None)
                    if job is None:
                        ended = True
                        break
                    pending.append(pool.apply_async(screen_batch, (job,)))
                if not pending:
                    break
                row = pending.pop(0).get()
                _write(fh, done, row)
                counts.update(row["counts"])
                survivors += row["passed"]
                visited += row["visited"]
                if counts["ступень 2"] >= CEILING["ступень 2"]:
                    here_stopped = stopped = "потолок ступени 2"
                elif time.time() - began > share:
                    here_stopped = "доля израсходована"
                if here_stopped:
                    break
            for rest in pending:              # what was already submitted is counted
                row = rest.get()
                _write(fh, done, row)
                counts.update(row["counts"])
                survivors += row["passed"]
                visited += row["visited"]
            here = time.time() - began
            covered[n] = (visited, whole, here)
            label = here_stopped or ("исчерпан" if visited >= whole else "пройден отрезок")
            print(f"  размер {n}: {label}; охват {visited} из {whole} "
                  f"({100 * visited / whole:.1f}%); расход {here:.0f}с; "
                  f"{visited / here if here else 0:.0f} выражений в секунду; "
                  f"прошло всего {len(survivors)}", flush=True)
            if stopped:
                break
    return survivors, counts, stopped, covered


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "d3_prep1_results.jsonl")
    most = int(sys.argv[3]) if len(sys.argv) > 3 else X.MOST
    only = len(sys.argv) > 4 and sys.argv[4] == "скрининг"
    straight = len(sys.argv) > 4 and sys.argv[4] == "3а"
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    started = time.time()
    done = load(path)
    print(f"результаты: {path}; уже записано {len(done)}; потолки {CEILING}", flush=True)
    with multiprocessing.Pool(workers) as pool:
        if straight:
            survivors, covered = from_file(done)
            counts, stopped = Counter(), "скрининг остановлен решением владельца (10.9)"
            print(f"  из файла: выживших {len(survivors)}; размеров {len(covered)}; "
                  f"пройдено {sum(v for v, _, _ in covered.values())}", flush=True)
        else:
            survivors, counts, stopped, covered = screen(pool, workers, path, done, most, started,
                                                         limit)
        spent = time.time() - started
        print(f"\nскрининг: {dict(counts)}; прошло {len(survivors)}; {spent:.0f}с", flush=True)
        if only:
            walked = sum(visited for visited, _, _ in covered.values())
            print(f"  пройдено выражений {walked} за {spent:.0f}с на {workers} процессах")
            # No extrapolation by a single rate: the cost per expression
            # differs by size and by where in the order one stands -- the
            # head of size 9 screened at 283 a second against 83 for the
            # whole of size 8 (10.2). Each size reports its own.
            for n in sorted(covered):
                visited, whole, here = covered[n]
                print(f"    размер {n}: охват {visited} из {whole}, расход {here:.0f}с, "
                      f"{visited / here if here else 0:.0f} в секунду")
            return
        survivors.sort(key=lambda s: s[0])
        capped = _by_quota(survivors, QUOTA_CAP)
        jobs = [(rank, item, inst, 0) for rank, _, item in capped for inst in FIRST_FOUR]
        rows, cost = _measure(pool, workers, path, done, jobs)
        left = HOURS * 3600 - (time.time() - started) - RESERVE_3B * 3600
        sizes = sorted({n for _, n, _ in capped})
        quota = max(1, int(left * workers / (cost * len(FIRST_FOUR) * max(1, len(sizes)))))
        quota = min(quota, QUOTA_CAP)
        print(f"  цена одного прогона {cost:.1f}с; остаток {left / 3600:.1f}ч; "
              f"размеров {len(sizes)}; квота {quota}", flush=True)
        chosen = _by_quota(survivors, quota)
        for n in sizes:
            print(f"  ступень 3а, размер {n}: взято {sum(1 for _, k, _ in chosen if k == n)} из "
                  f"{sum(1 for s in survivors if s[1] == n)}", flush=True)
        jobs = [(rank, item, inst, 0) for rank, _, item in chosen for inst in FIRST_FOUR]
        with open(path, "a", encoding="utf-8") as fh:
            for row in pool.imap(full_job, [j for j in jobs if _full_key(j) not in done],
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
    report(survivors, counts, stopped, kept, confirmed, by_rank, started, covered)


def report(survivors, counts, stopped, kept, confirmed, by_rank, started, covered):
    print(f"\n=== D3-prep-1, {time.time() - started:.0f}с")
    print(f"  скрининг: {dict(counts)}")
    print(f"  выражений, прошедших ступень 2: {len(survivors)}; ступень 3а: {len(kept)}")
    print(f"  остановка скрининга: {stopped or 'нет, пройдено до объявленного размера'}")
    print("  охват пространства по размерам (проверено из скольких):")
    for n in sorted(covered):
        visited, whole, here = covered[n]
        print(f"    размер {n}: {visited} из {whole} ({100 * visited / whole:.1f}%), "
              f"расход {here:.0f}с")
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
