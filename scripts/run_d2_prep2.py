"""H3, D2-prep-2: two-level research, Goal -> X -> Y.

docs/ГЛУБИНА_D2.md, declared (commit before this) after the research-question
contract passed its gate. A diagnostic: nothing is learnt, nothing claimed;
the truth is read only to grade, by the instrument.

    catalogue  8 plans of equal standing: D1b's three branches, the three A
               questions, the C mask, a P2-class step through A' (rebuilt from
               R2's experience; the run stops if the change is not R2's),
               width 16
    runs       per instance: the flat search; 8 one-level runs (plan p at the
               root, flat search below); 64 two-level runs (p1 at the root, p2
               on every question p1 derives, flat search below). Every node's
               flat search at most half its budget; the rest to the node's one
               plan; inside a plan, the shares its data gives
    arms       main, 400k: R3-R5, M3, M4 and C1, seeds 40..49
               no budget pressure, 1.6M: R4, R5, seeds 40..44
    support    two-level: the tree solves the root (right on the world) on an
               instance the flat search at that budget did not; the root chose
               p1; a node its answer was assembled from chose p2; neither
               assembly degenerate. One-level: the same with one level
    degenerate a node's answer takes, on its points, the values of its own flat
               answer or of one derived answer it used
    reproducible  two-level supports from two combinations at least whose
               observations X at the root differ
    channel    reproducible on 2 of 20 R4/R5 instances, or on 6 of 50 ladder
               instances without a one-level support

Resumable (added after a power failure lost a run two hours in): every run's
result is appended to a JSON-lines file as it arrives, and a restart skips
what the file already holds. The jobs, their budgets and the definitions
are unchanged; A' is rebuilt and checked on every start, as declared.

    python -u -X utf8 scripts/run_d2_prep2.py [workers] [results.jsonl]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mana.discovery import ladder, plans, problems, questions  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import Evaluator, size  # noqa: E402

BUDGET, BIG = 400000, 1600000
FLAT_SHARE = 0.5
WIDTH = 16
MAIN = [(f, k, s) for f, k in (("R", 3), ("R", 4), ("R", 5), ("M", 3), ("M", 4), ("C1", 1))
        for s in range(40, 50)]
BIGSET = [(f, k, s) for f, k in (("R", 4), ("R", 5)) for s in range(40, 45)]


def catalogue():
    return plans.D1B + plans.DIAGNOSTIC + (plans.p2_step(WIDTH),)


def _setup(inst):
    family, rung, seed = inst
    world = ladder.world(family, rung, seed)
    split = world.split(200, 300, seed)
    return world, split, replace(P.CURRENT, max_size=size(world.truth) + 2)


def _right(world, program) -> bool:
    return world.grade(lambda cols: discovery.predict(program, cols)) == 1.0


def _nondegenerate(solution, index) -> bool:
    node = solution.nodes[index]
    cols, _ = node.problem.data()
    ev = Evaluator(cols)
    mine = ev(node.program)
    if np.array_equal(mine, ev(node.found.program)):
        return False
    return not any(np.array_equal(mine, ev(solution.nodes[c].program)) for c in node.used)


def flat_job(args):
    inst, budget = args
    world, split, policy = _setup(inst)
    found = P.run(P.with_budget(policy, budget), split.train, split.train_outcomes)
    return {"inst": list(inst), "budget": budget, "right": _right(world, found.program)}


def tree_job(args):
    inst, budget, first, second, rules = args
    world, split, policy = _setup(inst)
    cat = catalogue()
    schedule = [(cat[first],)] + ([(cat[second],)] if second is not None else [])
    solved = questions.solve(problems.Problem.whole(split.train, split.train_outcomes), policy,
                             budget, flat_share=FLAT_SHARE, env={"ahead": rules},
                             schedule=schedule)
    root = solved.nodes[0]
    right = _right(world, solved.program)
    at_root = root.chosen == cat[first].name and _nondegenerate(solved, 0)
    deeper = second is not None and any(
        solved.nodes[c].chosen == cat[second].name and _nondegenerate(solved, c)
        for c in root.used)
    observation = next((r.observation for r in solved.records
                        if r.parent == 0 and r.plan == cat[first].name), "")
    return {"inst": list(inst), "budget": budget, "first": first, "second": second,
            "right": right, "at_root": at_root, "deeper": bool(deeper), "observation": observation,
            "spent": dict(solved.ledger.spent), "nodes": len(solved.nodes)}


def _key(kind, inst, budget, first=None, second=None) -> str:
    return json.dumps([kind, list(inst), budget, first, second])


def load(path):
    """What an earlier, interrupted run already computed."""
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


def stream(pool, fn, jobs, keys, done, path, label):
    """Run what is not done yet; append each result as it arrives."""
    todo = [(k, j) for k, j in zip(keys, jobs) if k not in done]
    print(f"  {label}: сделано раньше {len(jobs) - len(todo)}, осталось {len(todo)}", flush=True)
    started = time.time()
    futures = {pool.submit(fn, job): k for k, job in todo}
    with open(path, "a", encoding="utf-8") as fh:
        for n, future in enumerate(as_completed(futures), 1):
            k = futures[future]
            row = future.result()
            row["key"] = k
            done[k] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"    {label}: {n}/{len(todo)}, {time.time() - started:.0f}с", flush=True)


def rebuild_a_prime(pool):
    import run_p2 as P2
    import run_reflect as R2
    from mana.discovery import reflect
    rows = list(pool.map(R2.experience_job, [("A", n, s) for n in R2.TRAIN for s in range(10)]))
    change = reflect.improve(P.CURRENT, [r[5] for r in rows if r[4]])
    steps = [what.split(":")[0] for what, _ in change.steps]
    return change.after.rules, steps == P2.R2_CHANGE, steps


def report(title, instances, budget, flat, runs, cat):
    print(f"\n=== {title}")
    mine = [r for r in runs if r["budget"] == budget and r["inst"] in instances]
    unsolved = [i for i in instances if not flat[(i, budget)]]
    one = {i: {r["first"] for r in mine if r["inst"] == i and r["second"] is None
               and r["right"] and r["at_root"]} for i in unsolved}
    two = {i: [r for r in mine if r["inst"] == i and r["second"] is not None
               and r["right"] and r["at_root"] and r["deeper"]] for i in unsolved}
    repro = {i for i in unsolved if len({r["observation"] for r in two[i]}) >= 2}
    print("  ступень  плоский решил  одноуровневая опора  двухуровневая  воспроизводимая  новая")
    for f, k in sorted({i[:2] for i in instances}):
        rung = [i for i in instances if i[:2] == (f, k)]
        u = [i for i in rung if i in unsolved]
        print(f"  {f}{k}      {len(rung) - len(u):>6d}        {sum(1 for i in u if one[i]):>8d}"
              f"            {sum(1 for i in u if two[i]):>6d}        {sum(1 for i in u if i in repro):>8d}"
              f"      {sum(1 for i in u if i in repro and not one[i]):>4d}")
    pairs = Counter((cat[r["first"]].name, cat[r["second"]].name)
                    for i in unsolved for r in two[i])
    print(f"  пары (p1, p2) с двухуровневой опорой: {pairs.most_common(12)}")
    singles = Counter(cat[p].name for i in unsolved for p in one[i])
    print(f"  планы с одноуровневой опорой: {singles.most_common()}")
    solved_c1 = [r for r in mine if r["inst"][0] == "C1" and r["right"]]
    if any(i[0] == "C1" for i in instances):
        print(f"  C1 (вне каталога): деревьев, решивших C1, {len(solved_c1)}")
    deep = [i for i in unsolved if i[0] == "R" and i[1] in (4, 5)]
    ladder_inst = [i for i in unsolved if i[0] != "C1"]
    new = [i for i in ladder_inst if i in repro and not one[i]]
    return len([i for i in deep if i in repro]), len(new), sorted(repro)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "d2_prep2_results.jsonl")
    started = time.time()
    cat = catalogue()
    done = load(path)
    print(f"результаты: {path}; уже записано {len(done)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rules, same, steps = rebuild_a_prime(pool)
        print(f"стадия 0: A' пересобрана, изменение {steps}; {time.time() - started:.0f}с",
              flush=True)
        if not same:
            print("изменение не совпало с R2 — прогон остановлен")
            return
        arms = [(i, BUDGET) for i in MAIN] + [(i, BIG) for i in BIGSET]
        stream(pool, flat_job, arms, [_key("flat", i, b) for i, b in arms], done, path,
               "плоский поиск")
        jobs = [(i, b, p, None, rules) for i, b in arms for p in range(len(cat))]
        jobs += [(i, b, p, q, rules) for i, b in arms for p in range(len(cat))
                 for q in range(len(cat))]
        keys = [_key("tree", i, b, p, q) for i, b, p, q, _ in jobs]
        stream(pool, tree_job, jobs, keys, done, path, "деревья")
    flat = {(i, b): done[_key("flat", i, b)]["right"] for i, b in arms}
    runs = []
    for k in keys:
        row = dict(done[k])
        row["inst"] = tuple(row["inst"])
        runs.append(row)
    print(f"всего {time.time() - started:.0f}с; деревьев {len(runs)}")
    deep, new, repro = report("основное плечо, 400k", MAIN, BUDGET, flat, runs, cat)
    channel = deep >= 2 or new >= 6
    print(f"\n  R4/R5 с воспроизводимой двухуровневой опорой: {deep} из 20; новых (без "
          f"одноуровневой): {new} из 50; воспроизводимые: {repro}")
    print(f"  канал по объявленному правилу: {'ЕСТЬ' if channel else 'нет'}")
    report("без давления бюджета, 1.6 млн, R4/R5", BIGSET, BIG, flat, runs, cat)
    spent = [sum(r["spent"].values()) for r in runs if r["budget"] == BUDGET]
    print(f"\n  потрачено деревом при 400k (мед): {sorted(spent)[len(spent) // 2]}; узлов (мед): "
          f"{sorted(r['nodes'] for r in runs)[len(runs) // 2]}")


if __name__ == "__main__":
    main()
