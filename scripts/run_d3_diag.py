"""H3, диагностики Д1 и Д2 после D3-prep-1 (docs/ГЛУБИНА_D3.md, 12).

Ничего не утверждают и не отменяют: объясняют, во что упёрлась проба -- в
язык или в цену. Истина здесь не читается вовсе.

    Д1  отказ по бюджету (`_Short`) отдельно от отказа по форме (TypeError,
        IndexError и прочие), на 20 000 выражений, взятых равномерно из
        размеров 7-10, и на 324 кандидатах, упавших в пробе на всех
        четырёх экземплярах
    Д2  два на два: `candidates` до 6 или до 10, бюджет эксперимента 2 500
        или 40 000, на одних и тех же выражениях с `candidates`. Считается,
        сколько различных X даёт клетка и каковы их размеры

Как реализовано "одни и те же выражения при другом пределе словаря": у
выражения заменяется константа в `candidates(k)` на `k + 4` (так 6 -> 10),
всё остальное в нём не меняется.

    python -u -X utf8 scripts/run_d3_diag.py [workers] [д1|д2|обе] [results.jsonl]
"""
from __future__ import annotations

import collections
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mana.discovery import expressions as X  # noqa: E402
from mana.discovery import plans, problems, questions  # noqa: E402
import run_d3_prep1 as D3  # noqa: E402

SEED = 20260916
D1_SIZES = (7, 8, 9, 10)
D1_PER_SIZE = 5000
D2_SIZES = (8, 9, 10)
D2_WANTED = 2000
D2_DRAWS = 300000
#: The four cells of D2: (name, how much the vocabulary of candidates is
#: widened, what the experiment may spend).
CELLS = (("А: k≤6, 2.5k", 0, 2500), ("Б: k≤6, 40k", 0, 40000),
         ("В: k≤10, 2.5k", 4, 2500), ("Г: k≤10, 40k", 4, 40000))


def _try(e, it, share):
    """(X or None, why it failed) -- the budget apart from the form."""
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
    except questions._Short:
        return None, "бюджет"
    except Exception as bad:                      # noqa: BLE001 -- the class is the point
        return None, "форма: " + type(bad).__name__
    if values.shape != it["target"].shape:
        return None, "форма: не вектор точек"
    return x, ""


def _nodes(e) -> int:
    return 1 + sum(_nodes(p) for p in e[1:] if type(p) is tuple) if type(e) is tuple else 0


def _widen(e, by: int):
    """The same expression with candidates(k) asking for k + by."""
    if type(e) is not tuple or not e:
        return e
    if e[0] == "candidates" and e[1][0] == "const":
        return ("candidates", ("const", e[1][1] + by))
    return tuple(_widen(p, by) for p in e)


def _holds_candidates(e) -> bool:
    if type(e) is not tuple or not e:
        return False
    return e[0] == "candidates" or any(_holds_candidates(p) for p in e[1:])


# -- Д1 -------------------------------------------------------------------------

def d1_job(args):
    """Stage 1 on a sample of one size, the class of every failure kept."""
    size, count, seed = args
    rng = random.Random(seed + size)
    it = D3._screening(D3.SCREEN[0])
    share = D3.SCREEN_BUDGET - it["own"].evaluations
    counts: collections.Counter = collections.Counter()
    for _ in range(count):
        x, why = _try(X.sample(size, rng), it, share)
        counts[why or "исполнилась"] += 1
    return size, counts


def d1_full_job(item):
    """One of the candidates that faulted in the probe, at 400k, classified."""
    e = D3._tuples(item)
    world, split, policy, _ = D3._world(D3.FIRST_FOUR[0])
    plan = plans.with_experiment(plans.D1B[0], e)
    whole = problems.Problem.whole(split.train, split.train_outcomes)
    try:
        questions.solve(whole, policy, D3.FULL, plans=(plan,), flat_share=D3.FLAT_SHARE)
    except questions._Short:
        return "бюджет"
    except Exception as bad:                      # noqa: BLE001
        return "форма: " + type(bad).__name__
    return "исполнилась"


def fell_everywhere(path):
    """The candidates the probe recorded as faulting on every instance."""
    rows = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if row["kind"] == "full":
            rows[row["rank"]].append(row)
    return [mine[0]["item"] for mine in rows.values()
            if len(mine) >= 4 and all(r.get("fault") for r in mine)]


# -- Д2 -------------------------------------------------------------------------

def d2_sample(seed):
    """Expressions that hold a `candidates`, drawn uniformly, the same for
    every cell."""
    drawn, rng = [], random.Random(seed)
    for size in D2_SIZES:
        mine, tries = [], 0
        while len(mine) < D2_WANTED // len(D2_SIZES) and tries < D2_DRAWS:
            tries += 1
            e = X.sample(size, rng)
            if _holds_candidates(e):
                mine.append(e)
        drawn += mine
        print(f"  размер {size}: выражений с candidates {len(mine)} из {tries} проб", flush=True)
    return drawn


def d2_job(args):
    """One cell: what the same expressions give when a bar is lifted."""
    name, widen, share, listed = args
    it = D3._screening(D3.SCREEN[0])
    seen, sizes, why = {}, collections.Counter(), collections.Counter()
    for e in listed:
        x, bad = _try(_widen(D3._tuples(e), widen), it, share)
        if x is None:
            why[bad] += 1
            continue
        seen[json.dumps(x)] = x
        sizes[_nodes(x)] += 1
    return name, len(seen), dict(sorted(sizes.items())), dict(why.most_common(4))


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    what = sys.argv[2] if len(sys.argv) > 2 else "обе"
    path = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "d3_prep1_results.jsonl")
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if what in ("д1", "обе"):
            print("=== Д1: форма против бюджета", flush=True)
            jobs = [(n, D1_PER_SIZE, SEED) for n in D1_SIZES]
            whole: collections.Counter = collections.Counter()
            for size, counts in pool.map(d1_job, jobs):
                whole.update(counts)
                shown = dict(counts.most_common(6))
                print(f"  ступень 1, размер {size}: {shown}", flush=True)
            budget = whole["бюджет"]
            form = sum(v for k, v in whole.items() if k.startswith("форма"))
            ran = whole["исполнилась"]
            print(f"  ступень 1, всего: исполнилась {ran}, бюджет {budget}, форма {form}; "
                  f"доля формы среди отказов "
                  f"{100 * form / max(1, budget + form):.1f}%", flush=True)
            fell = fell_everywhere(path)
            print(f"  ступень 3а: кандидатов, упавших везде, {len(fell)}", flush=True)
            classes: collections.Counter = collections.Counter(pool.map(d1_full_job, fell))
            print(f"  ступень 3а, классы: {dict(classes.most_common())}", flush=True)
            form3 = sum(v for k, v in classes.items() if k.startswith("форма"))
            print(f"  ступень 3а, доля формы среди отказов "
                  f"{100 * form3 / max(1, len(fell) - classes['исполнилась']):.1f}%", flush=True)
        if what in ("д2", "обе"):
            print(f"\n=== Д2: какой запор держит щель; {time.time() - started:.0f}с", flush=True)
            listed = d2_sample(SEED)
            print(f"  выражений в выборке: {len(listed)}", flush=True)
            jobs = [(name, widen, share, listed) for name, widen, share in CELLS]
            print("  клетка            различных X   размеры X            отказы")
            for name, count, sizes, why in pool.map(d2_job, jobs):
                print(f"  {name:<16}  {count:>11}   {str(sizes):<20} {why}", flush=True)
    print(f"\nвсего {time.time() - started:.0f}с")


if __name__ == "__main__":
    main()
