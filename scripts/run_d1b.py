"""H3, D1b: does the recursive form, with D0's catalogue, gain at equal budget?

docs/ГЛУБИНА_D1.md, declared (commit 6ea1dd5) before this was written.

    arms       D1b-A      the claim: each node's flat search to its own stop
               D1b-S1/2   diagnostic: each node's flat search at most 1/2
               D1b-S1/4   diagnostic: at most 1/4
               A, order reversed -- the order control, reported only
    baseline   the flat search as it stands, 400k, size limit truth + 2
    sets       pairs      M3, M4, R3, R4, R5, seeds 20..29
               hidden     the same rungs, seeds 30..39
               counter    R1, M1, S1, C1, S2, S3, S4, seeds 20..29
               outside    R2, M2, M5, S5, seeds 20..29 -- reported only
    verdict    core.gates.judge on A; the same gates on the diagnostic arms
               only to decide whether D1b is sensitive to the budget split

    python -X utf8 scripts/run_d1b.py [workers]
"""
from __future__ import annotations

import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.core import gates  # noqa: E402
from mana.discovery import ladder, problems  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import ADD, size  # noqa: E402

BUDGET = 400000
ARMS = {"A": (1.0, problems.BRANCHES),
        "S1/2": (0.5, problems.BRANCHES),
        "S1/4": (0.25, problems.BRANCHES),
        "A, обратный порядок": (1.0, tuple(reversed(problems.BRANCHES)))}
CLAIMED = ("A", "S1/2", "S1/4")
RUNGS = [("M", 3), ("M", 4), ("R", 3), ("R", 4), ("R", 5)]
SETS = {"pairs": [(f, k, s) for f, k in RUNGS for s in range(20, 30)],
        "hidden": [(f, k, s) for f, k in RUNGS for s in range(30, 40)],
        "counter": [(f, k, s) for f, k in [("R", 1), ("M", 1), ("S", 1), ("C1", 1),
                                            ("S", 2), ("S", 3), ("S", 4)] for s in range(20, 30)],
        "outside": [(f, k, s) for f, k in [("R", 2), ("M", 2), ("M", 5), ("S", 5)]
                    for s in range(20, 30)]}


def _addends(p):
    return [p[1]] + _addends(p[2]) if p[0] == ADD else [p]


def root_is_addends(world, program) -> bool:
    """Instrument only: is the root's R the sum of some of the truth's addends?"""
    states = ladder._states()
    mine = ladder._vector(program, states)
    parts = [ladder._vector(a, states) for a in _addends(world.truth)]
    for n in range(1, len(parts) + 1):
        for chosen in itertools.combinations(parts, n):
            if np.array_equal(sum(chosen), mine):
                return True
    return False


def job(args):
    arm, (family, rung, seed) = args
    world = ladder.world(family, rung, seed)
    split = world.split(200, 300, seed)
    policy = replace(P.CURRENT, max_size=size(world.truth) + 2)
    grade = lambda p: world.grade(lambda cols: discovery.predict(p, cols)) == 1.0  # noqa: E731
    if arm == "flat":
        found = P.run(P.with_budget(policy, BUDGET), split.train, split.train_outcomes)
        return arm, (family, rung, seed), {"solved": grade(found.program),
                                           "evaluations": found.evaluations}
    share, order = ARMS[arm]
    began = time.time()
    solved = problems.solve(problems.Problem.whole(split.train, split.train_outcomes),
                            policy, BUDGET, catalogue=order, flat_share=share)
    right = grade(solved.program)
    depth, measured = (problems.effective_depth(solved, policy)
                       if right and len(solved.nodes) > 1 else (1, 0))
    return arm, (family, rung, seed), {
        "solved": right, "spent": dict(solved.ledger.spent), "nodes": len(solved.nodes),
        "asked": len(solved.nodes) > 1, "used_depth": solved.used_depth(),
        "effective_depth": depth, "measured": measured,
        "addends": root_is_addends(world, solved.found.program) if family == "R" else None,
        "seconds": time.time() - began}


def verdict(got, arm):
    at = lambda a, i: got[(a, i)]["solved"]  # noqa: E731
    pairs = [gates.PairedOutcome(f"{f}{k}/{s}", f, at("flat", (f, k, s)), at(arm, (f, k, s)))
             for f, k, s in SETS["pairs"]]
    hidden = SETS["hidden"]

    def acc(a, items, family=None):
        mine = [i for i in items if family is None or i[0] == family]
        return sum(at(a, i) for i in mine) / len(mine)

    counter = [i for i in SETS["counter"] if at("flat", i) and not at(arm, i)]
    evidence = gates.Evidence(
        paired_dev=pairs,
        baseline_hidden=acc("flat", hidden), candidate_hidden=acc(arm, hidden),
        baseline_hidden_by_domain={d: acc("flat", hidden, d) for d in ("R", "M")},
        candidate_hidden_by_domain={d: acc(arm, hidden, d) for d in ("R", "M")},
        counterexamples_sought=len(SETS["counter"]), counterexamples_found=len(counter))
    claim = gates.Claim(f"D1b-{arm}", "program",
                        "рекурсивная процедура с каталогом D0 решает лестницу чаще плоского поиска",
                        asserts_domains=("R", "M"))
    return gates.judge(claim, evidence), evidence, counter


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    instances = [i for items in SETS.values() for i in items]
    jobs = [("flat", i) for i in instances] + [(a, i) for a in ARMS for i in instances]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(job, jobs))
    got = {(arm, inst): row for arm, inst, row in runs}
    print(f"всего {time.time() - started:.0f}с")
    over = [(a, i) for (a, i), r in got.items() if a != "flat" and sum(r["spent"].values()) > BUDGET]
    print(f"бюджет превышен: {len(over)} прогонов {over[:5]}")

    print("\n=== решено (из 10) по ступеням: плоский | A | S1/2 | S1/4 | A обратный")
    for name, items in SETS.items():
        rungs = sorted({(f, k) for f, k, _ in items}, key=lambda x: (x[0], x[1]))
        for f, k in rungs:
            mine = [i for i in items if i[:2] == (f, k)]
            cells = [sum(got[(a, i)]["solved"] for i in mine) for a in ["flat"] + list(ARMS)]
            asked = sum(got[("A", i)]["asked"] for i in mine)
            print(f"  {name:8s} {f}{k}: " + " | ".join(f"{c:>2d}" for c in cells)
                  + f"   (A задала вопросы в {asked} из {len(mine)})")

    print("\n=== вердикты core.gates")
    signs, outcomes = {}, {}
    for arm in list(ARMS):
        v, evidence, counter = verdict(got, arm)
        m = v.measurements
        b, c = m["mcnemar"]["b"], m["mcnemar"]["c"]
        signs[arm] = int(np.sign(c - b))
        outcomes[arm] = tuple(sorted(v.failed_gates))
        role = "утверждение" if arm == "A" else ("контроль порядка" if "обратный" in arm
                                               else "диагностика")
        print(f"  {arm:20s} ({role}): {v.status} — {v.reason}; пары {m['dev_baseline']} -> "
              f"{m['dev_candidate']}, b={b} c={c} p={m['mcnemar']['p_value']:.4f}; скрытые "
              f"{evidence.baseline_hidden:.2f} -> {evidence.candidate_hidden:.2f}; контрпримеров "
              f"{len(counter)}/{len(SETS['counter'])} {counter[:8]}")
    sensitive = len({signs[a] for a in CLAIMED}) > 1 or len({outcomes[a] for a in CLAIMED}) > 1
    print(f"\n  чувствительность к распределению бюджета: {'ДА' if sensitive else 'нет'} "
          f"(знак c-b: {signs}; непройденные ворота: { {a: outcomes[a] for a in CLAIMED} })")
    print(f"  контроль порядка меняет вердикт A: "
          f"{'ДА' if outcomes['A'] != outcomes['A, обратный порядок'] else 'нет'}")

    print("\n=== выигранные пары A (решила процедура, плоский нет): эффективная глубина")
    for i in SETS["pairs"]:
        if got[("A", i)]["solved"] and not got[("flat", i)]["solved"]:
            r = got[("A", i)]
            print(f"  {i}: эффективная {r['effective_depth']}, использованная {r['used_depth']}, "
                  f"узлов {r['nodes']}")

    print("\n=== цена по статьям (сумма по всем экземплярам), вычислений")
    for arm in ARMS:
        total = {a: sum(got[(arm, i)]["spent"][a] for i in instances) for a in problems.ARTICLES}
        measured = sum(got[(arm, i)]["measured"] for i in instances)
        print(f"  {arm:20s} " + ", ".join(f"{a} {v}" for a, v in total.items())
              + f"; сворачивание (вне бюджета) {measured}")

    print("\n=== диагностика прибора: R корня — сумма слагаемых истины (ступени R)")
    for arm in CLAIMED:
        mine = [i for items in (SETS["pairs"], SETS["hidden"]) for i in items if i[0] == "R"]
        print(f"  {arm:6s}: {sum(bool(got[(arm, i)]['addends']) for i in mine)} из {len(mine)}")


if __name__ == "__main__":
    main()
