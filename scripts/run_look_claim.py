"""N3, P2c: the claim of probe P2, declared before the run, on seeds nobody has seen.

The claims (mana/discovery/__init__.py, written before this ran), each
against the search as it stands, within 400 000 programs, every look
counted:

    P2c-look   a look at the first 256 states by score, one step ahead
               through the rules of A' -- the policy MANA grew in R2 --
               solves T4 more often
    P2c-width  the same look through A' without its grown move (taken out
               by hand) solves T4 more often

Stage 1  A' is rebuilt, not written by hand: R2's dear runs (T1..T3, seeds
         0..9) and reflect.improve. If the change is not R2's, the run
         stops. T4 is in no experience.
Stage 2  the search, and each look, on fresh seeds, budget 400k, and
         core.gates.judge:
             dev pairs        T4 seeds 20..49
             hidden           T4 seeds 50..79
             counterexamples  W0, W3, W4 seeds 30..39 and T1, T2, T3 seeds
                              50..59: a question the search solved and
                              the look did not

    python -X utf8 scripts/run_look_claim.py [workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p2 as P2  # noqa: E402
import run_reflect as R2  # noqa: E402

from mana.core import gates  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import reflect  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery import selection  # noqa: E402

BUDGET = 400000
BUDGETS = (50000, 100000, 200000, 400000)
SETS = {"dev": [("T4", s) for s in range(20, 50)],
        "hidden": [("T4", s) for s in range(50, 80)],
        "counter": [(n, s) for n in ("W0", "W3", "W4") for s in range(30, 40)]
        + [(n, s) for n in R2.TRAIN for s in range(50, 60)]}


def run_job(job):
    arm, name, seed, policy = job
    world = R2.ALL[name]
    split = world.split(200, 300, seed)
    found = P.run(policy, split.train, split.train_outcomes, profile=True)
    solved = []
    for b in BUDGETS:
        program, _, _ = discovery.at_budget(found, b)
        solved.append(world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0)
    return arm, name, seed, solved


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(R2.experience_job, [("A", n, s) for n in R2.TRAIN for s in range(10)]))
        change = reflect.improve(P.CURRENT, [r[5] for r in rows if r[4]])
        steps = [what.split(":")[0] for what, _ in change.steps]
        print(f"стадия 1: опыт {sum(r[4] for r in rows)} выводов, изменение: {steps}; "
              f"{time.time() - started:.0f}с")
        if steps != P2.R2_CHANGE:
            print("изменение не совпало с R2 — прогон остановлен")
            return
        narrow = change.after.rules
        bare = tuple(r for r in narrow if not r.name.startswith("grown"))
        arms = {"search": P.with_budget(P.CURRENT, BUDGET),
                "look": P2.arm_policy((selection.look(256), narrow), BUDGET),
                "width": P2.arm_policy((selection.look(256), bare), BUDGET)}
        jobs = [(arm, n, s, policy) for arm, policy in arms.items()
                for items in SETS.values() for n, s in items]
        runs = list(pool.map(run_job, jobs))
    print(f"всего {time.time() - started:.0f}с")
    at = BUDGETS.index(BUDGET)
    got = {(arm, n, s): solved for arm, n, s, solved in runs}

    def acc(arm, items):
        return sum(got[(arm, n, s)][at] for n, s in items) / len(items)

    print(f"\n=== вердикт core.gates, бюджет {BUDGET // 1000}k, объявлен до прогона")
    for arm, claim in (("look", gates.Claim("P2c-look", "program",
                                            "взгляд 256 через правила A' решает T4 чаще поиска",
                                            asserts_domains=("T4",))),
                       ("width", gates.Claim("P2c-width", "program",
                                             "взгляд 256 через A' без хода решает T4 чаще поиска",
                                             asserts_domains=("T4",)))):
        counter = [(n, s) for n, s in SETS["counter"]
                   if got[("search", n, s)][at] and not got[(arm, n, s)][at]]
        evidence = gates.Evidence(
            paired_dev=[gates.PairedOutcome(f"T4/{s}", "T4", got[("search", n, s)][at],
                                            got[(arm, n, s)][at]) for n, s in SETS["dev"]],
            baseline_hidden=acc("search", SETS["hidden"]), candidate_hidden=acc(arm, SETS["hidden"]),
            baseline_hidden_by_domain={"T4": acc("search", SETS["hidden"])},
            candidate_hidden_by_domain={"T4": acc(arm, SETS["hidden"])},
            counterexamples_sought=len(SETS["counter"]), counterexamples_found=len(counter))
        verdict = gates.judge(claim, evidence)
        m = verdict.measurements
        print(f"  {claim.claim_id}: {verdict.status} — {verdict.reason}")
        print(f"      пары: {m['dev_baseline']} -> {m['dev_candidate']}, McNemar b={m['mcnemar']['b']} "
              f"c={m['mcnemar']['c']} p={m['mcnemar']['p_value']:.6f}; скрытые "
              f"{evidence.baseline_hidden:.2f} -> {evidence.candidate_hidden:.2f}; контрпримеров "
              f"{len(counter)}/{len(SETS['counter'])} {counter}")
    print("\n  решено при бюджете (пары T4 30 | скрытые T4 30 | контрпримеры: W0-W4 30, T1-T3 30):")
    print("    бюджет  " + "".join(f"{b // 1000:>19d}k" for b in BUDGETS))
    for arm in arms:
        line = f"    {arm:6s}  "
        for i in range(len(BUDGETS)):
            line += (f"{sum(got[(arm, n, s)][i] for n, s in SETS['dev']):>5d}"
                     f"{sum(got[(arm, n, s)][i] for n, s in SETS['hidden']):>4d}"
                     f"{sum(got[(arm, n, s)][i] for n, s in SETS['counter'][:30]):>5d}"
                     f"{sum(got[(arm, n, s)][i] for n, s in SETS['counter'][30:]):>5d} ")
        print(line)


if __name__ == "__main__":
    main()
