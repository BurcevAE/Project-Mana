"""Stage R2c: the speed claim, declared before the run, on seeds nobody has seen.

The claim (mana/discovery/__init__.py, written before this ran): the policy
MANA built in R2b solves more questions of its family within 100 000
programs than the policy it replaced.

Stage 1  the policy is rebuilt, not written by hand: R2b's experience
         (T1..T3, W0, W3, seeds 0..9) and reflect.improve, exactly as in
         R2b. If the change is not R2b's, the run stops.
Stage 2  old and new on fresh seeds, budget 100k, and core.gates.judge:
             dev pairs        T1..T3 seeds 30..39
             hidden           T1..T3 seeds 40..49
             counterexamples  W0, W3, W4 seeds 20..29
             transfer         T4 seeds 10..19, a second claim

    python scripts/run_speed_claim.py [workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_reflect as R2  # noqa: E402

from mana.core import gates  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import reflect  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402

BUDGET = 100000
BUDGETS = (10000, 25000, 50000, 100000)
TRAIN = R2.TRAIN
#: What R2b's change was: the run stops if the rebuilt one differs.
R2B_CHANGE = ["добавлено правило grown m13", "добавлено правило grown m14",
              "убрано правило «replace by a leaf»", "убрано правило «swap add»"]
SETS = {"dev": [(n, s) for n in TRAIN for s in range(30, 40)],
        "hidden": [(n, s) for n in TRAIN for s in range(40, 50)],
        "counter": [(n, s) for n in ("W0", "W3", "W4") for s in range(20, 30)],
        "transfer": [("T4", s) for s in range(10, 20)]}


def run_job(job):
    arm, name, seed, policy = job
    world = R2.ALL[name]
    split = world.split(200, 300, seed)
    found = P.run(P.with_budget(policy, BUDGET), split.train, split.train_outcomes,
                  profile=True)
    solved = []
    for b in BUDGETS:
        program, _, _ = discovery.at_budget(found, b)
        solved.append(world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0)
    return arm, name, seed, solved


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    variant = R2.VARIANTS["R2b"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(R2.experience_job, [("A", n, s) for n in variant["experience"]
                                                 for s in range(10)]))
        experience = [r[5] for r in rows if r[4]]
        change = reflect.improve(R2.STARTS["A"], experience)
        steps = [what.split(":")[0] for what, _ in change.steps]
        print(f"стадия 1: опыт {len(experience)} выводов, изменение: {steps}; "
              f"{time.time() - started:.0f}с")
        if steps != R2B_CHANGE:
            print("изменение не совпало с R2b — прогон остановлен")
            return
        jobs = [(arm, n, s, policy) for arm, policy in (("old", change.before), ("new", change.after))
                for items in SETS.values() for n, s in items]
        runs = list(pool.map(run_job, jobs))
    print(f"всего {time.time() - started:.0f}с")
    at = BUDGETS.index(BUDGET)
    got = {(arm, n, s): solved for arm, n, s, solved in runs}

    def acc(arm, items):
        return sum(got[(arm, n, s)][at] for n, s in items) / len(items)

    def by_domain(arm, items):
        return {d: sum(got[(arm, n, s)][at] for n, s in items if n == d) /
                sum(1 for n, _ in items if n == d) for d in TRAIN}

    evidence = gates.Evidence(
        paired_dev=[gates.PairedOutcome(f"{n}/{s}", n, got[("old", n, s)][at],
                                        got[("new", n, s)][at]) for n, s in SETS["dev"]],
        baseline_hidden=acc("old", SETS["hidden"]), candidate_hidden=acc("new", SETS["hidden"]),
        baseline_hidden_by_domain=by_domain("old", SETS["hidden"]),
        candidate_hidden_by_domain=by_domain("new", SETS["hidden"]),
        baseline_transfer=acc("old", SETS["transfer"]),
        candidate_transfer=acc("new", SETS["transfer"]),
        counterexamples_sought=len(SETS["counter"]),
        counterexamples_found=sum(got[("old", n, s)][at] and not got[("new", n, s)][at]
                                  for n, s in SETS["counter"]))
    print(f"\n=== вердикт core.gates, бюджет {BUDGET // 1000}k, объявлен до прогона")
    for claim in (gates.Claim("R2c-speed", "program",
                              "политика, построенная MANA в R2b, решает больше в пределах 100k программ",
                              asserts_domains=TRAIN),
                  gates.Claim("R2c-speed-transfer", "program",
                              "то же, с переносом на T4", asserts_transfer=True,
                              asserts_domains=TRAIN)):
        verdict = gates.judge(claim, evidence)
        m = verdict.measurements
        print(f"  {claim.claim_id}: {verdict.status} — {verdict.reason}")
        print(f"      пары: {m['dev_baseline']} -> {m['dev_candidate']}, McNemar b={m['mcnemar']['b']} "
              f"c={m['mcnemar']['c']} p={m['mcnemar']['p_value']:.4f}; скрытые "
              f"{evidence.baseline_hidden:.2f} -> {evidence.candidate_hidden:.2f}; контрпримеров "
              f"{evidence.counterexamples_found}/{evidence.counterexamples_sought}; T4 "
              f"{evidence.baseline_transfer:.1f} -> {evidence.candidate_transfer:.1f}")
    print("\n  кривые, решено (пары 30 | скрытые 30 | W0-W4 30 | T4 10):")
    print("    бюджет  " + "".join(f"{b // 1000:>16d}k" for b in BUDGETS))
    for arm in ("old", "new"):
        line = f"    {'старая' if arm == 'old' else 'новая':6s}  "
        for i in range(len(BUDGETS)):
            line += "".join(f"{sum(got[(arm, n, s)][i] for n, s in SETS[k]):>4d}"
                            for k in ("dev", "hidden", "counter", "transfer")) + " "
        print(line)


if __name__ == "__main__":
    main()
