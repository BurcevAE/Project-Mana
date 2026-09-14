"""Stage R2: MANA builds a change of its own search from its own experience,
and core.gates decides.

For each starting policy -- A, the search as it stands; B, the same with an
idle rule added ("double": n -> n + n) and "swap <" taken away, to see
whether the change follows the policy and its experience rather than being
one fixed trick:

Stage 1  experience: on T1, T2, T3, seeds 0..9, a cheap run of the policy
         (beam 4, 400k) and a dear one (beam 16, 2M) with its derivation.
Stage 2  the change: reflect.improve -- the policy in which the dear runs'
         derivations are written most briefly (rules added from repeated
         structure, rules dropped the experience never used).
Stage 3  the verdict, from core.gates, budget 400k fixed in advance:
             dev pairs       T1..T3 seeds 10..19 (30 pairs)
             hidden          T1..T3 seeds 20..29
             counterexamples W0, W3, W4 seeds 0..9: a run the old policy
                             solved and the new one did not
             transfer        T4 seeds 0..9, a second claim that asserts it
         and the recall curves up to 400k, at equal programs evaluated.

R2b (the user's control, after R2): the same generator and the same gates,
the experience widened to every training family -- T1, T2, T3, W0, W3,
seeds 0..9 -- to tell whether R2's specialist came from a narrow
experience or from the principle itself. W0, W3 now teach, so their
counterexample runs move to unseen seeds 10..19; W4 stays out of the
experience.

    python scripts/run_reflect.py [workers] [R2 | R2b]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.core import gates  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery import reflect  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import add  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

ALL = {**WORLDS, **FAMILY}
TRAIN = ("T1", "T2", "T3")
#: What each variant learns from, and where its counterexamples are sought.
VARIANTS = {"R2": {"experience": TRAIN, "counter_seeds": range(10), "starts": ("A", "B")},
            "R2b": {"experience": TRAIN + ("W0", "W3"), "counter_seeds": range(10, 20),
                    "starts": ("A",)}}
BUDGET = 400000
BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000)
IDLE = P.Rule("double", P.N, (add(P.N, P.N),))
STARTS = {
    "A": P.CURRENT,
    "B": P.with_rule(P.without(P.CURRENT, "swap <"), IDLE),
}


def label(b):
    return f"{b // 1000}k"


def experience_job(job):
    start, name, seed = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    policy = STARTS[start]
    cheap = P.run(policy, split.train, split.train_outcomes)
    dear = P.run(P.with_budget(P.with_keep(policy, 16), 2000000), split.train,
                 split.train_outcomes, trace=True)
    grade = lambda p: world.grade(lambda cols: discovery.predict(p, cols)) == 1.0  # noqa: E731
    leaves, conditions = discovery.vocabulary(split.train, split.train_outcomes)
    return (start, name, seed, grade(cheap.program), grade(dear.program),
            reflect.Experience(dear.derivation, leaves, conditions, name))


def run_job(job):
    start, arm, name, seed, policy = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    found = P.run(policy, split.train, split.train_outcomes, profile=True)
    solved = []
    for b in BUDGETS:
        program, _, _ = discovery.at_budget(found, b)
        solved.append(world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0)
    return start, arm, name, seed, solved


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    variant = VARIANTS[sys.argv[2] if len(sys.argv) > 2 else "R2"]
    starts = variant["starts"]
    started = time.time()
    changes = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(experience_job, [(s, n, seed) for s in starts
                                              for n in variant["experience"]
                                              for seed in range(10)]))
        for start in starts:
            mine = [r for r in rows if r[0] == start]
            experience = [r[5] for r in mine if r[4]]
            print(f"\n=== политика {start}: дешёвый прогон решил {sum(r[3] for r in mine)}/{len(mine)}, "
                  f"дорогой {sum(r[4] for r in mine)}/{len(mine)}; опыт — {len(experience)} выводов; "
                  f"{time.time() - started:.0f}с")
            change = reflect.improve(STARTS[start], experience)
            changes[start] = change
            print(f"  запись опыта: {change.bits_before:.1f} -> {change.bits_after:.1f} бит "
                  f"(кандидатов {change.considered})")
            for what, saved in change.steps:
                print(f"    {what}  (-{saved:.1f} бит)")
            print("  разница с исходной:", P.difference(STARTS[start], change.after))
        jobs = []
        sets = {"dev": [(n, s) for n in TRAIN for s in range(10, 20)],
                "hidden": [(n, s) for n in TRAIN for s in range(20, 30)],
                "counter": [(n, s) for n in ("W0", "W3", "W4") for s in variant["counter_seeds"]],
                "transfer": [("T4", s) for s in range(10)]}
        for start, change in changes.items():
            for arm, policy in (("old", change.before), ("new", change.after)):
                for which, items in sets.items():
                    jobs += [(start, arm, n, s, policy) for n, s in items]
        runs = list(pool.map(run_job, jobs))
    print(f"\nвсего {time.time() - started:.0f}с")
    at = BUDGETS.index(BUDGET)
    for start in starts:
        got = {(arm, n, s): solved for st, arm, n, s, solved in runs if st == start}

        def acc(arm, items, i=at):
            return sum(got[(arm, n, s)][i] for n, s in items) / len(items)

        def by_domain(arm, items):
            return {n: sum(got[(arm, m, s)][at] for m, s in items if m == n) /
                    sum(1 for m, _ in items if m == n) for n in TRAIN}

        paired = [gates.PairedOutcome(f"{n}/{s}", n, got[("old", n, s)][at], got[("new", n, s)][at])
                  for n, s in sets["dev"]]
        found_counter = sum(got[("old", n, s)][at] and not got[("new", n, s)][at]
                            for n, s in sets["counter"])
        evidence = gates.Evidence(
            paired_dev=paired,
            baseline_hidden=acc("old", sets["hidden"]), candidate_hidden=acc("new", sets["hidden"]),
            baseline_hidden_by_domain=by_domain("old", sets["hidden"]),
            candidate_hidden_by_domain=by_domain("new", sets["hidden"]),
            baseline_transfer=acc("old", sets["transfer"]),
            candidate_transfer=acc("new", sets["transfer"]),
            counterexamples_sought=len(sets["counter"]), counterexamples_found=found_counter)
        print(f"\n=== политика {start}: вердикт core.gates (бюджет {label(BUDGET)})")
        for claim in (gates.Claim(f"R2-{start}", "program", "самоизменение политики поиска",
                                  asserts_domains=TRAIN),
                      gates.Claim(f"R2-{start}-transfer", "program",
                                  "самоизменение политики поиска, с переносом на T4",
                                  asserts_transfer=True, asserts_domains=TRAIN)):
            verdict = gates.judge(claim, evidence)
            m = verdict.measurements
            print(f"  {claim.claim_id}: {verdict.status} — {verdict.reason}; dev {m['dev_baseline']} -> "
                  f"{m['dev_candidate']}, McNemar b={m['mcnemar']['b']} c={m['mcnemar']['c']} "
                  f"p={m['mcnemar']['p_value']:.3f}; скрытые {evidence.baseline_hidden:.2f} -> "
                  f"{evidence.candidate_hidden:.2f}; контрпримеров {found_counter}/"
                  f"{len(sets['counter'])}; T4 {evidence.baseline_transfer:.1f} -> "
                  f"{evidence.candidate_transfer:.1f}")
        print("  кривые полноты, решено (dev, 30 | скрытые, 30 | T4, 10 | W0-W4, 30):")
        print("    бюджет  " + "".join(f"{label(b):>18s}" for b in BUDGETS))
        for arm in ("old", "new"):
            line = f"    {'старая' if arm == 'old' else 'новая':6s}  "
            for i in range(len(BUDGETS)):
                line += (f"{sum(got[(arm, n, s)][i] for n, s in sets['dev']):>5d}"
                         f"{sum(got[(arm, n, s)][i] for n, s in sets['hidden']):>4d}"
                         f"{sum(got[(arm, n, s)][i] for n, s in sets['transfer']):>4d}"
                         f"{sum(got[(arm, n, s)][i] for n, s in sets['counter']):>5d}")
            print(line)
    if "B" in changes:
        a, b = changes["A"].after, changes["B"].after
        print("\n=== A' против B':", P.difference(a, b) or "одинаковы")


if __name__ == "__main__":
    main()
