"""Stage R1: the search as an object -- does it reproduce, and does it control?

Part 1  equivalence: search.search and the policy CURRENT, interpreted, on
        the seven questions x seeds 0..9 at the search's own budget, with
        the anytime profile -- program, bits, evaluations, rounds, history,
        where the answer was first seen, why it stopped, the whole budget
        curve. All must agree.
Part 2  control, fixed with the user: deliberately meaningless changes of
        the object, same question, seed and budget, and a prediction of
        what each does, written before the run (mana/discovery/__init__.py):
            keep 2       the beam of 2 instead of 4
            reversed     the rules in the opposite order
            no combine   without "combine with a leaf"

    python scripts/run_policy.py [workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.discovery import policy as P  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery.language import show  # noqa: E402
from mana.discovery.worlds import FAMILY, WORLDS  # noqa: E402

TASKS = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")
ALL = {**WORLDS, **FAMILY}
VARIANTS = {
    "keep 2": P.with_keep(P.CURRENT, 2),
    "reversed": P.reordered(P.CURRENT, list(reversed(range(len(P.CURRENT.rules))))),
    "no combine": P.without(P.CURRENT, "combine with a leaf"),
}


def _summary(found, world):
    rule = world.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    return {"program": show(found.program), "bits": found.bits, "evaluations": found.evaluations,
            "rounds": found.rounds, "found_at": found.found_at,
            "termination": found.termination, "rule": rule,
            "per_round": found.evaluations / max(found.rounds, 1)}


def equivalence_job(job):
    name, seed = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    a = discovery.search(split.train, split.train_outcomes, profile=True)
    b = P.run(P.CURRENT, split.train, split.train_outcomes, profile=True)
    fields = ("program", "bits", "evaluations", "rounds", "history", "found_at",
              "termination", "train_errors", "anytime")
    differ = [f for f in fields if getattr(a, f) != getattr(b, f)]
    return name, seed, differ, a.seconds, b.seconds, _summary(a, world)


def variant_job(job):
    name, seed, label = job
    world = ALL[name]
    split = world.split(200, 300, seed)
    return name, seed, label, _summary(P.run(VARIANTS[label], split.train, split.train_outcomes),
                                       world)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    print("политика CURRENT:\n" + P.describe(P.CURRENT))
    for label, variant in VARIANTS.items():
        print(f"  {label}: {P.difference(P.CURRENT, variant)}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        same = list(pool.map(equivalence_job, [(n, s) for n in TASKS for s in range(10)]))
        base = {(n, s): summary for n, s, _, _, _, summary in same}
        varied = list(pool.map(variant_job, [(n, s, label) for label in VARIANTS
                                             for n in TASKS for s in range(10)]))
    print(f"\n{time.time() - started:.0f}с")
    exact = [row for row in same if not row[2]]
    print(f"\n=== часть 1: тождество — совпало целиком {len(exact)}/{len(same)}")
    for name, seed, differ, *_ in same:
        if differ:
            print(f"    {name} сид {seed}: расходятся {differ}")
    slow = sum(row[4] for row in same) / max(sum(row[3] for row in same), 1e-9)
    print(f"  интерпретатор медленнее search.search в {slow:.1f} раза")

    print("\n=== часть 2: бессмысленные правки объекта (решено правило / всего, по задачам)")
    print("  задача    " + "".join(f"{label:>22s}" for label in ["как есть"] + list(VARIANTS)))
    for name in TASKS:
        line = f"  {name:8s}  {sum(base[(name, s)]['rule'] for s in range(10)):>18d}/10"
        for label in VARIANTS:
            rows = [r for n, s, lab, r in varied if n == name and lab == label]
            line += f"{sum(r['rule'] for r in rows):>19d}/10"
        print(line)
    for label in VARIANTS:
        rows = [(n, s, r) for n, s, lab, r in varied if lab == label]
        same_answer = sum(r["program"] == base[(n, s)]["program"] for n, s, r in rows)
        same_all = sum(all(r[k] == base[(n, s)][k] for k in ("program", "evaluations", "rounds",
                                                             "found_at")) for n, s, r in rows)
        per_round = sum(r["per_round"] for _, _, r in rows) / sum(
            base[(n, s)]["per_round"] for n, s, _ in rows)
        print(f"\n  {label}: тот же ответ {same_answer}/70, совпало всё {same_all}/70, "
              f"программ на раунд — {per_round:.2f} от исходного")
        if label == "reversed":
            stopped = [(n, s, r) for n, s, r in rows if base[(n, s)]["termination"] == discovery.SEARCH_EXHAUSTED]
            cut = [(n, s, r) for n, s, r in rows if base[(n, s)]["termination"] != discovery.SEARCH_EXHAUSTED]
            print(f"    где исходный поиск остановился сам: совпало всё "
                  f"{sum(all(r[k] == base[(n, s)][k] for k in ('program', 'evaluations', 'rounds')) for n, s, r in stopped)}"
                  f"/{len(stopped)}; где его оборвал бюджет: тот же ответ "
                  f"{sum(r['program'] == base[(n, s)]['program'] for n, s, r in cut)}/{len(cut)}")


if __name__ == "__main__":
    main()
