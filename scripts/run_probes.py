"""Experiment M, step M1b: experiments about methods, weighed with applying them.

Part 1, the M1 world, twice: under M1's protocol (what was learnt frozen at
test) and learning through the test. On the same boxes: the M1 Chooser, the
Explorer, the cascade in the writer's order, the oracle.

Part 2, the trap (trap.py): explosive, 20x cheaper at n = 2..4, dearer from
6 and past its budget at 8; steady, linear. 20 training boxes at n = 2..4,
then 12 at n = 5, 6, 8 in turn -- and again with 8 first, the trap as
posed -- learning throughout. The Chooser, the Explorer, the oracle.

And the Explorer's trajectory: what it expects of the methods after its
1st, 5th, 10th, 20th, 40th probe, and at the end of each phase.

    python scripts/run_probes.py [streams] [workers] [out.json]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.methods import choice, portfolio, solvers, trap  # noqa: E402
from mana.methods.world import LEVELS, LINEAR, STRUCTURES, BlackBox  # noqa: E402

TRAIN_N = (2, 3, 4)
TEST_N = (5, 6, 8)
MARKS = (1, 5, 10, 20, 40)
LETTER = {"exhaust": "A", "calculate": "B", "decompose": "C", "experiment": "D",
          "explosive": "X", "steady": "S"}
FAILED_BC = frozenset({"calculate", "decompose"})


def _word(methods):
    return "".join(LETTER[m] for m in methods)


def _oracle(box_of, n, levels, box_seed, registry=None):
    box = box_of()
    attempts = {m: solvers.attempt(m, box.answer, n, levels, box_seed, announce=False,
                                   methods=registry)
                for m in (registry or solvers.METHODS)}
    verdicts = {m: box.verdict(a.model) for m, a in attempts.items()}
    best = portfolio.oracle(attempts, verdicts)
    return {"method": best.method, "cost": best.cost}


def _snapshot(explorer, label, questions):
    view = {"label": label, "probes": explorer.probes, "drift": explorer.journal.drift()}
    for key, (method, before, n) in questions.items():
        seen = explorer.expect(method, before, n)
        view[key] = None if seen is None else [seen[0], seen[1]]
    return view


M1_QUESTIONS = {"D|BC@8": ("experiment", FAILED_BC, 8),
                "A|BC@6": ("exhaust", FAILED_BC, 6),
                "A|BC@8": ("exhaust", FAILED_BC, 8)}
TRAP_QUESTIONS = {"X@6": ("explosive", frozenset(), 6), "X@8": ("explosive", frozenset(), 8),
                  "S@8": ("steady", frozenset(), 8)}


def m1_stream(job):
    seed, protocol = job
    rng = np.random.default_rng([seed, 11])
    train = [(s, int(rng.choice(TRAIN_N))) for s in STRUCTURES for _ in range(10)]
    rng.shuffle(train)
    test = [(s, n) for s in STRUCTURES for n in TEST_N for _ in range(3)]
    chooser = choice.Chooser(choice.Experience(), seed=seed)
    explorer = choice.Explorer(seed=seed)
    rows, marks = [], []
    for phase, boxes in (("train", train), ("test", test)):
        for i, (structure, n) in enumerate(boxes):
            box_seed = seed * 1000 + (i if phase == "train" else 500 + i)
            learn = phase == "train" or protocol == "learning"

            def box_of():
                return BlackBox(n, structure, box_seed)

            box = box_of()
            picked, model = chooser.solve(box.answer, n, LEVELS, box_seed, learn=learn)
            m1 = {"cost": picked.cost, "right": box.verdict(model) == 1.0,
                  "tried": _word(picked.tried)}
            box = box_of()
            had = explorer.probes
            picked, model = explorer.solve(box.answer, n, LEVELS, box_seed, learn=learn)
            ex = {"cost": picked.cost, "right": box.verdict(model) == 1.0,
                  "tried": _word(picked.tried),
                  "probes": [[LETTER[m], size, q] for m, size, q in explorer.last_probes]}
            box = box_of()
            fixed, model, _ = portfolio.run_order(box.answer, n, LEVELS, box_seed)
            cascade = {"cost": fixed.cost, "right": box.verdict(model) == 1.0}
            rows.append({"stream": seed, "protocol": protocol, "phase": phase, "index": i,
                         "structure": structure, "n": n, "m1": m1, "explorer": ex,
                         "cascade": cascade,
                         "oracle": _oracle(box_of, n, LEVELS, box_seed)})
            for mark in MARKS:
                if had < mark <= explorer.probes:
                    marks.append(_snapshot(explorer, f"{mark}-я проба ({phase} {i + 1})",
                                           M1_QUESTIONS))
        marks.append(_snapshot(explorer, f"конец {phase}", M1_QUESTIONS))
    return rows, marks


#: The order of the trap's test boxes: sizes 5 and 6 before 8 let ordinary
#: learning bend the line first; 8 first is the trap as posed -- after
#: n = 2..4 only, straight to n = 8.
TRAP_ORDERS = {"5-6-8": (5, 6, 8), "8-5-6": (8, 5, 6)}


def trap_stream(job):
    seed, order = job
    rng = np.random.default_rng([seed, 13])
    train = [int(rng.choice(TRAIN_N)) for _ in range(20)]
    test = list(TRAP_ORDERS[order]) * 4
    names = tuple(trap.METHODS)
    chooser = choice.Chooser(choice.Experience(methods=names), seed=seed, registry=trap.METHODS)
    explorer = choice.Explorer(methods=names, registry=trap.METHODS, seed=seed)
    rows, marks = [], []
    for phase, sizes in (("train", train), ("test", test)):
        for i, n in enumerate(sizes):
            box_seed = seed * 1000 + (i if phase == "train" else 500 + i)

            def box_of():
                return BlackBox(n, LINEAR, box_seed, levels=trap.LEVELS)

            box = box_of()
            picked, model = chooser.solve(box.answer, n, trap.LEVELS, box_seed)
            m1 = {"cost": picked.cost, "right": box.verdict(model) == 1.0,
                  "tried": _word(picked.tried),
                  "expects_X8": chooser.experience.cost("explosive", 8, frozenset())}
            box = box_of()
            had = explorer.probes
            picked, model = explorer.solve(box.answer, n, trap.LEVELS, box_seed)
            ex = {"cost": picked.cost, "right": box.verdict(model) == 1.0,
                  "tried": _word(picked.tried),
                  "probes": [[LETTER[m], size, q] for m, size, q in explorer.last_probes]}
            rows.append({"stream": seed, "order": order, "phase": phase, "index": i, "n": n, "m1": m1,
                         "explorer": ex,
                         "oracle": _oracle(box_of, n, trap.LEVELS, box_seed, trap.METHODS)})
            for mark in MARKS:
                if had < mark <= explorer.probes:
                    marks.append(_snapshot(explorer, f"{mark}-я проба ({phase} {i + 1})",
                                           TRAP_QUESTIONS))
            if phase == "test":
                marks.append(_snapshot(explorer, f"после ящика test {i + 1} (n={n})",
                                       TRAP_QUESTIONS))
        marks.append(_snapshot(explorer, f"конец {phase}", TRAP_QUESTIONS))
    return rows, marks


def _fmt(view, keys):
    parts = [f"проб {view['probes']:>3d}", f"дрейф {view['drift']:.2f}"]
    for key in keys:
        seen = view.get(key)
        parts.append(f"{key} " + ("—" if seen is None else f"{seen[0]:>11.0f} ±{seen[1]:.1f}"))
    return "  ".join(parts)


def report_m1(rows, marks, protocol):
    arms = (("M1", "m1"), ("Explorer", "explorer"), ("каскад", "cascade"))
    rows = [r for r in rows if r["protocol"] == protocol]
    test = [r for r in rows if r["phase"] == "test"]
    print(f"\n=== мир M1, протокол «{protocol}»: проверка, решено / медиана вопросов")
    print("  мир       n " + "".join(f"{label:>18s}" for label, _ in arms) + f"{'оракул':>18s}"
          + "   пробы Explorer (метод+размер)")
    for structure in STRUCTURES:
        for n in TEST_N:
            mine = [r for r in test if r["structure"] == structure and r["n"] == n]
            line = f"  {structure:8s} {n:>2d} "
            for _, key in arms:
                line += (f"{sum(r[key]['right'] for r in mine):>8d}/{len(mine):<2d}"
                         f"{int(np.median([r[key]['cost'] for r in mine])):>8d}")
            line += (f"{sum(r['oracle']['method'] is not None for r in mine):>8d}/{len(mine):<2d}"
                     f"{int(np.median([r['oracle']['cost'] for r in mine])):>8d}")
            probes = Counter(f"{m}{size}" for r in mine for m, size, _ in r["explorer"]["probes"])
            line += "   " + (" ".join(f"{k}:{v}" for k, v in probes.most_common(4)) or "—")
            print(line)
    for name, part in (("решаемые", [r for r in test if r["oracle"]["method"] is not None]),
                       ("безнадёжные", [r for r in test if r["oracle"]["method"] is None])):
        line = f"  {name:12s}"
        for label, key in arms + (("оракул", "oracle"),):
            solved = sum((r[key]["right"] if key != "oracle" else r["oracle"]["method"] is not None)
                         for r in part)
            line += f"  {label} {solved}/{len(part)} {sum(r[key]['cost'] for r in part):>9d}"
        print(line)
    train = [r for r in rows if r["phase"] == "train"]
    print(f"  проб на ящик: обучение {np.mean([len(r['explorer']['probes']) for r in train]):.2f}, "
          f"проверка {np.mean([len(r['explorer']['probes']) for r in test]):.2f}; "
          f"вопросов на пробы в проверке {sum(q for r in test for _, _, q in r['explorer']['probes'])}")
    print("  обучение, вопросы / оракул по десяткам ящиков:  "
          + "  ".join(f"M1 {sum(r['m1']['cost'] for r in b) / sum(r['oracle']['cost'] for r in b):.2f}"
                      f" Ex {sum(r['explorer']['cost'] for r in b) / sum(r['oracle']['cost'] for r in b):.2f}"
                      for b in ([r for r in train if s <= r["index"] < s + 10] for s in range(0, 40, 10))))
    for seed in (0, 1):
        print(f"  траектория, поток {seed}:")
        for view in marks.get((seed, protocol), []):
            print(f"    {view['label']:28s} {_fmt(view, M1_QUESTIONS)}")


def report_trap(rows, marks, order):
    rows = [r for r in rows if r["order"] == order]
    marks = {seed: m for (seed, o), m in marks.items() if o == order}
    test = [r for r in rows if r["phase"] == "test"]
    print(f"\n=== ловушка, порядок проверки {order}: по ящикам, сумма по потокам "
          "(сколько потоков сожгли бюджет X)")
    print("  ящик  n        M1: вопросов  сожгли    Explorer: вопросов  сожгли  пробы     оракул")
    for i in range(12):
        mine = [r for r in test if r["index"] == i]
        burnt_m1 = sum(r["m1"]["cost"] >= solvers.BUDGET for r in mine)
        burnt_ex = sum(r["explorer"]["cost"] >= solvers.BUDGET for r in mine)
        probes = sum(len(r["explorer"]["probes"]) for r in mine)
        print(f"  {i + 1:>4d} {mine[0]['n']:>2d}  {sum(r['m1']['cost'] for r in mine):>18d}"
              f"  {burnt_m1:>6d}  {sum(r['explorer']['cost'] for r in mine):>18d}  {burnt_ex:>6d}"
              f"  {probes:>5d}  {sum(r['oracle']['cost'] for r in mine):>9d}")
    for label, key in (("M1", "m1"), ("Explorer", "explorer")):
        print(f"  {label:9s} решено {sum(r[key]['right'] for r in test)}/{len(test)}, "
              f"вопросов {sum(r[key]['cost'] for r in test)}; "
              f"порядки на n=8: {Counter(r[key]['tried'] for r in test if r['n'] == 8).most_common(4)}")
    print(f"  оракул    вопросов {sum(r['oracle']['cost'] for r in test)}")
    train = [r for r in rows if r["phase"] == "train"]
    print(f"  обучение: M1 {sum(r['m1']['cost'] for r in train)}, "
          f"Explorer {sum(r['explorer']['cost'] for r in train)}, "
          f"оракул {sum(r['oracle']['cost'] for r in train)}; проб Explorer "
          f"{sum(len(r['explorer']['probes']) for r in train)}")
    for seed in (0, 1):
        print(f"  траектория, поток {seed}:")
        for view in marks.get(seed, []):
            print(f"    {view['label']:28s} {_fmt(view, TRAP_QUESTIONS)}")


def main() -> None:
    streams = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    started = time.time()
    jobs = [(seed, protocol) for protocol in ("frozen", "learning") for seed in range(streams)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        m1_parts = list(pool.map(m1_stream, jobs))
        trap_jobs = [(seed, order) for order in TRAP_ORDERS for seed in range(streams)]
        trap_parts = list(pool.map(trap_stream, trap_jobs))
    m1_rows = [row for rows, _ in m1_parts for row in rows]
    m1_marks = {job: marks for job, (_, marks) in zip(jobs, m1_parts)}
    trap_rows = [row for rows, _ in trap_parts for row in rows]
    trap_marks = {job: marks for job, (_, marks) in zip(trap_jobs, trap_parts)}
    print(f"{time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps({"m1": m1_rows, "m1_marks": {f"{k[0]}/{k[1]}": v for k, v in m1_marks.items()},
                                   "trap": trap_rows,
                                   "trap_marks": {f"{k[0]}/{k[1]}": v for k, v in trap_marks.items()}},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сырые данные: {out}")
    for protocol in ("frozen", "learning"):
        report_m1(m1_rows, m1_marks, protocol)
    for order in TRAP_ORDERS:
        report_trap(trap_rows, trap_marks, order)


if __name__ == "__main__":
    main()
