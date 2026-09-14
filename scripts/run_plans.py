"""Experiment M, step M1c: a probe is worth what it changes in the best plan.

The M1 world and the trap (trap.py), each under two protocols -- what was
learnt frozen at test, and learning through it -- so that choosing what to
learn in advance is told apart from learning it along the way. On the same
boxes and seeds as M1 and M1b: the M1 Chooser, the Planner twice, the
cascade in the writer's order, the oracle. The Explorer of M1b ran on these
very boxes; its figures are in mana/methods/__init__.py.

The Planner's two arms differ only in how a probe's worth is reckoned
(choice.py): Δ -- the cost of the best plan now less the expected cost of
the best plan after; изм -- the expected saving of the plan after over the
present plan, both weighed after the outcome, nothing where the plan would
not change.

Every decision of a Planner is kept: the plan, the action, for a probe the
plans its outcomes lead to and with what chance, the plan after the actual
outcome, and whether it repeated a probe the journal already held. The
report is read against the five criteria fixed before the run.

    python scripts/run_plans.py [streams] [workers] [out.json]
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
TRAP_ORDERS = {"5-6-8": (5, 6, 8), "8-5-6": (8, 5, 6)}
PROTOCOLS = ("frozen", "learning")
PLANNERS = (("Planner Δ", "planner", choice.DIFFERENCE), ("Planner изм", "change", choice.CHANGE))
LETTER = {"exhaust": "A", "calculate": "B", "decompose": "C", "experiment": "D",
          "explosive": "X", "steady": "S"}


def _word(methods):
    return "".join(LETTER[m] for m in methods) or "—"


def _decisions(planner):
    out = []
    for d in planner.last_decisions:
        out.append({"action": d.action, "method": d.method and LETTER[d.method], "size": d.size,
                    "before": _word(d.before), "plan": _word(d.plan.names),
                    "plan_cost": d.plan.cost, "plan_success": d.plan.success,
                    "worth": d.worth, "spend": d.spend,
                    "branches": [[_word(b.plan), b.probability, b.cost] for b in d.branches],
                    "change": d.predicted_change, "repeat": d.repeat,
                    "after": None if d.after is None else _word(d.after), "actual": d.actual})
    return out


def _oracle(box_of, n, levels, box_seed, registry=None):
    box = box_of()
    attempts = {m: solvers.attempt(m, box.answer, n, levels, box_seed, announce=False,
                                   methods=registry)
                for m in (registry or solvers.METHODS)}
    verdicts = {m: box.verdict(a.model) for m, a in attempts.items()}
    best = portfolio.oracle(attempts, verdicts)
    return {"method": best.method, "cost": best.cost}


def _play(agent, box_of, n, levels, box_seed, learn):
    box = box_of()
    picked, model = agent.solve(box.answer, n, levels, box_seed, learn=learn)
    out = {"cost": picked.cost, "right": box.verdict(model) == 1.0, "tried": _word(picked.tried)}
    if isinstance(agent, choice.Planner):
        out["decisions"] = _decisions(agent)
    return out


def m1_stream(job):
    seed, protocol = job
    rng = np.random.default_rng([seed, 11])
    train = [(s, int(rng.choice(TRAIN_N))) for s in STRUCTURES for _ in range(10)]
    rng.shuffle(train)
    test = [(s, n) for s in STRUCTURES for n in TEST_N for _ in range(3)]
    agents = {"m1": choice.Chooser(choice.Experience(), seed=seed)}
    for _, key, worth in PLANNERS:
        agents[key] = choice.Planner(seed=seed, worth=worth)
    rows = []
    for phase, boxes in (("train", train), ("test", test)):
        for i, (structure, n) in enumerate(boxes):
            box_seed = seed * 1000 + (i if phase == "train" else 500 + i)
            learn = phase == "train" or protocol == "learning"

            def box_of():
                return BlackBox(n, structure, box_seed)

            row = {"world": "M1", "stream": seed, "protocol": protocol, "phase": phase,
                   "index": i, "structure": structure, "n": n,
                   "oracle": _oracle(box_of, n, LEVELS, box_seed)}
            for key, agent in agents.items():
                row[key] = _play(agent, box_of, n, LEVELS, box_seed, learn)
            box = box_of()
            fixed, model, _ = portfolio.run_order(box.answer, n, LEVELS, box_seed)
            row["cascade"] = {"cost": fixed.cost, "right": box.verdict(model) == 1.0}
            rows.append(row)
    return rows


def trap_stream(job):
    seed, order, protocol = job
    rng = np.random.default_rng([seed, 13])
    train = [int(rng.choice(TRAIN_N)) for _ in range(20)]
    test = list(TRAP_ORDERS[order]) * 4
    names = tuple(trap.METHODS)
    agents = {"m1": choice.Chooser(choice.Experience(methods=names), seed=seed,
                                   registry=trap.METHODS)}
    for _, key, worth in PLANNERS:
        agents[key] = choice.Planner(methods=names, registry=trap.METHODS, seed=seed, worth=worth)
    rows = []
    for phase, sizes in (("train", train), ("test", test)):
        for i, n in enumerate(sizes):
            box_seed = seed * 1000 + (i if phase == "train" else 500 + i)
            learn = phase == "train" or protocol == "learning"

            def box_of():
                return BlackBox(n, LINEAR, box_seed, levels=trap.LEVELS)

            row = {"world": "trap", "stream": seed, "order": order, "protocol": protocol,
                   "phase": phase, "index": i, "n": n,
                   "oracle": _oracle(box_of, n, trap.LEVELS, box_seed, trap.METHODS)}
            for key, agent in agents.items():
                row[key] = _play(agent, box_of, n, trap.LEVELS, box_seed, learn)
            rows.append(row)
    return rows


def _probes(rows, key):
    return [d for r in rows for d in r[key]["decisions"] if d["action"] == "probe"]


def _criteria_2_3(rows, key, label):
    probes = _probes(rows, key)
    if not probes:
        print(f"    {label}: проб нет")
        return
    changed = [d["after"] != d["plan"] for d in probes]
    print(f"    {label}: проб {len(probes)}, вопросов на них {sum(d['actual'] for d in probes)}; "
          f"план изменился после {sum(changed)} ({np.mean(changed):.0%}), "
          f"ожидалось {np.mean([d['change'] for d in probes]):.0%}")
    for low, high in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        band = [d for d in probes if low <= d["change"] < high]
        if band:
            print(f"        ожидаемый шанс {low:.1f}-{min(high, 1):.1f}: проб {len(band):>4d}, "
                  f"ожидалось {np.mean([d['change'] for d in band]):.0%}, "
                  f"изменился {np.mean([d['after'] != d['plan'] for d in band]):.0%}")
    repeats = [d for d in probes if d["repeat"]]
    if repeats:
        which = Counter(str(d["method"]) + str(d["size"]) for d in repeats).most_common(4)
        print(f"        повторы уже купленной пробы: {len(repeats)}, план изменился после "
              f"{sum(d['after'] != d['plan'] for d in repeats)}; {which}")
    else:
        print("        повторов уже купленной пробы нет")


def report_m1(rows, protocol):
    rows = [r for r in rows if r["world"] == "M1" and r["protocol"] == protocol]
    arms = (("M1", "m1"),) + tuple((label, key) for label, key, _ in PLANNERS) + (("каскад", "cascade"),)
    test = [r for r in rows if r["phase"] == "test"]
    print(f"\n=== мир M1, протокол «{protocol}»: проверка, решено / медиана вопросов")
    print("  мир       n " + "".join(f"{label:>18s}" for label, _ in arms) + f"{'оракул':>18s}")
    for structure in STRUCTURES:
        for n in TEST_N:
            mine = [r for r in test if r["structure"] == structure and r["n"] == n]
            line = f"  {structure:8s} {n:>2d} "
            for _, key in arms:
                line += (f"{sum(r[key]['right'] for r in mine):>8d}/{len(mine):<2d}"
                         f"{int(np.median([r[key]['cost'] for r in mine])):>8d}")
            line += (f"{sum(r['oracle']['method'] is not None for r in mine):>8d}/{len(mine):<2d}"
                     f"{int(np.median([r['oracle']['cost'] for r in mine])):>8d}")
            print(line)
    print("  критерии 1 и 5:")
    for name, part in (("решаемые", [r for r in test if r["oracle"]["method"] is not None]),
                       ("безнадёжные", [r for r in test if r["oracle"]["method"] is None])):
        line = f"    {name:12s}"
        for label, key in arms + (("оракул", "oracle"),):
            solved = sum((r[key]["right"] if key != "oracle" else r["oracle"]["method"] is not None)
                         for r in part)
            line += f"  {label} {solved}/{len(part)} {sum(r[key]['cost'] for r in part):>9d}"
        print(line)
    train = [r for r in rows if r["phase"] == "train"]
    line = "    обучение, вопросы / оракул по десяткам:"
    for label, key in arms[:-1]:
        line += f"  {label} " + " ".join(
            f"{sum(r[key]['cost'] for r in b) / sum(r['oracle']['cost'] for r in b):.2f}"
            for b in ([r for r in train if s <= r["index"] < s + 10] for s in range(0, 40, 10)))
    print(line)
    for label, key, _ in PLANNERS:
        print(f"  критерии 2 и 3, {label}:")
        _criteria_2_3(train, key, "обучение")
        _criteria_2_3(test, key, "проверка")


def report_trap(rows, order, protocol):
    rows = [r for r in rows if r["world"] == "trap" and r["order"] == order
            and r["protocol"] == protocol]
    test = [r for r in rows if r["phase"] == "test"]
    keys = (("M1", "m1"),) + tuple((label, key) for label, key, _ in PLANNERS)
    print(f"\n=== ловушка, порядок {order}, протокол «{protocol}»: по ящикам, сумма по потокам"
          " (вопросов / сожгли бюджет / проб)")
    print("  ящик  n" + "".join(f"{label:>30s}" for label, _ in keys) + f"{'оракул':>12s}")
    for i in range(12):
        mine = [r for r in test if r["index"] == i]
        line = f"  {i + 1:>4d} {mine[0]['n']:>2d}"
        for _, key in keys:
            probes = len(_probes(mine, key)) if key != "m1" else 0
            line += (f"{sum(r[key]['cost'] for r in mine):>16d}"
                     f"{sum(r[key]['cost'] >= solvers.BUDGET for r in mine):>7d}{probes:>7d}")
        print(line + f"{sum(r['oracle']['cost'] for r in mine):>12d}")
    for label, key in keys:
        print(f"  {label:12s} решено {sum(r[key]['right'] for r in test)}/{len(test)}, "
              f"вопросов {sum(r[key]['cost'] for r in test)}; на n=8: "
              f"{Counter(r[key]['tried'] for r in test if r['n'] == 8).most_common(3)}")
    print(f"  оракул       вопросов {sum(r['oracle']['cost'] for r in test)}")
    for label, key, _ in PLANNERS:
        print(f"  критерии 2 и 3, {label}:")
        _criteria_2_3(test, key, "проверка")


def show_box(row, key, title):
    print(f"\n  {title}: n={row['n']}, поток {row['stream']}, "
          f"итог {row[key]['tried']} за {row[key]['cost']}")
    for d in row[key]["decisions"]:
        print(f"    план {d['plan']:6s} стоимость {d['plan_cost']:>9.0f} успех {d['plan_success']:.2f}"
              f"  провалились: {d['before']}")
        if d["action"] == "probe":
            print(f"      → проба {d['method']} на {d['size']} входах: ожидалось {d['spend']:.0f} "
                  f"вопросов, ценность {d['worth']:.0f}; вышло {d['actual']}, план после: {d['after']}"
                  + ("  [повтор]" if d["repeat"] else ""))
            for plan, p, cost in d["branches"]:
                print(f"          исход ведёт к плану {plan:6s} с шансом {p:.2f}, стоимость {cost:>9.0f}")
        elif d["action"] == "apply":
            print(f"      → применить {d['method']}: {d['actual']} вопросов")
        else:
            print("      → сдаться")


def main() -> None:
    streams = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    started = time.time()
    m1_jobs = [(seed, protocol) for protocol in PROTOCOLS for seed in range(streams)]
    trap_jobs = [(seed, order, protocol) for order in TRAP_ORDERS for protocol in PROTOCOLS
                 for seed in range(streams)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for part in pool.map(m1_stream, m1_jobs) for row in part]
        rows += [row for part in pool.map(trap_stream, trap_jobs) for row in part]
    print(f"{time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сырые данные: {out}")
    for protocol in PROTOCOLS:
        report_m1(rows, protocol)
    for order in TRAP_ORDERS:
        for protocol in PROTOCOLS:
            report_trap(rows, order, protocol)
    print("\n=== почему: решения Planner изм на нескольких ящиках (поток 0)")
    picks = [
        ("frozen", lambda r: r["structure"] == "blocks" and r["n"] == 8, "M1, blocks, опыт заморожен"),
        ("frozen", lambda r: r["structure"] == "global" and r["n"] == 6, "M1, global, опыт заморожен"),
        ("frozen", lambda r: r["structure"] == "linear" and r["n"] == 5, "M1, linear, опыт заморожен"),
    ]
    for protocol, test_fn, title in picks:
        row = next(r for r in rows if r["world"] == "M1" and r["protocol"] == protocol
                   and r["phase"] == "test" and r["stream"] == 0 and test_fn(r))
        show_box(row, "change", title)
    for order in TRAP_ORDERS:
        trap_rows = [r for r in rows if r["world"] == "trap" and r["order"] == order
                     and r["protocol"] == "learning" and r["phase"] == "test"
                     and r["stream"] == 0 and r["n"] == 8]
        for k, row in enumerate(trap_rows[:2]):
            show_box(row, "change", f"ловушка {order}, {k + 1}-й ящик n=8, с обучением")


if __name__ == "__main__":
    main()
