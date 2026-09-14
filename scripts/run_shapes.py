"""Experiment M, step M1d: laws of cost told apart by experience.

The Planner of M1c, the hypothesis proper, twice: weighing one line per
method (as in M1c) and weighing laws of cost (choice.ShapeBelief). The main
measure, fixed before the run: calibration -- the plan changes a probe was
expected to bring, against the changes it brought.

Part A, the trap (trap.py):
    transfer   trained at n = 2, 3, 4, 6 (16 boxes, learning), then four
               boxes of n = 8 with what was learnt frozen: is the wall
               expected before the costly method runs?
    order      12 boxes in one of three orders of sizes -- 2 4 6 8, 8 2 6 4,
               3 7 4 9, three times each, learning throughout -- then one
               n = 8 box, frozen: do the beliefs arrive at the same law
               whatever the order?
Part B, the M1 world, frozen and learning, as in M1c.

    python scripts/run_shapes.py [streams] [workers] [out.json] [parts: A, B or AB]
"""
from __future__ import annotations

import json
import math
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
ORDERS = {"2-4-6-8": (2, 4, 6, 8), "8-2-6-4": (8, 2, 6, 4), "3-7-4-9": (3, 7, 4, 9)}
PLANNERS = (("линия", "line", choice.Planner), ("законы", "shape", choice.ShapePlanner))
LETTER = {"exhaust": "A", "calculate": "B", "decompose": "C", "experiment": "D",
          "explosive": "X", "steady": "S"}


def _word(methods):
    return "".join(LETTER[m] for m in methods) or "—"


def _decisions(planner):
    return [{"action": d.action, "method": d.method and LETTER[d.method], "size": d.size,
             "plan": _word(d.plan.names), "plan_cost": d.plan.cost,
             "change": d.predicted_change, "repeat": d.repeat,
             "after": None if d.after is None else _word(d.after), "actual": d.actual}
            for d in planner.last_decisions]


def _play(agent, box_of, n, levels, box_seed, learn):
    box = box_of()
    picked, model = agent.solve(box.answer, n, levels, box_seed, learn=learn)
    out = {"cost": picked.cost, "right": box.verdict(model) == 1.0, "tried": _word(picked.tried)}
    if isinstance(agent, choice.Planner):
        out["decisions"] = _decisions(agent)
    return out


def _oracle(box_of, n, levels, box_seed, registry=None):
    box = box_of()
    attempts = {m: solvers.attempt(m, box.answer, n, levels, box_seed, announce=False,
                                   methods=registry)
                for m in (registry or solvers.METHODS)}
    verdicts = {m: box.verdict(a.model) for m, a in attempts.items()}
    best = portfolio.oracle(attempts, verdicts)
    return {"method": best.method, "cost": best.cost}


def _view(planner, method, n):
    """What the planner expects of a method at size n on a new box."""
    belief = planner.belief(method, frozenset(), -1, (), planner.journal.drift())
    if belief is None:
        return None
    mu, sd = belief.predict(n)
    fits, _ = belief.outlook(n, planner.budget)
    view = {"median": math.exp(mu), "sd": sd, "fits": fits}
    if isinstance(belief, choice.ShapeBelief):
        view.update(belief.law())
    return view


def _trap_agents(seed):
    names = tuple(trap.METHODS)
    agents = {"m1": choice.Chooser(choice.Experience(methods=names), seed=seed,
                                   registry=trap.METHODS)}
    for _, key, kind in PLANNERS:
        agents[key] = kind(methods=names, registry=trap.METHODS, seed=seed, worth=choice.CHANGE)
    return agents


def _trap_box(n, box_seed):
    return lambda: BlackBox(n, LINEAR, box_seed, levels=trap.LEVELS)


def transfer_stream(seed):
    rng = np.random.default_rng([seed, 17])
    train = [2, 3, 4, 6] * 4
    rng.shuffle(train)
    agents = _trap_agents(seed)
    rows = []
    for phase, sizes in (("train", train), ("test", [8] * 4)):
        if phase == "test":
            rows.append({"part": "transfer", "stream": seed, "phase": "views",
                         "views": {key: {m: _view(agents[key], m, 8) for m in trap.METHODS}
                                   for _, key, _ in PLANNERS}})
        for i, n in enumerate(sizes):
            box_seed = seed * 1000 + (i if phase == "train" else 500 + i)
            box_of = _trap_box(n, box_seed)
            row = {"part": "transfer", "stream": seed, "phase": phase, "index": i, "n": n,
                   "oracle": _oracle(box_of, n, trap.LEVELS, box_seed, trap.METHODS)}
            for key, agent in agents.items():
                row[key] = _play(agent, box_of, n, trap.LEVELS, box_seed, phase == "train")
            rows.append(row)
    return rows


def order_stream(job):
    seed, order = job
    agents = _trap_agents(seed)
    rows = []
    sizes = list(ORDERS[order]) * 3
    for i, n in enumerate(sizes + [8]):
        last = i == len(sizes)
        if last:
            rows.append({"part": "order", "order": order, "stream": seed, "phase": "views",
                         "views": {key: {m: _view(agents[key], m, 8) for m in trap.METHODS}
                                   for _, key, _ in PLANNERS}})
        box_seed = seed * 1000 + 700 + i
        box_of = _trap_box(n, box_seed)
        row = {"part": "order", "order": order, "stream": seed,
               "phase": "final" if last else "learn", "index": i, "n": n,
               "oracle": _oracle(box_of, n, trap.LEVELS, box_seed, trap.METHODS)}
        for key, agent in agents.items():
            row[key] = _play(agent, box_of, n, trap.LEVELS, box_seed, not last)
        rows.append(row)
    return rows


def m1_stream(job):
    seed, protocol = job
    rng = np.random.default_rng([seed, 11])
    train = [(s, int(rng.choice(TRAIN_N))) for s in STRUCTURES for _ in range(10)]
    rng.shuffle(train)
    test = [(s, n) for s in STRUCTURES for n in TEST_N for _ in range(3)]
    agents = {key: kind(seed=seed, worth=choice.CHANGE) for _, key, kind in PLANNERS}
    rows = []
    for phase, boxes in (("train", train), ("test", test)):
        for i, (structure, n) in enumerate(boxes):
            box_seed = seed * 1000 + (i if phase == "train" else 500 + i)
            learn = phase == "train" or protocol == "learning"

            def box_of():
                return BlackBox(n, structure, box_seed)

            row = {"part": "M1", "stream": seed, "protocol": protocol, "phase": phase,
                   "index": i, "structure": structure, "n": n,
                   "oracle": _oracle(box_of, n, LEVELS, box_seed)}
            for key, agent in agents.items():
                row[key] = _play(agent, box_of, n, LEVELS, box_seed, learn)
            box = box_of()
            fixed, model, _ = portfolio.run_order(box.answer, n, LEVELS, box_seed)
            row["cascade"] = {"cost": fixed.cost, "right": box.verdict(model) == 1.0}
            rows.append(row)
    return rows


def calibration(rows, key, label):
    probes = [d for r in rows for d in r.get(key, {}).get("decisions", []) if d["action"] == "probe"]
    if not probes:
        print(f"    {label:10s} проб нет")
        return
    expected = np.array([d["change"] for d in probes])
    seen = np.array([d["after"] != d["plan"] for d in probes], dtype=float)
    print(f"    {label:10s} проб {len(probes):>5d}, вопросов {sum(d['actual'] for d in probes):>9d}; "
          f"ожидали смену плана {expected.mean():.0%}, увидели {seen.mean():.0%}, "
          f"разрыв {abs(expected.mean() - seen.mean()):.2f}, Брайер {np.mean((expected - seen) ** 2):.3f}, "
          f"повторов {sum(d['repeat'] for d in probes)}")
    for low, high in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        band = [i for i, e in enumerate(expected) if low <= e < high]
        if band:
            print(f"        {low:.1f}-{min(high, 1):.1f}: проб {len(band):>4d}, "
                  f"ожидали {expected[band].mean():.0%}, увидели {seen[band].mean():.0%}")


def _fmt_view(view):
    if view is None:
        return "—"
    text = f"медиана {view['median']:>11.0f} ±{view['sd']:.1f}, успеет {view['fits']:.2f}"
    if "bend" in view:
        text += f", излом {view['bend']:.2f} в {view['at']}"
    return text


def report_transfer(rows):
    rows = [r for r in rows if r["part"] == "transfer"]
    test = [r for r in rows if r["phase"] == "test"]
    views = [r for r in rows if r["phase"] == "views"]
    print("\n=== ловушка, перенос: обучение n = 2, 3, 4, 6; проверка n = 8, опыт заморожен")
    for label, key in (("M1", "m1"),) + tuple((label, key) for label, key, _ in PLANNERS):
        burnt = sum(r[key]["cost"] >= solvers.BUDGET for r in test)
        first = Counter(r[key]["decisions"][0]["action"] + " " + str(r[key]["decisions"][0]["method"])
                        + (str(r[key]["decisions"][0]["size"] or "")) for r in test
                        if key != "m1" and r[key]["decisions"])
        print(f"  {label:7s} решено {sum(r[key]['right'] for r in test)}/{len(test)}, "
              f"вопросов {sum(r[key]['cost'] for r in test):>9d}, сожгли бюджет {burnt}/{len(test)}; "
              f"порядки {Counter(r[key]['tried'] for r in test).most_common(3)}"
              + (f"; первое действие {first.most_common(3)}" if first else ""))
    print(f"  оракул  вопросов {sum(r['oracle']['cost'] for r in test)}")
    print("  что ожидалось от X при n = 8 перед проверкой (потоки 0-4):")
    for v in views[:5]:
        print(f"    поток {v['stream']}: линия {_fmt_view(v['views']['line']['explosive'])}")
        print(f"             законы {_fmt_view(v['views']['shape']['explosive'])}")
    for _, key, _ in PLANNERS:
        calibration(test, key, key)


def report_orders(rows):
    rows = [r for r in rows if r["part"] == "order"]
    print("\n=== ловушка, порядок размеров: 12 ящиков с обучением, затем n = 8 с замороженным опытом")
    for order in ORDERS:
        mine = [r for r in rows if r["order"] == order]
        learn = [r for r in mine if r["phase"] == "learn"]
        final = [r for r in mine if r["phase"] == "final"]
        views = [r for r in mine if r["phase"] == "views"]
        print(f"  порядок {order}:")
        for label, key, _ in PLANNERS:
            medians = [v["views"][key]["explosive"]["median"] for v in views
                       if v["views"][key]["explosive"] is not None]
            fits = [v["views"][key]["explosive"]["fits"] for v in views
                    if v["views"][key]["explosive"] is not None]
            bends = [v["views"][key]["explosive"].get("bend") for v in views
                     if v["views"][key]["explosive"] is not None]
            geo = math.exp(np.mean(np.log(medians))) if medians else float("nan")
            spread = (max(medians) / min(medians)) if medians else float("nan")
            bend = f", излом {np.mean(bends):.2f}" if bends and bends[0] is not None else ""
            print(f"    {label:7s} X@8: медиана {geo:>11.0f} (между потоками x{spread:.1f}), "
                  f"успеет {np.mean(fits):.2f}{bend}; обучение {sum(r[key]['cost'] for r in learn):>9d}"
                  f" вопросов, сожгли {sum(r[key]['cost'] >= solvers.BUDGET for r in learn)}; "
                  f"итоговый n=8: {Counter(r[key]['tried'] for r in final).most_common(2)}, "
                  f"сожгли {sum(r[key]['cost'] >= solvers.BUDGET for r in final)}")
        print(f"    M1      обучение {sum(r['m1']['cost'] for r in learn):>9d}, сожгли "
              f"{sum(r['m1']['cost'] >= solvers.BUDGET for r in learn)}; итоговый n=8 сожгли "
              f"{sum(r['m1']['cost'] >= solvers.BUDGET for r in final)}; оракул обучение "
              f"{sum(r['oracle']['cost'] for r in learn)}")
    for _, key, _ in PLANNERS:
        calibration([r for r in rows if r["phase"] != "views"], key, key)


def report_m1(rows, protocol):
    rows = [r for r in rows if r["part"] == "M1" and r["protocol"] == protocol]
    test = [r for r in rows if r["phase"] == "test"]
    print(f"\n=== мир M1, протокол «{protocol}»")
    for name, part in (("решаемые", [r for r in test if r["oracle"]["method"] is not None]),
                       ("безнадёжные", [r for r in test if r["oracle"]["method"] is None])):
        line = f"  {name:12s}"
        for label, key in tuple((label, key) for label, key, _ in PLANNERS) + (("каскад", "cascade"),):
            line += f"  {label} {sum(r[key]['right'] for r in part)}/{len(part)} {sum(r[key]['cost'] for r in part):>9d}"
        print(line + f"  оракул {sum(r['oracle']['cost'] for r in part)}")
    train = [r for r in rows if r["phase"] == "train"]
    for _, key, _ in PLANNERS:
        calibration(train, key, f"{key}/обуч")
        calibration(test, key, f"{key}/пров")


def main() -> None:
    streams = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    parts = sys.argv[4] if len(sys.argv) > 4 else "AB"
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if "A" in parts:
            rows += [row for part in pool.map(transfer_stream, range(streams)) for row in part]
            jobs = [(seed, order) for order in ORDERS for seed in range(streams)]
            rows += [row for part in pool.map(order_stream, jobs) for row in part]
        if "B" in parts:
            jobs = [(seed, protocol) for protocol in ("frozen", "learning") for seed in range(streams)]
            rows += [row for part in pool.map(m1_stream, jobs) for row in part]
    print(f"{time.time() - started:.0f}с")
    if out is not None:
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(f"сырые данные: {out}")
    if "A" in parts:
        report_transfer(rows)
        report_orders(rows)
    if "B" in parts:
        for protocol in ("frozen", "learning"):
            report_m1(rows, protocol)


if __name__ == "__main__":
    main()
