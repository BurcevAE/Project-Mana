"""N3, probe P2: the look of P1b through a narrower neighbourhood, one MANA grew.

P1b found a look one step ahead unaffordable: it costs a state's whole
neighbourhood, 5 800 of about 6 000 programs wrappings in a condition. One
change here: the rules the look makes successors with. The search itself
keeps its own rules; every successor the look sees is still evaluated in
the search's budget and can be the answer.

Stage 1  the rules are rebuilt, not written by hand: the dear runs of R2b
         (T1..T3, W0, W3, seeds 0..9) and reflect.improve -- on T1..T3 alone
         it gives R2's policy A' (the move added, "add a condition" dropped:
         narrow), on all five R2b's (the move added, "add a condition" kept:
         wide). The run stops if either change differs from what R2 and R2b
         recorded. T4 is in no experience.
Stage 2  arms, seven questions x seeds 0..9, 800k:
             R1                         the search as it stands
             look 32 / 256 through A'   the change
             look 256 through A' without its grown rules
                                        a control written by hand: narrow,
                                        without the move
             look 256 through R2b'      a control: the move, the width kept
         Part 1, of what kind: reversals on T4's first rounds, as in P1.
         Part 2: recall by budget, the share of the budget spent looking,
         answers made only by a look, and on T4 where step 6's kept state
         stood in round 1.

    python scripts/run_p2.py [workers]
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

from mana.discovery import policy as P  # noqa: E402
from mana.discovery import reflect  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery import selection  # noqa: E402
from mana.discovery.language import LESS, cmp, const, get, if_  # noqa: E402

TASKS = ("W0", "W3", "T1", "T2", "T3", "T4", "W4")
BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000, 800000)
STONE = if_(cmp(LESS, get("x"), const(3)), get("z"), get("y"))
#: What R2 and R2b recorded: the run stops if a rebuilt change differs.
R2_CHANGE = ["добавлено правило grown m13", "добавлено правило grown m14",
             "убрано правило «add a condition»", "убрано правило «replace by a leaf»",
             "убрано правило «swap add»"]
R2B_CHANGE = ["добавлено правило grown m13", "добавлено правило grown m14",
              "убрано правило «replace by a leaf»", "убрано правило «swap add»"]


def label(b):
    return f"{b // 1000}k"


def arm_policy(arm, budget):
    program, ahead = arm
    return P.with_budget(P.with_ahead(P.with_selection(P.CURRENT, program), ahead), budget)


def kind_job(job):
    name, arm, seed = job
    split = R2.ALL["T4"].split(200, 300, seed)
    policy = arm_policy(arm, 60000)
    rounds = []
    P.run(policy, split.train, split.train_outcomes, log_rounds=rounds)
    ctx = P.selection_context(policy, split.train, split.train_outcomes)
    tested = [selection.reversals(arm[0], pool, ctx) for pool, _ in rounds[1:3]]
    return name, seed, sum(tested), len(tested)


def run_job(job):
    name, arm, task, seed = job
    world = R2.ALL[task]
    split = world.split(200, 300, seed)
    policy = arm_policy(arm, max(BUDGETS))
    rounds = [] if task == "T4" else None
    found = P.run(policy, split.train, split.train_outcomes, profile=True, trace=True,
                  log_rounds=rounds)
    solved = []
    for b in BUDGETS:
        program, _, _ = discovery.at_budget(found, b)
        solved.append(world.grade(lambda cols, p=program: discovery.predict(p, cols)) == 1.0)
    leaves, _ = discovery.vocabulary(split.train, split.train_outcomes)
    # No parent from the search's own rounds: the answer was made only by a look.
    by_look = len(found.derivation) == 1 and found.program not in leaves
    stone = None
    if rounds and len(rounds) > 1:
        pool, kept = rounds[1]
        ctx = P.selection_context(policy, split.train, split.train_outcomes)
        if STONE in pool:
            mine = ctx.score(STONE)
            stone = (STONE in kept, 1 + sum(1 for p in pool if ctx.score(p) < mine))
    return (name, task, seed, solved, found.looked, found.evaluations, found.rounds, by_look,
            stone)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(R2.experience_job, [("A", n, s) for n in R2.VARIANTS["R2b"]["experience"]
                                                 for s in range(10)]))
        narrow = reflect.improve(P.CURRENT, [r[5] for r in rows if r[4] and r[1] in R2.TRAIN])
        wide = reflect.improve(P.CURRENT, [r[5] for r in rows if r[4]])
        for title, change, recorded in (("A'", narrow, R2_CHANGE), ("R2b'", wide, R2B_CHANGE)):
            steps = [what.split(":")[0] for what, _ in change.steps]
            print(f"стадия 1, {title}: {steps}")
            if steps != recorded:
                print("изменение не совпало с записанным — прогон остановлен")
                return
        print(f"стадия 1: {time.time() - started:.0f}с")
        bare = tuple(r for r in narrow.after.rules if not r.name.startswith("grown"))
        arms = {"R1": (selection.R1, None),
                "взгляд 32, A'": (selection.look(32), narrow.after.rules),
                "взгляд 256, A'": (selection.look(256), narrow.after.rules),
                "256, A' без хода": (selection.look(256), bare),
                "взгляд 256, R2b'": (selection.look(256), wide.after.rules)}
        kinds = list(pool.map(kind_job, [(n, a, s) for n, a in arms.items() if n != "R1"
                                         for s in range(3)]))
        runs = list(pool.map(run_job, [(n, a, t, s) for n, a in arms.items() for t in TASKS
                                       for s in range(10)]))
    print(f"всего {time.time() - started:.0f}с")
    print("\n=== часть 1: какого рода отбор (T4, раунды 1-2, убираем по одному удержанному)")
    for name in arms:
        mine = [r for r in kinds if r[0] == name]
        if mine:
            print(f"  {name:17s} раундов с разворотом {sum(r[2] for r in mine)}/{sum(r[3] for r in mine)}")
    print("\n=== часть 2: решено при бюджете")
    print("  бюджет                " + "".join(f"{label(b):>7s}" for b in BUDGETS)
          + "   на взгляд   раундов   ответ сделан взглядом")
    for title, tasks in (("T4 (из 10)", ("T4",)),
                         ("остальные шесть (из 60)", ("W0", "W3", "T1", "T2", "T3", "W4"))):
        print(f"  {title}")
        for name in arms:
            mine = [r for r in runs if r[0] == name and r[1] in tasks]
            line = "".join(f"{sum(r[3][i] for r in mine):>7d}" for i in range(len(BUDGETS)))
            share = sum(r[4] for r in mine) / max(sum(r[5] for r in mine), 1)
            rounds = sorted(r[6] for r in mine)[len(mine) // 2]
            made = sum(1 for r in mine if r[3][-1] and r[7])
            print(f"    {name:18s}{line}   {share:>9.0%}   {rounds:>7d}   {made:>3d} из {sum(r[3][-1] for r in mine)}")
    print("\n=== состояние из контроля шага 6 в раунде 1 T4: удержано? место по оценке")
    for name in arms:
        cells = []
        for r in sorted((r for r in runs if r[0] == name and r[1] == "T4"), key=lambda r: r[2]):
            cells.append("—" if r[8] is None else f"{'да' if r[8][0] else 'нет'} {r[8][1]}")
        print(f"  {name:17s} " + ", ".join(cells))


if __name__ == "__main__":
    main()
