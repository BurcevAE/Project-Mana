"""N3, P2d: what the look does -- a diagnostic on P2c's dev seeds.

Nothing is claimed, nothing learnt, no hypothesis changed. The look
core.gates accepted in P2c (the first 256 states by score, one step ahead
through the rules of A') and the look without the move, on T4 seeds
20..49 at 400k, each run traced and its rounds logged. Every round's
choice is then recomputed from the log -- the same program, the same
rules -- and checked against what the run kept.

First pass:
1  which states the look keeps: the three it picks each round from the
   first 256 -- their rank by score, their shape, how much better their
   best successor is, and which rule makes it
2  kept against its twin: for each pick, the states of the same shortlist
   that differ from it in one leaf and were dropped -- their own score
   and their look's
3  how each answer was reached: made inside a look; through a pick and
   on through the successor the look saw in it; through a pick and on by
   another way; or through no pick at all
(4, what MANA's experience holds, is read from the code; see the package
docstring.)

Second pass, added after the first: for each solved run the first program
to grade exact, not only the one returned; whether it was made by the
search's rounds (it stands in some logged pool) or only by a look; the
state a look made it from; and how the rounds reached that state -- back
through the logged pools, each program to a state kept the round before
that makes it by the search's own rules. Where a recomputed choice departs
from the kept one. And P2's seeds, T4 0..9 at 800k.

    python -X utf8 scripts/run_look_anatomy.py [workers]
"""
from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p2 as P2  # noqa: E402
import run_reflect as R2  # noqa: E402

from mana.discovery import policy as P  # noqa: E402
from mana.discovery import reflect  # noqa: E402
from mana.discovery import search as discovery  # noqa: E402
from mana.discovery import selection  # noqa: E402
from mana.discovery.language import CONST, GET, children, show  # noqa: E402

SEEDS = range(20, 50)
BUDGET = 400000
#: P2's seeds and budget, to set this pass beside P2's counts.
P2_SEEDS, P2_BUDGET = range(10), 800000
M = 256
INSIDE, SEEN, OTHER, NONE, UNSOLVED = ("внутри взгляда", "через увиденного наследника",
                                       "из выбранной точки другим путём", "мимо выбранных",
                                       "не решено")


def leaves_of(p):
    if p[0] in (GET, CONST):
        return [p]
    return [leaf for kid in children(p) for leaf in leaves_of(kid)]


def shape(p) -> str:
    return re.sub(r"\b[a-z]\b|\b\d+\b", "·", show(p))


def twins(p, q) -> bool:
    if shape(p) != shape(q):
        return False
    return sum(a != b for a, b in zip(leaves_of(p), leaves_of(q))) == 1


def job(args):
    arm, rules, seed, budget = args
    world = R2.ALL["T4"]
    split = world.split(200, 300, seed)
    policy = P2.arm_policy((selection.look(M), rules), budget)
    rounds = []
    found = P.run(policy, split.train, split.train_outcomes, profile=True, trace=True,
                  log_rounds=rounds)

    def exact(p):
        return world.grade(lambda cols: discovery.predict(p, cols)) == 1.0

    solved = exact(found.program)
    ctx = P.selection_context(policy, split.train, split.train_outcomes)
    ahead = P._ahead(policy)
    leaves, conditions = discovery.vocabulary(split.train, split.train_outcomes)
    score = ctx.score

    def order(p):
        return (score(p),) + ctx.tie(p)

    def best_of(p):
        made = ctx.successors(p)
        if not made:
            return None, math.inf
        b = min(made, key=order)
        return b, score(b)

    agree, departed, picks, pairs, gains, roles, looked = 0, [], [], [], [], {}, []
    made_round = {}
    for r, (pool, beam) in enumerate(rounds):
        for p in pool:
            made_round.setdefault(p, r)
        if selection.choose(policy.selection.program, pool, ctx) == list(beam):
            agree += 1
        else:
            departed.append(r)
        short = sorted(pool, key=order)[:M]
        rank = {p: i + 1 for i, p in enumerate(short)}
        look = {p: best_of(p) for p in short}
        looked.append(short)
        gains += [score(p) - v for p, (_, v) in look.items() if v < math.inf]
        top = score(short[0])
        if beam:
            roles.setdefault(beam[0], []).append((r, "best", None))
        for p in beam[1:]:
            b, v = look[p] if p in look else best_of(p)
            rule = "—" if b is None else next(
                (ru.name for ru in ahead.rules if P.produces(ahead, ru, p, b, leaves, conditions)), "?")
            picks.append({"round": r, "rank": rank.get(p, 0), "above": score(p) - top,
                          "gain": score(p) - v, "rule": rule, "shape": shape(p),
                          "exact": bool(b is not None and ctx.right(b).all())})
            roles.setdefault(p, []).append((r, "look", b))
            for q in short:
                if q in beam or not twins(p, q):
                    continue
                bq, vq = look[q]
                pairs.append({"round": r, "own": score(q) - score(p), "look": vq - v,
                              "rank": (rank.get(p, 0), rank[q]), "pick": show(p), "twin": show(q),
                              "seen": show(b) if b else "", "twin seen": show(bq) if bq else "",
                              "solved": solved})

    def picked(p):
        return any(role == "look" for _, role, _ in roles.get(p, []))

    derivation = found.derivation
    after, maker_picked = None, None
    if not solved:
        kind = UNSOLVED
    elif len(derivation) == 1 and derivation[0] not in leaves:
        kind = INSIDE
        for r, short in enumerate(looked):
            makers = [p for p in short if found.program in ctx.successors(p)]
            if makers:
                maker_picked = any(picked(p) for p in makers)
                break
    else:
        on = [(i, s) for i, s in enumerate(derivation) if picked(s)]
        if not on:
            kind = NONE
        else:
            i, s = on[0]
            seen = [b for _, role, b in roles[s] if role == "look"][0]
            following = derivation[i + 1] if i + 1 < len(derivation) else None
            kind = SEEN if following == seen else OTHER
            after = len(derivation) - 1 - i
    path = " -> ".join(show(p) + ("[взгляд]" if picked(p) else "[лучшее]" if p in roles else "")
                       for p in derivation)

    # -- second pass ---------------------------------------------------------
    made_by = {}

    def chain_of(s):
        """How the rounds reached s: back through the logged pools, each
        program to a state kept the round before that makes it by the
        search's own rules."""
        out = [s]
        while made_round.get(s, 0) > 0:
            parent = None
            for k in rounds[made_round[s] - 1][1]:
                if k not in made_by:
                    made_by[k] = set(P.successors(P.CURRENT, k, leaves, conditions))
                if s in made_by[k]:
                    parent = k
                    break
            if parent is None:
                out.append(None)
                break
            out.append(parent)
            s = parent
        return list(reversed(out))

    first = next((p for _, p, _ in found.anytime if exact(p)), None)
    second = None
    if first is not None:
        by_look = first not in made_round and first not in leaves
        maker = None
        if by_look:
            for r, short in enumerate(looked):
                makers = [p for p in short if first in ctx.successors(p)]
                if makers:
                    maker = next((p for p in makers if picked(p)), makers[0])
                    break
            chain = chain_of(maker) if maker is not None else [None]
        else:
            chain = chain_of(first)[:-1]
        through = [p for p in chain if p is not None and picked(p)]
        if by_look and maker is not None and picked(maker):
            through = [p for p in through if p != maker]
        second = {"by look": by_look, "maker": maker is not None,
                  "maker picked": bool(maker is not None and picked(maker)),
                  "broken": None in chain, "through picks": len(through),
                  "chain": " -> ".join("?" if p is None else show(p)
                                       + ("[взгляд]" if picked(p) else "[лучшее]" if p in roles else "")
                                       for p in chain)
                  + (f" => {show(first)}" if by_look else f" -> {show(first)}"),
                  "returned by look": found.program not in made_round and found.program not in leaves}
    cut = found.evaluations >= budget
    return (arm, seed, solved, agree, len(rounds), picks, pairs, gains, kind, after,
            maker_picked, path, departed, cut, second, budget)


def q(values, share):
    values = sorted(values)
    return values[min(len(values) - 1, int(share * len(values)))] if values else float("nan")


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(R2.experience_job, [("A", n, s) for n in R2.TRAIN for s in range(10)]))
        change = reflect.improve(P.CURRENT, [r[5] for r in rows if r[4]])
        steps = [what.split(":")[0] for what, _ in change.steps]
        print(f"стадия 1: изменение {steps}; {time.time() - started:.0f}с")
        if steps != P2.R2_CHANGE:
            print("изменение не совпало с R2 — прогон остановлен")
            return
        narrow = change.after.rules
        arms = {"взгляд A'": narrow,
                "без хода": tuple(r for r in narrow if not r.name.startswith("grown"))}
        jobs = [(a, rules, s, BUDGET) for a, rules in arms.items() for s in SEEDS]
        jobs += [(a, rules, s, P2_BUDGET) for a, rules in arms.items() for s in P2_SEEDS]
        runs = list(pool.map(job, jobs))
    print(f"всего {time.time() - started:.0f}с")
    for arm in arms:
        mine = [r for r in runs if r[0] == arm and r[15] == BUDGET]
        picks = [p for r in mine for p in r[5]]
        later_picks = [p for p in picks if p["round"] > 0]
        pairs = [p for r in mine for p in r[6]]
        gains = [g for r in mine for g in r[7]]
        print(f"\n===== {arm}: решено {sum(r[2] for r in mine)}/{len(mine)}; выбор, пересчитанный "
              f"по логу, совпал в {sum(r[3] for r in mine)} раундах из {sum(r[4] for r in mine)}")
        at_end = [(r[1], d) for r in mine for d in r[12] if r[13] and d == r[4] - 1]
        elsewhere = [(r[1], d, r[4] - 1, r[13]) for r in mine for d in r[12]
                     if not (r[13] and d == r[4] - 1)]
        print(f"     расхождения: в последнем раунде, срезанном бюджетом, {len(at_end)}; "
              f"иначе {len(elsewhere)} (сид, раунд, последний, срезан): {elsewhere}")
        print("  1. кого берёт взгляд")
        print(f"     выбранных {len(picks)}; место по оценке: мин {q([p['rank'] for p in picks], 0)}, "
              f"25% {q([p['rank'] for p in picks], .25)}, мед {q([p['rank'] for p in picks], .5)}, "
              f"75% {q([p['rank'] for p in picks], .75)}, макс {q([p['rank'] for p in picks], 1)}; "
              f"вне первых 32: {sum(p['rank'] > 32 for p in picks)}")
        print(f"     после раунда 0 ({len(later_picks)}): место мед "
              f"{q([p['rank'] for p in later_picks], .5)}, 75% {q([p['rank'] for p in later_picks], .75)}; "
              f"вне первых 32: {sum(p['rank'] > 32 for p in later_picks)}")
        print(f"     хуже лучшего состояния раунда, бит (мед): {q([p['above'] for p in picks], .5):.1f}")
        print(f"     выигрыш взгляда, бит (своя оценка − оценка лучшего наследника): выбранные мед "
              f"{q([p['gain'] for p in picks], .5):.1f}; весь список 256 мед {q(gains, .5):.1f}, "
              f"90% {q(gains, .9):.1f}")
        print(f"     лучший наследник точен на обучении: {sum(p['exact'] for p in picks)} из {len(picks)}")
        print(f"     каким правилом сделан лучший наследник: "
              f"{Counter(p['rule'] for p in picks).most_common()}")
        print(f"     формы: {Counter(p['shape'] for p in picks).most_common(6)}")
        print("  2. выбранный против отброшенного близнеца (тот же вид, один лист другой)")
        if pairs:
            own = [p["own"] for p in pairs]
            look = [p["look"] for p in pairs]
            print(f"     пар {len(pairs)}; своя оценка близнеца − выбранного, бит: мед {q(own, .5):.1f}, "
                  f"25% {q(own, .25):.1f}, 75% {q(own, .75):.1f}; близнец лучше по своей оценке: "
                  f"{sum(o < 0 for o in own)}")
            print(f"     оценка взгляда близнеца − выбранного, бит: мед {q(look, .5):.1f}, "
                  f"25% {q(look, .25):.1f}, 75% {q(look, .75):.1f}")
            print(f"     |своя разница| < 10 бит и разница взгляда > 50 бит: "
                  f"{sum(abs(o) < 10 and l > 50 for o, l in zip(own, look))} из {len(pairs)}")
            shown = [p for p in pairs if p["solved"] and p["round"] == 1][:4]
            for p in shown:
                print(f"       выбран  {p['pick']} (место {p['rank'][0]}) -> видит {p['seen']}")
                print(f"       близнец {p['twin']} (место {p['rank'][1]}, своя {p['own']:+.1f}, "
                      f"взгляд {p['look']:+.1f}) -> видит {p['twin seen']}")
        print("  3. как получен возвращённый ответ")
        kinds = Counter(r[8] for r in mine)
        print(f"     {kinds.most_common()}")
        inside = [r for r in mine if r[8] == INSIDE]
        if inside:
            print(f"     внутри взгляда: сделан взглядом на выбранное состояние "
                  f"{sum(1 for r in inside if r[10])}, на невыбранное "
                  f"{sum(1 for r in inside if r[10] is False)}")
        later = [r[9] for r in mine if r[9] is not None]
        if later:
            print(f"     шагов от выбранного состояния до ответа: мед {q(later, .5)}, макс {q(later, 1)}")
        for kind in (SEEN, OTHER, NONE):
            for r in [r for r in mine if r[8] == kind][:2]:
                print(f"       {kind}, сид {r[1]}: {r[11]}")
        print("  3'. первый точный ответ прогона (второй проход)")
        firsts = [r[14] for r in mine if r[14] is not None]
        look_made = [s for s in firsts if s["by look"]]
        print(f"     первых точных {len(firsts)}: сделан только взглядом {len(look_made)}, "
              f"раундами {len(firsts) - len(look_made)}")
        print(f"     сделан взглядом: точка взгляда выбрана {sum(s['maker picked'] for s in look_made)}, "
              f"не выбрана {sum(s['maker'] and not s['maker picked'] for s in look_made)}, "
              f"точка не найдена {sum(not s['maker'] for s in look_made)}")
        print(f"     путь раундов к точке (к ответу, если его сделали раунды) проходит через ранее "
              f"выбранные взглядом: {sum(s['through picks'] > 0 for s in firsts)} из {len(firsts)}; "
              f"путь не восстановлен: {sum(s['broken'] for s in firsts)}")
        for s in [s for s in look_made if s["through picks"] > 0][:2] + \
                 [s for s in look_made if s["through picks"] == 0][:2]:
            print(f"       {s['chain']}")
    print("\n===== сверка с P2: T4 сиды 0..9, 800k")
    for arm in arms:
        mine = [r for r in runs if r[0] == arm and r[15] == P2_BUDGET]
        firsts = [r[14] for r in mine if r[14] is not None]
        print(f"  {arm}: решено {sum(r[2] for r in mine)}; первый точный сделан взглядом "
              f"{sum(s['by look'] for s in firsts)} из {len(firsts)}; возвращённый при 800k сделан "
              f"взглядом {sum(s['returned by look'] for s in firsts)} из {len(firsts)}")


if __name__ == "__main__":
    main()
