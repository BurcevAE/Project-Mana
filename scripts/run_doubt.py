"""Doubt in the space of models before deciding (docs/РОЖДЕНИЕ_ГИПОТЕЗ.md, 12).

The control is I from section 11, unchanged; D is I with doubt before a
leader is accepted. Both on the same 2560 law-seed pairs.

    gate 2   I in the new code reproduces every row of section 11's run
    report   wrong closures, correct, open, accuracy on unseen states,
             paired transitions, cost to close, doubt calls

    python -u -X utf8 scripts/run_doubt.py [workers] [old birth_results.jsonl] [new.jsonl]
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_birth as B  # noqa: E402


def _load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    old_path, new_path = sys.argv[2], Path(sys.argv[3])
    started = time.time()
    jobs = [(p, law, seed) for p in ("I", "D") for law in B.LAWS for seed in B.SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(B.job, jobs, chunksize=16))
    new_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
    print(f"прогон {time.time() - started:.0f}с, строк {len(rows)}")

    old = {(r["law"], r["seed"]): r for r in _load(old_path) if r["policy"] == "I"}
    I = {(r["law"], r["seed"]): r for r in rows if r["policy"] == "I"}
    D = {(r["law"], r["seed"]): r for r in rows if r["policy"] == "D"}
    differ = [k for k, r in old.items() if any(r[f] != I[k].get(f) for f in r)]
    print(f"\n=== ворота 2: I в новом коде = I раздела 11 поле в поле: "
          f"{len(old) - len(differ)} из {len(old)}")
    for k in differ[:10]:
        print(f"  расходится: {k}")

    def share(rs, verdict):
        return 100 * sum(1 for r in rs if r["verdict"] == verdict) / len(rs)

    print("\n=== I против D по разрезам")
    print("  разрез  n     I верно  I неверно  D верно  D неверно  D открыто  D не объясн.")
    for s in ("S1", "S2", "S3", "все"):
        keys = [k for k in I if s == "все" or I[k]["stratum"] == s]
        ir, dr = [I[k] for k in keys], [D[k] for k in keys]
        print(f"  {s:<6} {len(keys):>5}  {share(ir, 'correct'):>7.1f}%  {share(ir, 'wrong'):>8.1f}%"
              f"  {share(dr, 'correct'):>6.1f}%  {share(dr, 'wrong'):>8.1f}%"
              f"  {share(dr, 'open'):>8.1f}%  {share(dr, 'unexplained'):>10.1f}%")

    def unseen(rs):
        mine = [r for r in rs if r["hidden_right"] is not None and r["hidden"] > 0]
        total = sum(r["hidden"] for r in mine)
        return 100 * sum(r["hidden_right"] for r in mine) / max(1, total), len(mine), total

    for name, rs in (("I", list(I.values())), ("D", list(D.values()))):
        acc, n, total = unseen(rs)
        print(f"  точность на невиданных, {name}: {acc:.1f}% ({n} отвеченных, {total} состояний)")

    moves = collections.Counter((I[k]["verdict"], D[k]["verdict"]) for k in I)
    print(f"\n=== попарно I → D: {dict(moves.most_common())}")
    fixed = moves[("wrong", "correct")]
    broken = moves[("correct", "wrong")] + moves[("correct", "open")] + moves[("correct", "unexplained")]
    print(f"  I неверно → D верно: {fixed} ({100 * fixed / len(I):.1f}%); "
          f"I верно → D не верно: {broken} ({100 * broken / len(I):.1f}%)")

    med = lambda xs: statistics.median([x for x in xs if x is not None]) if any(  # noqa: E731
        x is not None for x in xs) else None
    print(f"\n=== цена до закрытия, медиана: I {med([r['to_settle'] for r in I.values()])}, "
          f"D {med([r['to_settle'] for r in D.values()])}")
    doubts = [r["doubts"] for r in D.values()]
    reopened = [r["doubt_reopened"] for r in D.values()]
    print(f"  вызовов сомнения у D: медиана {med(doubts)}, всего {sum(doubts)}; "
          f"открыли вопрос {sum(reopened)} раз; испытаний, где сомнение открывало, "
          f"{sum(1 for x in reopened if x)}")
    wrong_d = [r for r in D.values() if r["verdict"] == "wrong"]
    print(f"  неверных у D по разрезам: {dict(collections.Counter(r['stratum'] for r in wrong_d))}")
    print(f"  законов, где D неверна во всех 10 сидах: "
          f"{sum(1 for law in B.LAWS if all(D[(law, s)]['verdict'] == 'wrong' for s in B.SEEDS))}")


if __name__ == "__main__":
    main()
