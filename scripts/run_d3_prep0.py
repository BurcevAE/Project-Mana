"""H3, D3-prep-0, check 5 (docs/ГЛУБИНА_D3.md, 9.2): candidates(s) left the
catalogue as it was.

The one-level runs of D2-prep-2 -- 8 plans at the root of the 60 instances
of its main arm, 400k, A' rebuilt from R2's experience and checked as there
-- are run again and compared with D2-prep-2's records: the answer right on
the world, the support at the root, the root's observation X, the ledger,
the number of nodes. `take` and `argmin`/`argmax` now read a list once;
this says whether A, C and the P2-class step still do what they did.

    python -u -X utf8 scripts/run_d3_prep0.py [workers] d2_prep2_results.jsonl
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d2_prep2 as D2  # noqa: E402

COMPARED = ("right", "at_root", "deeper", "observation", "spent", "nodes")


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    recorded = D2.load(sys.argv[2])
    started = time.time()
    cat = D2.catalogue()
    jobs = [(i, D2.BUDGET, p, None) for i in D2.MAIN for p in range(len(cat))]
    missing = [j for j in jobs if D2._key("tree", *j) not in recorded]
    print(f"записей D2-prep-2: {len(recorded)}; прогонов для сверки {len(jobs)}, "
          f"без записи {len(missing)}", flush=True)
    if missing:
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rules, same, steps = D2.rebuild_a_prime(pool)
        print(f"A' пересобрана, изменение {steps}; {time.time() - started:.0f}с", flush=True)
        if not same:
            print("изменение не совпало с R2 — сверка остановлена")
            return
        rows = list(pool.map(D2.tree_job, [j + (rules,) for j in jobs]))
    differ = []
    for job, row in zip(jobs, rows):
        old = recorded[D2._key("tree", *job)]
        bad = [f for f in COMPARED if old[f] != row[f]]
        if bad:
            differ.append((job, bad))
    print(f"всего {time.time() - started:.0f}с\n")
    print(f"  одноуровневые прогоны D2-prep-2 повторились во всех сравниваемых полях: "
          f"{len(jobs) - len(differ)} из {len(jobs)}")
    for name in dict.fromkeys(p.name for p in cat):
        mine = [d for d in differ if cat[d[0][2]].name == name]
        print(f"    {name}: расхождений {len(mine)}")
    for (inst, _, p, _), bad in differ:
        print(f"    {inst} {cat[p].name}: расходится {bad}")


if __name__ == "__main__":
    main()
