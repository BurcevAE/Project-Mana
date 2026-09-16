"""H3, D3-prep-1: what one stage-3a run costs at each size, so the quota can
be fixed before the probe (docs/ГЛУБИНА_D3.md, 10.3 and 10.5).

Not a probe and not a verdict: expressions are drawn uniformly from the same
space (the declared seed below), put through stages 1 and 2 exactly as the
probe does, and those that survive are run once at 400k on one existence
instance -- timed. An equal quota of candidates per size is only equal work
if a run costs the same at every size, and it does not have to.

    python -u -X utf8 scripts/run_d3_prep1_quota.py [workers] [per size] [seed]
"""
from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mana.discovery import expressions as X  # noqa: E402
import run_d3_prep1 as D3  # noqa: E402

SIZES = (3, 4, 5, 6, 7, 8, 9, 10)
#: How many draws per size before giving up on finding survivors.
TRIES = 400


def survivors_job(args):
    """Draw expressions of one size; keep those that pass stages 1 and 2."""
    most, wanted, seed = args
    rng = random.Random(seed + most)
    kept, tried = [], 0
    started = time.time()
    while len(kept) < wanted and tried < TRIES:
        tried += 1
        e = X.sample(most, rng)
        for inst in D3.SCREEN:
            it = D3._screening(inst)
            why, x, values = D3._admissible(e, it)
            if x is None:
                continue
            if D3._useful(x, values, it):
                kept.append(e)
                break
    return most, kept, tried, time.time() - started


def timed_job(args):
    most, item, inst = args
    started = time.time()
    row = D3.full_job((0, item, inst, 0))
    return most, time.time() - started, row["right"], row["at_root"], row["fault"]


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    wanted = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20260916
    started = time.time()
    print(f"сид выборки {seed}; по {wanted} выживших на размер; "
          f"размеры {SIZES}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        drawn = list(pool.map(survivors_job, [(n, wanted, seed) for n in SIZES]))
        for most, kept, tried, spent in drawn:
            print(f"  размер {most}: выживших {len(kept)} из {tried} проб, {spent:.0f}с",
                  flush=True)
        jobs = [(most, list(e), D3.FIRST_FOUR[0]) for most, kept, _, _ in drawn for e in kept]
        rows = list(pool.map(timed_job, jobs))
    print(f"\n  размер  прогонов  секунд (мед)  прав  опора  сбоев   {time.time() - started:.0f}с")
    per_size = {}
    for most in SIZES:
        mine = [r for r in rows if r[0] == most]
        if not mine:
            print(f"  {most:>6d}         0            —")
            continue
        seconds = sorted(r[1] for r in mine)
        median = seconds[len(seconds) // 2]
        per_size[most] = median
        print(f"  {most:>6d}  {len(mine):>8d}  {median:>12.1f}  {sum(r[2] for r in mine):>4d}"
              f"  {sum(r[3] for r in mine):>5d}  {sum(r[4] for r in mine):>5d}")
    if per_size:
        whole = sum(per_size.values()) * 4          # four instances per candidate
        print(f"\n  один кандидат на всех размерах: {whole:.0f}с процессорного времени")
        for quota in (100, 200, 300, 500):
            hours = quota * whole / 6 / 3600
            print(f"  квота {quota:>4d} на размер: ступень 3а ≈ {hours:.1f} ч на 6 процессах")


if __name__ == "__main__":
    main()
