"""Both duel rules on real self-play, from the same record.

One arm per process, each in its own data directory seeded with a copy of
a real game record. Only the bench's experiment loop runs -- no network,
no ladder: pick a lever, play duel slices, decide, adopt, re-test. Same
record, same candidates, same duel seeds; the arms differ only in when an
experiment stops.

    python scripts/run_sequential_duel.py RECORD.jsonl OUT_DIR fixed|sequential [max_games] [LEDGER.jsonl]

LEDGER seeds the findings ledger, so questions already answered there are
not asked again -- the way to measure a later stage (weighted levers)
without replaying the earlier ones.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD, OUT, ARM = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
MAX_GAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 700
LEDGER = Path(sys.argv[5]) if len(sys.argv) > 5 else None
if ARM not in ("fixed", "sequential"):
    raise SystemExit("arm: fixed | sequential")

DATA = OUT / ARM
shutil.rmtree(DATA, ignore_errors=True)
(DATA / "lichess").mkdir(parents=True)
shutil.copy(RECORD, DATA / "lichess" / "games.jsonl")
if LEDGER is not None:
    (DATA / "findings").mkdir(parents=True)
    shutil.copy(LEDGER, DATA / "findings" / "findings.jsonl")
os.environ["MANA_DATA_DIR"] = str(DATA)       # before anything reads a path
sys.path.insert(0, str(ROOT))

from mana import events  # noqa: E402
from mana.cognition import chess_action, chess_bench, chess_version  # noqa: E402


class _NoBot:
    """The ladder is not played here; the bench only needs something to stop."""
    finished: list = []
    seats: dict = {}
    judge_depth = 0

    def stop(self) -> None:
        pass


played = {"games": 0}
_real_duel = chess_action.duel


def _counted(*args, **kwargs):
    out = _real_duel(*args, **kwargs)
    played["games"] += out.games
    return out


chess_action.duel = _counted

log = open(DATA / "events.log", "w", encoding="utf-8")
events.subscribe(lambda e: e.text and log.write(
    f"{time.strftime('%H:%M:%S')} {e.text}\n") and log.flush())

bench = chess_bench.Bench(client=object(), bot=_NoBot(),
                          ladder=chess_bench.Ladder(), gap=0.0,
                          sequential=(ARM == "sequential"))
bench.conclude()
start = time.time()
steps = 0
while played["games"] < MAX_GAMES:
    said = bench._experiment()
    steps += 1
    if not said:
        break
    if steps % 10 == 0:
        print(f"{time.time() - start:6.0f}с  партий {played['games']:4d}  {said[:110]}",
              flush=True)

causal = [f for f in bench._book().latest() if f.question == chess_action.QUESTION]
rows = []
for f in causal:
    m = f.measurement
    rows.append({"approach": f.approach.get("property"),
                 "then": f.approach.get("then"), "verdict": f.verdict,
                 "games": m.get("games"), "decided": m.get("trials"),
                 "drawn": m.get("drawn"), "won": m.get("changed_won"),
                 "lost": m.get("unchanged_won"),
                 "sequential": m.get("sequential"),
                 "control": f.conditions.get("control"), "note": f.note})
summary = {"arm": ARM, "games": played["games"], "seconds": round(time.time() - start),
           "experiments": rows,
           "versions": [a.as_dict() for a in chess_version.history()]}
(DATA / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
print(f"\n{ARM}: партий {played['games']}, {summary['seconds']}с, опытов {len(rows)}")
for row in rows:
    print(f"  {row['verdict']:13s} партий {row['games']:4d} решённых {row['decided']:4d} "
          f"({row['won']}/{row['lost']}, ничьих {row['drawn']})  {row['approach']} {row['then'] or ''}")
for a in chess_version.history():
    print(f"  версия {a.version}: {a.name()} [{a.state}] {a.note[:80]}")
