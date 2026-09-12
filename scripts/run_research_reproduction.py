"""Step 2a of docs/RESEARCH_CONTRACT.md: does `mana/research` do what the
lab loop (`cognition/inquiry.py` 1.14, boundary mode, rule `claims`) did?

For every device, price and task the two loops run on two copies of one
device. Counted: identical sequences of actions (button, configuration,
outcome), identical action totals and stop reasons, identical standings
and uncovered parts for every question.

    python scripts/run_research_reproduction.py A|C|D|E [все|половина] [first_seed] [seeds]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_research_loop import reproduce  # noqa: E402

LETTER = sys.argv[1]
HALF = len(sys.argv) > 2 and sys.argv[2] == "половина"
FIRST = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 50
OPTIONS = {"A": {}, "C": {"needs": 2}, "D": {"needs": 3}, "E": {"hidden_pair": True}}[LETTER]
COSTS = (4.0, 16.0, 64.0, 128.0)

runs = 0
differs = {"acts": 0, "actions": 0, "stopped": 0, "standing": 0}
first = []
for seed in range(FIRST, FIRST + SEEDS):
    for cost in COSTS:
        result = reproduce(seed, OPTIONS, cost, half=HALF)
        runs += 1
        for what, (lab, new) in result.items():
            if lab != new:
                differs[what] += 1
                if len(first) < 5:
                    first.append((seed, cost, what))
print(f"=== {LETTER}, задача «{'половина' if HALF else 'все'}», сиды {FIRST}-{FIRST + SEEDS - 1}, "
      f"цены {COSTS}: сравнений {runs} ===")
print("  расхождений: " + ", ".join(f"{k} {v}" for k, v in differs.items()))
if first:
    print(f"  первые: {first}")
