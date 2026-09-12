"""Step 3 of docs/RESEARCH_CONTRACT.md: the research loop on a world whose
situations are not given (`world/universe.SmallWorld`).

Per run and question: the standing; the audit through `core/standing.py`
against the world's truth (errors inside the declared boundary); for the
effect that happens only sometimes, whether it was ever called verified.
Also: how many of the 160 reachable situations were found, actions taken,
replays that landed elsewhere.

    python scripts/run_universe_research.py [seeds] [budget]
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_research_universe import EFFECTS, audit, run  # noqa: E402
from mana.core import standing  # noqa: E402
from mana.research import loop  # noqa: E402
from mana.research.adapters import universe as adapter  # noqa: E402

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 300
COSTS = (16.0, 64.0, 256.0)
HEAT = adapter.effect_subject("heat_rod", "rod.expanded")

for cost in COSTS:
    per_question = defaultdict(Counter)
    violations = runs = 0
    chance_inside = 0
    known = actions = elsewhere = 0
    stopped = Counter()
    for seed in range(SEEDS):
        world, knowledge, report = run(seed, cost, BUDGET)
        runs += 1
        known += len(world.known)
        actions += world.actions_taken
        elsewhere += world.landed_elsewhere
        stopped[report.stopped] += 1
        for subject, e in knowledge.items():
            per_question[subject][e.standing.status if e.standing else "открыто"] += 1
        for subject, (result, uncertain) in audit(world, knowledge).items():
            violations += not result.honest
            chance_inside += bool(uncertain)
    print(f"\n=== цена ошибки {cost:g}: {runs} запусков, бюджет {BUDGET} ===")
    print(f"  ответов с ошибкой внутри границы: {violations}; ответов, чья граница "
          f"включила ситуацию, где мир отвечает лишь иногда: {chance_inside}")
    print(f"  найдено ситуаций в среднем {known / runs:.1f} из 160 достижимых; "
          f"действий {actions / runs:.1f}; повторов, попавших не туда, {elsewhere}; "
          f"остановка: {dict(stopped)}")
    for subject in sorted(per_question):
        mark = "  ← случайный эффект" if subject == HEAT else ""
        print(f"  {subject:<42} {dict(per_question[subject])}{mark}")
