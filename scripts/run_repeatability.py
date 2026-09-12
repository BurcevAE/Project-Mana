"""Noise as a claim of the answer: repeatability assumed against checked.

On the synthetic device (A|C|D|E, deterministic) -- what checking costs and
whether honesty and standings hold. On `world/universe.py` (U) -- whether
the effect the world gives four times in five is still called verified.

    python scripts/run_repeatability.py A|C|D|E|U [seeds] [budget] [допущена|проверяется]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from mana.core import standing  # noqa: E402
from mana.research import contract, loop  # noqa: E402
from mana.research.adapters import device as dev_adapter  # noqa: E402
from mana.research.adapters import universe as uni_adapter  # noqa: E402
from mana.world import device as dev  # noqa: E402
from mana.world.universe import SmallWorld  # noqa: E402
from test_research_universe import EFFECTS, audit  # noqa: E402

LETTER = sys.argv[1]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else (10 if LETTER == "U" else 50)
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else (300 if LETTER == "U" else 250)
COSTS = (16.0, 64.0, 256.0) if LETTER == "U" else (16.0, 64.0, 128.0)
MODES = {"допущена": (True,), "проверяется": (False,)}.get(
    sys.argv[4] if len(sys.argv) > 4 else "", (True, False))
OPTIONS = {"A": {}, "C": {"needs": 2}, "D": {"needs": 3}, "E": {"hidden_pair": True}}
HEAT = uni_adapter.effect_subject("heat_rod", "rod.expanded")


def run_device(seed, cost, assume):
    device = dev.random_device(seed, **OPTIONS[LETTER])
    knowledge = dev_adapter.knowledge_for(device)
    loop.inquire(lambda: loop.unsettled(knowledge), dev_adapter.DeviceWorld(device), BUDGET,
                 contract.Stakes(error_cost=cost, assume_repeatable=assume),
                 challenge=dev_adapter.challenge_for(device))
    violations = 0
    for button, e in knowledge.items():
        if e.settled and e.settled[0] == loop.ANSWERED:
            _, _, lead = e.leading_class(e.space)
            wrong = dev_adapter.wrong_situations(device, button, lead)
            violations += not standing.audit(e.standing, wrong).honest
    return knowledge, violations, device.actions, None


def run_universe(seed, cost, assume):
    world = uni_adapter.UniverseWorld(SmallWorld(seed), effects=EFFECTS)
    knowledge = uni_adapter.knowledge_for(world)
    loop.inquire(lambda: loop.unsettled(knowledge), world, BUDGET,
                 contract.Stakes(error_cost=cost, assume_repeatable=assume),
                 challenge=uni_adapter.challenge_for(world))
    violations = sum(not result.honest for result, _ in audit(world, knowledge).values())
    heat = knowledge[HEAT].standing
    return knowledge, violations, world.actions_taken, heat.status if heat else "открыто"


run = run_universe if LETTER == "U" else run_device
print(f"=== {LETTER}: {SEEDS} запусков, бюджет {BUDGET} ===")
for cost in COSTS:
    for assume in MODES:
        statuses, heat = Counter(), Counter()
        violations = actions = 0
        for seed in range(SEEDS):
            knowledge, v, a, h = run(seed, cost, assume)
            violations += v
            actions += a
            for e in knowledge.values():
                statuses[e.standing.status if e.standing else "открыто"] += 1
            if h is not None:
                heat[h] += 1
        mode = "допущена" if assume else "проверяется"
        line = (f"цена {cost:>5g}, повторяемость {mode:<11}: действий {actions / SEEDS:>6.1f}; "
                f"с ошибкой внутри границы {violations:>3}; проверено/условно/не объясн./открыто "
                f"{statuses[standing.VERIFIED_FOR_TASK]}/{statuses[standing.CONDITIONAL]}/"
                f"{statuses[standing.UNEXPLAINED]}/{statuses['открыто']}")
        if heat:
            line += f"; эффект нагрева: {dict(heat)}"
        print(line)
