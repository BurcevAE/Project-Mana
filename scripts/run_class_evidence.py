"""Step 2b of docs/RESEARCH_CONTRACT.md: evidence counted for the class of
indistinguishable explanations, against counting it for the member that
led. Both modes on the same devices; per mode:

    tie-dependent  closed answers whose standing would differ had another
                   member of the answering class won the tie of weights
    violations     answers with an error inside their declared boundary,
                   by `core/standing.audit` against the oracle
    standings      verified for the task / conditional / unexplained / open
    actions        mean per device

    python scripts/run_class_evidence.py A|C|D|E [first_seed] [seeds]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mana.core import standing  # noqa: E402
from mana.research import contract, loop  # noqa: E402
from mana.research.adapters import device as adapter  # noqa: E402
from mana.world import device as dev  # noqa: E402

LETTER = sys.argv[1]
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
OPTIONS = {"A": {}, "C": {"needs": 2}, "D": {"needs": 3}, "E": {"hidden_pair": True}}[LETTER]
COSTS = (16.0, 32.0, 64.0, 128.0) if LETTER == "E" else (4.0, 16.0, 64.0, 128.0)


def run(seed, cost, evidence):
    device = dev.random_device(seed, **OPTIONS)
    knowledge = adapter.knowledge_for(device)
    loop.inquire(lambda: loop.unsettled(knowledge), adapter.DeviceWorld(device), 250,
                 contract.Stakes(error_cost=cost), challenge=adapter.challenge_for(device),
                 evidence=evidence)
    statuses = Counter()
    dependent = violations = 0
    for button, e in knowledge.items():
        if e.standing is None:
            statuses["открыто"] += 1
            continue
        statuses[e.standing.status] += 1
        if e.settled[0] != loop.ANSWERED:
            continue
        standings = {(a.inside, a.status, a.uncovered)
                     for a in (e.answer_for(e.get(n)) for n in e.settled[1])}
        dependent += len(standings) > 1
        _, _, lead = e.leading_class(e.space)
        wrong = adapter.wrong_situations(device, button, lead)
        violations += not standing.audit(e.standing, wrong).honest
    return statuses, dependent, violations, device.actions


print(f"=== {LETTER}: сиды {FIRST}-{FIRST + SEEDS - 1}, цены ошибки {COSTS} ===")
for evidence in (loop.MEMBER, loop.CLASS):
    statuses = Counter()
    dependent = violations = actions = runs = 0
    worst = 0
    for cost in COSTS:
        per_cost = 0
        for seed in range(FIRST, FIRST + SEEDS):
            s, d, v, a = run(seed, cost, evidence)
            statuses += s
            dependent += d
            violations += v
            per_cost += v
            actions += a
            runs += 1
        worst = max(worst, per_cost)
    closed = statuses[standing.VERIFIED_FOR_TASK] + statuses[standing.CONDITIONAL]
    print(f"{evidence:<7} закрытых ответов {closed:>5}: зависят от ничьей {dependent:>4}; "
          f"с ошибкой внутри границы {violations:>3} (худшая цена {worst}); "
          f"проверено/условно/не объясн./открыто "
          f"{statuses[standing.VERIFIED_FOR_TASK]}/{statuses[standing.CONDITIONAL]}/"
          f"{statuses[standing.UNEXPLAINED]}/{statuses['открыто']}; "
          f"действий {actions / runs:.1f}")
