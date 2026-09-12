"""The honesty audit of core/standing, on the runs it was measured in.

For every answer the research loop (`cognition/inquiry.py`, boundary mode)
closes, core draws the boundary again from the loop's state and the oracle
says where the answer is wrong. Two counts per world and rule:

    violations   answers with an error inside their declared boundary
    mismatches   answers where core's boundary or standing differs from the
                 loop's own -- must be zero

    python scripts/run_standing_audit.py A|C|D|E [first_seed] [seeds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mana.cognition import inquiry  # noqa: E402
from mana.core import standing  # noqa: E402
from mana.world import device as dev  # noqa: E402

LETTER = sys.argv[1]
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
OPTIONS = {"A": {}, "C": {"needs": 2}, "D": {"needs": 3}, "E": {"hidden_pair": True}}[LETTER]
COSTS = (16, 32, 64, 128) if LETTER == "E" else (4, 8, 16, 32, 64, 128)


def audit_run(seed, rule, cost):
    device = dev.random_device(seed, **OPTIONS)
    knowledge = dev.knowledge_for(device)
    stakes = inquiry.Stakes(error_cost=cost)
    probes = dev.DeviceProbes(device)
    inquiry.inquire(lambda: inquiry.unsettled(knowledge), probes, 250,
                    challenge=dev.challenge_for(device), stakes=stakes, boundary=rule)
    answered = violations = mismatches = 0
    for button, hs in knowledge.items():
        if not (hs.settled and hs.settled[0] == inquiry.ANSWERED):
            continue
        answered += 1
        # The member of the answering class the loop closed on: evidence is
        # counted per member there, so a different tie-break is a different
        # boundary (found by this audit; see docs/RESEARCH_CONTRACT.md).
        _, _, answer = hs.leading_class(hs.space)
        space = [(hs._config_key(p), p["conditions"]) for p in hs.space]
        independent = [p["conditions"] for (p, o), r in zip(hs.history, hs.evidence)
                       if hs._vouches(r, p, o, answer)]
        task = {hs._config_key(p): stakes.weight(p) for p in hs.space}
        record = standing.standing(button, answer.name, space, hs._relevant(answer),
                                   hs.observed_coverage(), independent, rule, task)
        mismatches += (record.inside != frozenset(hs._inside(answer))
                       or record.status != hs.applicability["status"])
        rule_true = device.rules[button]
        wrong = {c for c in device.reachable()
                 if (answer.predict(probes.params(button, c)).get(True, 0.0) > 0.5)
                 != rule_true.holds(c)}
        violations += not standing.audit(record, wrong).honest
    return answered, violations, mismatches


print(f"=== {LETTER}: сиды {FIRST}-{FIRST + SEEDS - 1}, цены ошибки {COSTS} ===")
for rule in (standing.CLAIMS, standing.PATTERN):
    runs = answered = violations = mismatches = 0
    worst = 0
    for cost in COSTS:
        per_cost = 0
        for seed in range(FIRST, FIRST + SEEDS):
            a, v, m = audit_run(seed, rule, cost)
            runs += 1
            answered += a
            violations += v
            per_cost += v
            mismatches += m
        worst = max(worst, per_cost)
    print(f"{rule:<8} запусков {runs:>4}, закрытых ответов {answered:>5}, "
          f"с ошибкой внутри границы {violations:>4} (худшая цена: {worst}), "
          f"расхождений ядра с циклом {mismatches}")
