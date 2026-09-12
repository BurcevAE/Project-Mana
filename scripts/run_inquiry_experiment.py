"""The inquiry experiment: one MODEL_CHECK, two ways of pricing it.

    E-geometric  a probe can show the model wrong in proportion to how far
                 it is from anything observed
    E-claims     in proportion to how many of the leading explanation's own
                 commitments it would test for the first time

    python scripts/run_inquiry_experiment.py [seeds] [budget] [variants]

`variants` is a string of letters, e.g. ACD; default ABB'CD as "ABPCD".
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mana.cognition import inquiry  # noqa: E402
from mana.world import device as dev  # noqa: E402

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 250
WHICH = sys.argv[3] if len(sys.argv) > 3 else "ABPCD"

VARIANTS = {
    "A": ("A. детерминированный", dict(noise=0.0), BUDGET),
    "B": ("B. шум 10%", dict(noise=0.1), BUDGET),
    "P": ("B'. шум 10%, бюджет вдвое", dict(noise=0.1), BUDGET * 2),
    "C": ("C. b0 = два переключателя сразу", dict(needs=2), BUDGET),
    "D": ("D. b0 = три переключателя сразу", dict(needs=3), BUDGET),
}

POLICIES = [
    ("enumeration", lambda d, b: dev.run_enumeration(d, b)),
    ("inquiry", lambda d, b: dev.run_inquiry(d, b)[0]),
    ("E-geometric", lambda d, b: dev.run_inquiry(d, b, model_check=True,
                                                 vulnerability=inquiry.GEOMETRIC)[0]),
    ("E-claims", lambda d, b: dev.run_inquiry(d, b, model_check=True,
                                              vulnerability=inquiry.CLAIMS)[0]),
    ("E-claimlevel", lambda d, b: dev.run_inquiry(d, b, model_check=True,
                                                  vulnerability=inquiry.CLAIMS,
                                                  claim_level=True)[0]),
]


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else float("nan")


def paired(a_rows, b_rows, budget):
    wins = losses = ties = 0
    for a, b in zip(a_rows, b_rows):
        x = a.to_correct if a.to_correct is not None else budget + 1
        y = b.to_correct if b.to_correct is not None else budget + 1
        wins += x < y
        losses += x > y
        ties += x == y
    return wins, losses, ties, dev.sign_test(wins, losses)


def run_variant(title, options, budget):
    rows = {name: [] for name, _ in POLICIES}
    for seed in range(SEEDS):
        for name, run in POLICIES:
            rows[name].append(run(dev.random_device(seed, **options), budget))

    print(f"\n=== {title}: {SEEDS} устройств × 5 кнопок, бюджет {budget} ===")
    print(f"{'политика':<12} {'верно':>6} {'ошибка':>7} {'не объясн.':>10} {'открыто':>8} "
          f"{'действий':>9} {'до верного':>11} {'проверок':>9} {'сама':>6}")
    for name, trials in rows.items():
        itself = sum(1 for t in trials if t.stopped == "NO_QUESTIONS")
        print(f"{name:<12} {sum(t.correct for t in trials):>6} {sum(t.wrong for t in trials):>7} "
              f"{sum(t.unexplained for t in trials):>10} {sum(t.open for t in trials):>8} "
              f"{mean(t.actions for t in trials):>9.1f} "
              f"{mean(t.to_correct for t in trials):>11.1f} "
              f"{mean(t.checks for t in trials):>9.1f} {itself:>3}/{len(trials):<2}")
    if options.get("needs", 1) > 1:
        for name in rows:
            counts = {}
            for t in rows[name]:
                counts[t.verdicts["b0"]] = counts.get(t.verdicts["b0"], 0) + 1
            print(f"  b0 у {name:<12}: {counts}")
    w, l, t, pv = paired(rows["E-claimlevel"], rows["E-claims"], budget)
    print(f"  E-claimlevel против E-claims до верного: быстрее {w}, медленнее {l}, "
          f"поровну {t}; p={pv:.2g}")


if __name__ == "__main__":
    for letter in WHICH:
        run_variant(*VARIANTS[letter])
