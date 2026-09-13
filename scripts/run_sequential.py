"""The sequential verdict against the fixed thirty, on duels with a known effect.

Each simulated duel game is drawn with the draw share of the real record
and, when decided, won by the changed player with probability `share`.
Both rules see the same games.

  фиксированное  -- the chess duel of 2.90: stop at 30 decided or 70 games;
                    Wilson interval at alpha/q; adopted only when the
                    interval excludes 0.5 on the winning side
  последовательное -- core.sequential: SPRT, min_effect 0.10, reversible
                    (alpha 5%, beta 20%), ceiling in decided outcomes

    python scripts/run_sequential.py [runs] [ceiling] [q]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.cognition.self_model import wilson_interval  # noqa: E402
from mana.core import sequential as seq  # noqa: E402
from mana.core.gates import ACCEPTED, NOT_EVALUATED, REJECTED  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
CEILING = int(sys.argv[2]) if len(sys.argv) > 2 else 400
QUESTIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

#: Eight duels of 2026-09-12 on the real record: 88 drawn of 336 games.
DRAWS = 88 / 336
SHARES = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
PLAN = seq.Plan(min_effect=0.10, consequence=seq.REVERSIBLE, ceiling=CEILING)


def game(rng, share):
    """None for a draw, True when the changed player won."""
    if rng.random() < DRAWS:
        return None
    return rng.random() < share


def fixed(rng, share):
    won = lost = games = 0
    while won + lost < 30 and games < 70:
        games += 1
        result = game(rng, share)
        if result is True:
            won += 1
        elif result is False:
            lost += 1
    trials = won + lost
    if trials < 30:
        return NOT_EVALUATED, trials, games
    z = NormalDist().inv_cdf(1 - (0.05 / QUESTIONS) / 2)
    low, high = wilson_interval(won, trials, z=z)
    return (ACCEPTED if low > 0.5 else REJECTED), trials, games


def sequential(rng, share):
    test = seq.SequentialTest(PLAN)
    games = 0
    while not test.decided:
        games += 1
        result = game(rng, share)
        if result is not None:
            test.observe(result)
    return test.status, test.trials, games


def run(rule, share, seed):
    rng = random.Random(seed)
    tally = {ACCEPTED: 0, REJECTED: 0, NOT_EVALUATED: 0}
    trials = games = 0
    for _ in range(RUNS):
        status, t, g = rule(rng, share)
        tally[status] += 1
        trials += t
        games += g
    return ({k: v / RUNS for k, v in tally.items()}, trials / RUNS, games / RUNS)


def main() -> None:
    print(f"дуэлей на точку: {RUNS}; ничьих {DRAWS:.0%}; фиксированное: 30 решённых, "
          f"q={QUESTIONS}; последовательное: δ={PLAN.min_effect:.0%}, "
          f"α={PLAN.alpha:.0%}, β={PLAN.beta:.0%}, потолок {CEILING} решённых")
    print()
    print(f"{'доля побед':>10} | {'правило':16} | {'принято':>7} {'отказ':>6} "
          f"{'не оцен.':>8} | {'решённых':>8} {'партий':>7} | прогноз Вальда")
    print("-" * 96)
    for index, share in enumerate(SHARES):
        ahead = seq.forecast(PLAN, share)
        for name, rule in (("фиксированное", fixed), ("последовательное", sequential)):
            rates, trials, games = run(rule, share, seed=1000 + index)
            tail = (f"принять {ahead['accept']:.0%}, ~{ahead['trials']:.0f} решённых"
                    if rule is sequential else "")
            print(f"{share:>10.0%} | {name:16} | {rates[ACCEPTED]:>7.1%} "
                  f"{rates[REJECTED]:>6.1%} {rates[NOT_EVALUATED]:>8.1%} | "
                  f"{trials:>8.1f} {games:>7.1f} | {tail}")
        print("-" * 96)


if __name__ == "__main__":
    main()
