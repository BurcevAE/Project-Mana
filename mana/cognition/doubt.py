"""
mana.cognition.doubt — does MANA have grounds, in its own state, to doubt the
space of explanations it considered before accepting a leader?
(docs/РАСШИРЕНИЕ_ПРОСТРАНСТВА.md)

Belief inside the hypothesis set is belief among the explanations MANA
enumerated. OTHER, which was meant to stand for the rest, predicts one half
on every probe and loses half its weight with every observation, while an
unconsidered program consistent with the history predicts every observation
exactly and loses nothing. So a wrong leader can hold 0.99 inside the set.

Here belief in the leader is estimated over the whole language instead:

    P_language(leader) = prior mass of programs consistent with the history
                         that behave as the leader on the probe space
                         / prior mass of all programs consistent with it

The prior is the description measure itself (`discovery.prior`). The rule is
the one inquiry already has: accept at CONFIDENCE. The estimate is
sequential -- batches until a Wilson interval lies clear of CONFIDENCE; a
compute cap reached undecided counts as not enough, so doubt remains.

The same estimate says two different things: behaviours consistent with the
history that the set does not represent (the considered space is not
enough), and no consistent program at all (the language is not enough --
only recorded).

What it reads: the history, the probe space, the set's own predictions. Not
the world.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..discovery import description, prior
from ..discovery.language import Evaluator, show
from .explain import _predicts
from .inquiry import CONFIDENCE, OTHER, Hypothesis, HypothesisSet

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Declared once (docs/РАСШИРЕНИЕ_ПРОСТРАНСТВА.md, 10): the interval, the
#: batch, and the compute cap of one estimate.
Z = 1.96
BATCH = 2000
CAP = 200_000


@dataclass
class Estimate:
    belief: Optional[float]
    low: float
    high: float
    consistent: int
    agreeing: int
    drawn: int
    enough: bool
    decided: bool
    language_short: bool
    rivals: List[Hypothesis] = field(default_factory=list)


def wilson(agreeing: int, n: int, z: float = Z) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = agreeing / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def over_language(conditions: Sequence[str], noise: float = 0.0, seed: int = 0,
                  confidence: float = CONFIDENCE, batch: int = BATCH, cap: int = CAP
                  ) -> Callable[[HypothesisSet, Hypothesis, Sequence[Dict]], Estimate]:
    names = list(conditions)

    def columns(rows: Sequence[Dict]) -> Dict[str, np.ndarray]:
        return {n: np.array([int(bool(r["conditions"][n])) for r in rows], dtype=np.int64)
                for n in names}

    def estimate(hypotheses: HypothesisSet, lead: Hypothesis, space: Sequence[Dict]) -> Estimate:
        history = hypotheses.history
        seen = Evaluator(columns([params for params, _ in history])) if history else None
        works = np.array([bool(outcome) for _, outcome in history])
        probes = Evaluator(columns(space))
        leader = tuple(lead.predict(p).get(True, 0.0) > 0.5 for p in space)
        known = {tuple(h.predict(p).get(True, 0.0) > 0.5 for p in space)
                 for h in hypotheses.live() if h.name != OTHER}
        rng = random.Random(seed * 1_000_003 + len(history))
        consistent = agreeing = drawn = 0
        shortest: Dict[tuple, Tuple[float, str, tuple]] = {}
        low, high, decided = 0.0, 1.0, False
        while drawn < cap:
            for _ in range(batch):
                program = prior.sample(names, rng)
                drawn += 1
                if seen is not None and not np.array_equal(np.asarray(seen(program)) != 0, works):
                    continue
                consistent += 1
                behaviour = tuple(bool(v) for v in np.asarray(probes(program)) != 0)
                if behaviour == leader:
                    agreeing += 1
                elif behaviour not in known:
                    bits = description.program_bits(program, len(names))
                    held = shortest.get(behaviour)
                    if held is None or (bits, show(program)) < held[:2]:
                        shortest[behaviour] = (bits, show(program), program)
            low, high = wilson(agreeing, consistent)
            if consistent and (low >= confidence or high < confidence):
                decided = True
                break
        rivals = [Hypothesis(f"программа {text}", _predicts(program, names, noise))
                  for _, text, program in sorted(shortest.values())]
        belief = agreeing / consistent if consistent else None
        return Estimate(belief, low, high, consistent, agreeing, drawn,
                        enough=decided and low >= confidence, decided=decided,
                        language_short=consistent == 0, rivals=rivals)

    return estimate
