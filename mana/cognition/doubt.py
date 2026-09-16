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

    P_language(leader) = posterior mass of programs that behave as the
                         leader on the probe space,
    posterior(p)      ∝  2^-bits(p) · [p consistent with the history]

The rule is the one inquiry already has: accept at CONFIDENCE.

How the posterior is sampled (section 12). Drawing from the prior and
rejecting inconsistent programs fails exactly when the model is well
confirmed: on seven observations a handful of 200 000 draws were consistent.
So the programs are drawn from the posterior by Metropolis-Hastings. The
edits of the discovery search are not reversible -- nothing unwraps
add(node, leaf) -- so the step regenerates a subtree from the prior code
itself: pick a node, redraw its subtree within the size left by MAX_SIZE;
inconsistent -- rejected; else accepted with probability old size / new
size. For this target that is the exact Metropolis-Hastings rule.

Chains start from each behaviour among the exact programs discovery finds.
The Wilson interval is taken over the effective sample size, read from the
chains' own autocorrelation. Batches until the interval clears CONFIDENCE;
a cap reached undecided is not enough, so doubt remains.

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
from ..discovery.language import Evaluator, nodes, replace, show, size
from ..discovery.search import MAX_SIZE
from .explain import _predicts, exact_programs
from .inquiry import CONFIDENCE, OTHER, Hypothesis, HypothesisSet

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

#: Declared once (docs/РАСШИРЕНИЕ_ПРОСТРАНСТВА.md, 10 and 12): the interval,
#: the batch, and the cap of one estimate, in steps of the chains.
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
    ess: float = 0.0


def wilson_rate(p: float, n: float, z: float = Z) -> Tuple[float, float]:
    """Wilson interval for a share p observed over n (possibly effective)
    samples."""
    if n <= 0:
        return 0.0, 1.0
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4 * n * n))) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson(agreeing: int, n: int, z: float = Z) -> Tuple[float, float]:
    return wilson_rate(agreeing / n, n, z) if n else (0.0, 1.0)


def effective_size(series: Sequence[float]) -> float:
    """Samples a chain is worth: N over one plus twice the autocorrelations,
    summed up to the first that is not positive. A series that never
    changes has no correlation to discount and counts in full."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 2 or x.var() == 0.0:
        return float(n)
    x = x - x.mean()
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum))[:n].real
    acf /= acf[0]
    total = 0.0
    for k in range(1, n):
        if acf[k] <= 0.0:
            break
        total += acf[k]
    return float(max(1.0, min(n, n / (1.0 + 2.0 * total))))


def by_rejection(conditions: Sequence[str], history, leader: tuple, space: Sequence[Dict],
                 draws: int, seed: int = 0, most: int = MAX_SIZE) -> Tuple[int, int]:
    """(agreeing, consistent) by drawing from the prior and rejecting --
    unbiased, and the reference the chains are checked against."""
    names = list(conditions)
    rng = random.Random(seed)
    seen = Evaluator(_columns(names, [p for p, _ in history])) if history else None
    works = np.array([bool(o) for _, o in history])
    probes = Evaluator(_columns(names, space))
    agreeing = consistent = 0
    for _ in range(draws):
        program = prior.sample(names, rng, most)
        if seen is not None and not np.array_equal(np.asarray(seen(program)) != 0, works):
            continue
        consistent += 1
        agreeing += tuple(bool(v) for v in np.asarray(probes(program)) != 0) == leader
    return agreeing, consistent


def _columns(names: Sequence[str], rows: Sequence[Dict]) -> Dict[str, np.ndarray]:
    return {n: np.array([int(bool(r["conditions"][n])) for r in rows], dtype=np.int64)
            for n in names}


def over_language(conditions: Sequence[str], noise: float = 0.0, seed: int = 0,
                  confidence: float = CONFIDENCE, batch: int = BATCH, cap: int = CAP,
                  most: int = MAX_SIZE
                  ) -> Callable[[HypothesisSet, Hypothesis, Sequence[Dict]], Estimate]:
    names = list(conditions)

    def estimate(hypotheses: HypothesisSet, lead: Hypothesis, space: Sequence[Dict]) -> Estimate:
        history = hypotheses.history
        seen = Evaluator(_columns(names, [p for p, _ in history])) if history else None
        works = np.array([bool(o) for _, o in history])
        probes = Evaluator(_columns(names, space))
        rng = random.Random(seed * 1_000_003 + len(history))

        def consistent(program) -> bool:
            return seen is None or np.array_equal(np.asarray(seen(program)) != 0, works)

        def behaviour(program) -> tuple:
            return tuple(bool(v) for v in np.asarray(probes(program)) != 0)

        leader = tuple(lead.predict(p).get(True, 0.0) > 0.5 for p in space)
        known = {tuple(h.predict(p).get(True, 0.0) > 0.5 for p in space)
                 for h in hypotheses.live() if h.name != OTHER}

        # Starting states: one per behaviour among the exact programs discovery
        # finds; failing that, a consistent program met in the prior.
        starts: Dict[tuple, tuple] = {}
        for program, _ in exact_programs(names, history, most=8):
            if size(program) <= most:
                starts.setdefault(behaviour(program), program)
        if not starts:
            for _ in range(batch):
                program = prior.sample(names, rng, most)
                if consistent(program):
                    starts.setdefault(behaviour(program), program)
                    break
        if not starts:
            return Estimate(None, 0.0, 1.0, 0, 0, batch, enough=False, decided=False,
                            language_short=True)

        chains = list(starts.values())
        marks: List[List[float]] = [[] for _ in chains]
        shortest: Dict[tuple, Tuple[float, str, tuple]] = {}
        steps = 0
        low, high, decided, ess = 0.0, 1.0, False, 0.0
        while steps < cap:
            for _ in range(batch):
                i = steps % len(chains)
                current = chains[i]
                paths = list(nodes(current))
                path, old = paths[rng.randrange(len(paths))]
                room = most - (size(current) - size(old))
                proposal = replace(current, path, prior.sample(names, rng, room))
                if consistent(proposal) and rng.random() < min(1.0, size(current) / size(proposal)):
                    current = chains[i] = proposal
                steps += 1
                shape = behaviour(current)
                marks[i].append(1.0 if shape == leader else 0.0)
                if shape != leader and shape not in known:
                    bits = description.program_bits(current, len(names))
                    held = shortest.get(shape)
                    if held is None or (bits, show(current)) < held[:2]:
                        shortest[shape] = (bits, show(current), current)
            total = sum(len(m) for m in marks)
            share = sum(sum(m) for m in marks) / total
            ess = sum(effective_size(m) for m in marks if m)
            low, high = wilson_rate(share, ess)
            if low >= confidence or high < confidence:
                decided = True
                break
        total = sum(len(m) for m in marks)
        agreeing = int(sum(sum(m) for m in marks))
        rivals = [Hypothesis(f"программа {text}", _predicts(program, names, noise))
                  for _, text, program in sorted(shortest.values())]
        return Estimate(agreeing / total, low, high, total, agreeing, steps,
                        enough=decided and low >= confidence, decided=decided,
                        language_short=False, rivals=rivals, ess=ess)

    return estimate
