"""
mana.discovery.invent — changing the language: a variable nobody observed.

The question a program cannot answer about itself: "P explains almost
everything except E -- is E structured, and what is missing?" Compressing
fragments cannot answer it; it only repacks what the language can already
say. What can is a new quantity.

The operation, in five moves, none of which knows what any world is:

    1  P, the shortest program in the current language, leaves errors E.
    2  B, the shortest program for E alone: what explains the anomalies.
    3  Two explanations that together cover the most of what was seen --
       chosen among the leaves, P and B, by their bits plus the bits for
       what neither explains. Where exactly one of them is right, that is
       the value of a hidden binary quantity v1.
    4  How v1 comes about is searched like any program -- the same edits,
       the same poor language -- over what is seen at a step and v1's own
       previous value. A quantity that needs its past is a memory; one
       that does not is a concept the observation already contained.
    5  The outcome is then written with v1 in the language, and the change
       is kept only if everything together -- the outcome program, v1's
       definition, and the errors left -- is shorter than P alone.

The last move is the whole guard. Noise has no structure a program of
its own past can compress, so on noise the new variable costs more bits
than it saves and is refused; the same currency that stops a table of
exceptions stops an invented cause for them.

What is given, said plainly: the form of the change -- a hidden quantity
that chooses between two explanations and evolves by a program of its
past. Not that it is a switch, a counter or a threshold: that is what
step 4 has to find.

The first version took the pair to be P and B, and failed on W2. The
shortest program in a language too poor for the world is a compromise --
if(x < if(1 < x, 5 - a, a), y, x) -- that smears the hidden cause over
what is observed, and its errors inherit the smear: B came out as
if(x == 2, x, if(x == 3, x, if(x == 4, x, y))), the labels carried x's
artefacts, and no transition could be found. Two explanations chosen for
what they cover together -- x and y, a hundred per cent between them --
do not carry the compromise.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import log2
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import description, search
from .language import Evaluator, Program, get, if_, show

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The invented quantity, and its value one step earlier.
VAR = "v1"
PREV = "v1_prev"

#: Fewer errors than this and there is nothing to explain.
MIN_ANOMALIES = 8

#: Declaring a new variable, and naming its starting value.
DECLARE_BITS = 2.0


@dataclass
class Invented:
    accepted: bool
    note: str
    base: search.Found
    anomaly: Optional[search.Found] = None
    outcome: Optional[Program] = None
    transition: Optional[Program] = None
    init: int = 0
    bits: float = 0.0
    outcome_bits: float = 0.0
    transition_bits: float = 0.0
    #: Points where exactly one of the pair was right: where v1 was seen.
    labelled: float = 0.0
    #: The two explanations v1 was introduced to choose between.
    pair: Tuple[Program, ...] = ()

    def describe(self) -> str:
        between = (f" (выбор между {show(self.pair[0])} и {show(self.pair[1])})"
                   if len(self.pair) == 2 else "")
        if not self.accepted:
            return f"язык не изменён{between}: {self.note}"
        return (f"новая переменная {VAR}{between}: {VAR} = {show(self.transition)} "
                f"(начало {self.init}; {PREV} — её прошлое значение); "
                f"исход = {show(self.outcome)}; {self.bits:.1f} бит против "
                f"{self.base.bits:.1f} без неё")


def _flat(columns: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: np.asarray(values).reshape(-1) for name, values in columns.items()}


def run(transition: Program, init: int, columns: Dict[str, np.ndarray]) -> np.ndarray:
    """v1 along every episode, from its starting value, step by step."""
    episodes, steps = np.asarray(next(iter(columns.values()))).shape
    out = np.zeros((episodes, steps), dtype=np.int64)
    prev = np.full(episodes, int(init), dtype=np.int64)
    for t in range(steps):
        at = {name: np.asarray(values)[:, t] for name, values in columns.items()}
        at[PREV] = prev
        now = (Evaluator(at)(transition) != 0).astype(np.int64)
        out[:, t] = now
        prev = now
    return out


def _transition_rows(columns: Dict[str, np.ndarray], labels: np.ndarray,
                     init: int) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """Steps where v1 and its previous value are both known."""
    episodes = labels.shape[0]
    prev = np.concatenate([np.full((episodes, 1), int(init)), labels[:, :-1]], axis=1)
    known = (labels >= 0) & (prev >= 0)
    if int(known.sum()) < MIN_ANOMALIES:
        return None
    rows = {name: np.asarray(values)[known] for name, values in columns.items()}
    rows[PREV] = prev[known]
    return rows, labels[known]


def _pair(flat: Dict[str, np.ndarray], actual: np.ndarray,
          candidates: Sequence[Program], variables: int,
          alphabet: int) -> Tuple[Program, Program]:
    """The two explanations that together describe the most for the least.

    Scored as if a perfect selector existed: both programs, plus the
    points neither explains. The selector itself is paid for later, by
    the program that has to compute it.
    """
    evaluator = Evaluator(flat)
    right = {candidate: evaluator(candidate) == actual for candidate in candidates}
    n = len(actual)
    best: Optional[Tuple[float, Program, Program]] = None
    for first, second in itertools.combinations(candidates, 2):
        missed = n - int(np.count_nonzero(right[first] | right[second]))
        # Both programs, then what neither explains: how many, where, and
        # the value at each.
        bits = (description.program_bits(first, variables)
                + description.program_bits(second, variables)
                + log2(n + 1) + description._log2_choose(n, missed)
                + missed * log2(max(2, alphabet)))
        if best is None or bits < best[0]:
            best = (bits, first, second)
    return best[1], best[2]


def invent(columns: Dict[str, np.ndarray], outcomes: np.ndarray,
           budget: int = search.BUDGET) -> Invented:
    """The shortest description with one invented variable, or the reason
    there is none."""
    outcomes = np.asarray(outcomes, dtype=np.int64)
    shape = outcomes.shape
    flat = _flat(columns)
    actual = outcomes.reshape(-1)
    alphabet = description.alphabet_of(actual)
    base = search.search(flat, actual, budget=budget)
    base_predicted = search.predict(base.program, flat)
    wrong = base_predicted != actual
    if int(wrong.sum()) < MIN_ANOMALIES:
        return Invented(False, f"ошибок {int(wrong.sum())}, объяснять нечего", base)

    anomaly = search.search({name: values[wrong] for name, values in flat.items()},
                            actual[wrong], budget=budget)
    leaves, _ = search.vocabulary(flat, actual)
    first, second = _pair(flat, actual, leaves + [base.program, anomaly.program],
                          len(columns), alphabet)
    evaluator = Evaluator(flat)
    right_a, right_b = evaluator(first) == actual, evaluator(second) == actual
    labels = np.full(actual.shape, -1, dtype=np.int64)
    labels[right_a & ~right_b] = 1
    labels[right_b & ~right_a] = 0
    labels = labels.reshape(shape)
    labelled = float(np.mean(labels >= 0))

    variables = len(columns) + 1
    selector = if_(get(VAR), first, second)
    best: Optional[Tuple[float, Program, int, np.ndarray, float]] = None
    for init in (0, 1):
        rows = _transition_rows(columns, labels, init)
        if rows is None:
            continue
        step = search.search(rows[0], rows[1], budget=budget)
        value = run(step.program, init, columns).reshape(-1)
        outcome_part = (description.program_bits(selector, variables)
                        + description.error_bits(Evaluator({**flat, VAR: value})(selector),
                                                 actual, alphabet))
        transition_part = description.program_bits(step.program, variables)
        total = outcome_part + transition_part + DECLARE_BITS
        if best is None or total < best[0]:
            best = (total, step.program, init, value, transition_part)
    if best is None:
        return Invented(False, "где скрытая величина видна, слишком мало точек",
                        base, anomaly, labelled=labelled, pair=(first, second))

    total, transition, init, value, transition_part = best
    # With the variable in the language, the outcome program is searched
    # again from nothing: the selector is one way to use v1, not the only.
    refined = search.search({**flat, VAR: value}, actual, budget=budget)
    outcome, outcome_part = selector, total - transition_part - DECLARE_BITS
    if refined.bits < outcome_part:
        outcome, outcome_part = refined.program, refined.bits
    total = outcome_part + transition_part + DECLARE_BITS
    accepted = total < base.bits
    note = ("короче на %.1f бит" % (base.bits - total) if accepted
            else "не окупается: %.1f бит против %.1f без неё" % (total, base.bits))
    return Invented(accepted, note, base, anomaly, outcome, transition, init,
                    total, outcome_part, transition_part, labelled,
                    pair=(first, second))


def hidden(found: Invented, columns: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """The invented quantity along new episodes, or None if there is none."""
    if not found.accepted or found.transition is None:
        return None
    return run(found.transition, found.init, columns)


def predict(found: Invented, columns: Dict[str, np.ndarray]) -> np.ndarray:
    shape = np.asarray(next(iter(columns.values()))).shape
    flat = _flat(columns)
    if not found.accepted:
        return search.predict(found.base.program, flat).reshape(shape)
    value = hidden(found, columns).reshape(-1)
    return Evaluator({**flat, VAR: value})(found.outcome).reshape(shape)
