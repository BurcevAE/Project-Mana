"""
mana.cognition.explain — new explanations when none of the known ones will do
(docs/РОЖДЕНИЕ_ГИПОТЕЗ.md).

`inquiry` has a signal it used to end on: OTHER leads, "none of my
hypotheses explains this". The signal says the space of explanations is not
enough -- not that the question has no answer. This turns it into a call on
discovery: from the history of the question alone -- the configurations
probed and what came of them -- build programs exact on that history, and
offer them as hypotheses.

They are candidates, not answers. Inquiry weighs them against everything
seen, tells them apart by acting, and challenges the leader before
accepting it, as it does any explanation it was given.

What the generator is given: the history. What it is not given: the rule,
its class, its name, or anything about the world beyond the conditions the
observations carry.

    offered     every distinct program the search evaluated on its way --
                its pools and beams, the anytime curve and the answer -- that
                is exact on the history, shortest description first, at most
                MOST
    mass        two to the minus the program's bits; `inquiry` shares out
                the prior mass OTHER holds in proportion to it
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..discovery import description
from ..discovery import policy as P
from ..discovery.language import Evaluator, show
from .inquiry import Hypothesis, HypothesisSet

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: At most so many explanations a call (declared, 9).
MOST = 8

#: Evaluations the search may spend on one call (declared, 10.2).
BUDGET = 20000


def _predicts(program, names: Sequence[str], noise: float) -> Callable[[Dict], Dict[bool, float]]:
    def predict(params: Dict) -> Dict[bool, float]:
        conditions = params["conditions"]
        columns = {n: np.array([int(bool(conditions[n]))], dtype=np.int64) for n in names}
        works = bool(Evaluator(columns)(program)[0] != 0)
        return ({True: 1.0 - noise, False: noise} if works
                else {True: noise, False: 1.0 - noise})
    return predict


def from_history(conditions: Sequence[str], noise: float = 0.0, most: int = MOST,
                 budget: int = BUDGET
                 ) -> Callable[[HypothesisSet], Tuple[List[Hypothesis], List[float]]]:
    """A generator of explanations for questions over these conditions."""
    names = list(conditions)

    def explain(hypotheses: HypothesisSet) -> Tuple[List[Hypothesis], List[float]]:
        rows = hypotheses.history
        if not rows:
            return [], []
        columns = {n: np.array([int(bool(params["conditions"][n])) for params, _ in rows],
                               dtype=np.int64) for n in names}
        works = np.array([bool(outcome) for _, outcome in rows])
        rounds: list = []
        found = P.run(P.with_budget(P.CURRENT, budget), columns, works.astype(np.int64),
                      profile=True, log_rounds=rounds)
        evaluator = Evaluator(columns)
        exact: Dict[tuple, float] = {}
        # Every program the search evaluated, not only its records: on a few
        # rows the measure prefers a short program with one error to an exact
        # one, and the record-breakers may hold no exact program at all.
        seen = [found.program] + [p for _, p, _ in found.anytime]
        seen += [p for pool, beam in rounds for p in list(pool) + list(beam)]
        for program in seen:
            if program in exact:
                continue
            if np.array_equal(np.asarray(evaluator(program)) != 0, works):
                exact[program] = description.program_bits(program, len(names))
        chosen = sorted(exact.items(), key=lambda kv: (kv[1], show(kv[0])))[:most]
        born = [Hypothesis(f"программа {show(p)}", _predicts(p, names, noise))
                for p, _ in chosen]
        return born, [2.0 ** -bits for _, bits in chosen]

    return explain
