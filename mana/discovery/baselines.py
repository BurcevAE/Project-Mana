"""
mana.discovery.baselines — what a learner without structure gets.

    table   remember every training state; an unseen one gets the most
            common outcome. Perfect on what it saw, blind beyond it.
    tree    a decision tree (scikit-learn), the ordinary machine-learning
            answer: splits on the variables, no notion of "copy y".

They are the floor the search has to clear. Like the search, they are
handed data and never the world.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence, Tuple

import numpy as np

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


def _rows(columns: Dict[str, Sequence[int]]) -> np.ndarray:
    names = sorted(columns)
    return np.stack([np.asarray(columns[name], dtype=np.int64) for name in names], axis=1)


def table_predict(train: Dict[str, Sequence[int]], outcomes: Sequence[int],
                  test: Dict[str, Sequence[int]]) -> np.ndarray:
    remembered: Dict[tuple, Counter] = {}
    for row, outcome in zip(_rows(train), outcomes):
        remembered.setdefault(tuple(row.tolist()), Counter())[int(outcome)] += 1
    common = Counter(int(o) for o in outcomes).most_common(1)[0][0]
    return np.array([remembered[key].most_common(1)[0][0]
                     if (key := tuple(row.tolist())) in remembered else common
                     for row in _rows(test)], dtype=np.int64)


def tree_predict(train: Dict[str, Sequence[int]], outcomes: Sequence[int],
                 test: Dict[str, Sequence[int]], seed: int = 0) -> Tuple[np.ndarray, int]:
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=seed)
    model.fit(_rows(train), np.asarray(outcomes, dtype=np.int64))
    return model.predict(_rows(test)).astype(np.int64), int(model.tree_.node_count)
