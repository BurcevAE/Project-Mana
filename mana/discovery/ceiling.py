"""
mana.discovery.ceiling — what the language can say, measured apart from any
search.

A search that does not find a program has not shown there is none. So two
questions are asked here without walking the way the search walks:

    exists    is there a program of this language that fits the data
              exactly? Answered by a certificate where one is known: a
              program in the language that reproduces every state.
    smallest  how small can it be? Answered by enumerating programs
              bottom-up, size by size, keeping one per distinct behaviour
              on the data -- observational equivalence: two programs that
              give the same outputs on everything seen are the same answer
              there, and only the smaller is kept, which cannot lose a
              smallest program because anything built from the dropped one
              behaves like something built, no larger, from the kept one.
              The first size with an exact fit is the smallest. If the
              enumeration stops first -- the distinct behaviours outgrow the
              cap -- what is known is a lower bound: nothing smaller fits.

A measuring instrument, not a learner, and not a fair competitor: it is
exhaustive over the sizes it reaches, which no learner of a real world can
afford, and it is handed the target it measures. It exists to tell a
search's failure from the language's.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .language import (EQUAL, LESS, Evaluator, Program, add, cmp, if_, show, sub)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

MAX_SIZE = 7
MAX_CLASSES = 300000

FOUND, SIZE, CLASSES = "found", "size", "classes"


@dataclass
class Ceiling:
    #: Size of the smallest exact program, when the enumeration reached it.
    smallest: Optional[int]
    program: Optional[Program]
    #: Every size up to this one was enumerated completely without a fit.
    lower_bound: int
    #: Distinct behaviours enumerated.
    classes: int
    #: FOUND, or why it stopped without: SIZE (hit max_size) or CLASSES.
    stopped: str
    seconds: float

    def describe(self) -> str:
        if self.smallest is not None:
            return (f"наименьшая точная программа — {self.smallest} узлов: "
                    f"{show(self.program)}")
        why = "предел размера" if self.stopped == SIZE else "предел числа поведений"
        return (f"до {self.lower_bound} узлов точной программы нет "
                f"({why}; перебрано {self.classes} поведений)")


def smallest_fit(columns: Dict[str, Sequence[int]], target: Sequence[int],
                 leaves: Sequence[Program], max_size: int = MAX_SIZE,
                 max_classes: int = MAX_CLASSES) -> Ceiling:
    """The smallest program over these leaves that reproduces the target."""
    started = time.time()
    evaluate = Evaluator(columns)
    goal = np.asarray(target, dtype=np.int16).tobytes()
    levels: Dict[int, List[Tuple[Program, np.ndarray]]] = {}
    #: Distinct "is non-zero" masks, the only thing a condition contributes.
    masks: Dict[int, List[Tuple[Program, np.ndarray]]] = {}
    seen, masks_seen = set(), set()
    count = 0

    def stop(stopped: str, bound: int) -> Ceiling:
        return Ceiling(None, None, bound, count, stopped, time.time() - started)

    def admit(size: int, program: Program, vector: np.ndarray) -> Optional[Ceiling]:
        nonlocal count
        key = vector.tobytes()
        if key in seen:
            return None
        seen.add(key)
        count += 1
        levels.setdefault(size, []).append((program, vector))
        mask = vector != 0
        mask_key = mask.tobytes()
        if mask_key not in masks_seen:
            masks_seen.add(mask_key)
            masks.setdefault(size, []).append((program, mask))
        if key == goal:
            return Ceiling(size, program, size - 1, count, FOUND, time.time() - started)
        return None

    for leaf in leaves:
        hit = admit(1, leaf, np.asarray(evaluate(leaf), dtype=np.int16))
        if hit:
            return hit
    for size in range(2, max_size + 1):
        for left in range(1, size - 1):
            right = size - 1 - left
            for pa, va in levels.get(left, []):
                for pb, vb in levels.get(right, []):
                    for program, vector in (
                            (add(pa, pb), va + vb), (sub(pa, pb), va - vb),
                            (cmp(LESS, pa, pb), (va < vb).astype(np.int16)),
                            (cmp(EQUAL, pa, pb), (va == vb).astype(np.int16))):
                        hit = admit(size, program, vector)
                        if hit:
                            return hit
                        if count >= max_classes:
                            return stop(CLASSES, size - 1)
        for cond in range(1, size - 2):
            for then in range(1, size - 1 - cond):
                other = size - 1 - cond - then
                if other < 1:
                    continue
                for pc, mc in masks.get(cond, []):
                    for pa, va in levels.get(then, []):
                        for pb, vb in levels.get(other, []):
                            hit = admit(size, if_(pc, pa, pb), np.where(mc, va, vb))
                            if hit:
                                return hit
                            if count >= max_classes:
                                return stop(CLASSES, size - 1)
    return stop(SIZE, max_size)
