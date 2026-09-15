"""
mana.discovery.ladder — a ladder of necessary depth (H3, D0).

docs/ГЛУБИНА_D0.md, section 4. Worlds whose truth is built from levels of
decomposition and generated per seed, so that the ladder is not a dozen
favourite programs. Like worlds.py, an oracle: the truth is written here,
and the learner modules must never import this one -- a test reads their
source to hold them to it.

    R   residuals: a sum of atoms                 a1 + (a2 + (... + ak))
    S   cases: nested conditions                  if(c1, l1, if(c2, l2, ... l(k+1)))
    M   R and S levels alternating, from a random start, around an atom
    C1  out of D1's catalogue: if((v + w) + c < u + t, 1, 0) -- a
        comparison of sums, which no operator of the catalogue inverts

Rung k has k levels; rung 1 of R and S is flat and is the control C0.
Noise -- the control C2 -- is a parameter, declared and left out of the
first calibration. F, a derived quantity used more than once, is built
in a later step.

The first calibration (docs/ГЛУБИНА_D0.md, 4.6) found no gap in S: the
flat search's own edit "add a condition" is a greedy decomposition by
cases, and it solved S3 on 10 seeds of 10. By the owner's decision S is a
control since -- a decomposition native to the flat search, where D1 must
not gain -- and the ladder is R and M.

Only what the flat search can reach. A truth is made of leaves, +, -, and
if, with a comparison only as the condition of an if -- what the search's
edits make. A truth the search could not even write would make it fail by
reach, not by depth. And an atom of R is a condition, if(c, a, b), not a
linear sum: linear atoms add up to one short linear form, which has no
depth at all.

The rule of every world is computed by an interpreter written here, not by
language.Evaluator: an oracle that shared the learner's evaluator would
agree with it on its bugs.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np

from .language import (ADD, CMP, CONST, GET, IF, LESS, SUB, Program, add, cmp, const, get,
                       if_, show)
from .worlds import World

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

VARIABLES = ("w", "x", "y", "z")
LOW, HIGH = 0, 9
#: Every family the generator builds.
FAMILIES = ("R", "S", "M")
#: The instrument: the families measured for depth.
LADDER = ("R", "M")
#: Controls with rungs: a decomposition native to the flat search.
NATIVE = ("S",)
CONTROLS = ("C1",)
RUNGS = (1, 2, 3, 4, 5)
#: Every branch of a condition holds on at least this share of the world.
MIN_BRANCH = 0.10
#: The control C2: a share of outcomes replaced at random. The second wave.
NOISE = 0.10
ATTEMPTS = 2000


def value(p: Program, state: Dict[str, int]) -> int:
    """The truth on one state, by this module's own interpreter."""
    kind = p[0]
    if kind == GET:
        return int(state[p[1]])
    if kind == CONST:
        return int(p[1])
    if kind == CMP:
        a, b = value(p[2], state), value(p[3], state)
        return int(a < b) if p[1] == LESS else int(a == b)
    if kind == ADD:
        return value(p[1], state) + value(p[2], state)
    if kind == SUB:
        return value(p[1], state) - value(p[2], state)
    if kind == IF:
        return value(p[2], state) if value(p[1], state) != 0 else value(p[3], state)
    raise ValueError(f"not in the ladder's language: {kind!r}")


def _vector(p: Program, cols: Dict[str, np.ndarray]) -> np.ndarray:
    """The same interpreter over many states at once, for the generator."""
    kind = p[0]
    if kind == GET:
        return cols[p[1]]
    if kind == CONST:
        return np.full(len(cols[VARIABLES[0]]), p[1], dtype=np.int64)
    if kind == CMP:
        a, b = _vector(p[2], cols), _vector(p[3], cols)
        return ((a < b) if p[1] == LESS else (a == b)).astype(np.int64)
    if kind == ADD:
        return _vector(p[1], cols) + _vector(p[2], cols)
    if kind == SUB:
        return _vector(p[1], cols) - _vector(p[2], cols)
    if kind == IF:
        return np.where(_vector(p[1], cols) != 0, _vector(p[2], cols), _vector(p[3], cols))
    raise ValueError(f"not in the ladder's language: {kind!r}")


_STATES: Dict[str, np.ndarray] = {}


def _states() -> Dict[str, np.ndarray]:
    if not _STATES:
        grid = np.array(list(itertools.product(range(LOW, HIGH + 1), repeat=len(VARIABLES))),
                        dtype=np.int64)
        _STATES.update({v: grid[:, i] for i, v in enumerate(VARIABLES)})
    return _STATES


def _rng(family: str, rung: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([sum(ord(ch) * 31 ** i for i, ch in enumerate(family)),
                                  rung, seed, 101])


def _condition(rng: np.random.Generator) -> Program:
    v = VARIABLES[int(rng.integers(len(VARIABLES)))]
    kind = int(rng.integers(3))
    if kind == 0:
        return cmp(LESS, get(v), const(int(rng.integers(1, HIGH + 1))))
    if kind == 1:
        return cmp(LESS, const(int(rng.integers(LOW, HIGH))), get(v))
    w = [u for u in VARIABLES if u != v][int(rng.integers(len(VARIABLES) - 1))]
    return cmp(LESS, get(v), get(w))


def _leaf(rng: np.random.Generator) -> Program:
    if rng.random() < 0.7:
        return get(VARIABLES[int(rng.integers(len(VARIABLES)))])
    return const(int(rng.integers(LOW, HIGH + 1)))


def _share(condition: Program, region: np.ndarray) -> float:
    return float(np.mean((_vector(condition, _states()) != 0) & region))


def _atom(rng: np.random.Generator, taken: set) -> Program:
    """if(c, a, b): a condition true on 10%..90% of the world, two different
    leaves, and a condition no other atom of the same truth uses."""
    everywhere = np.ones(len(_states()[VARIABLES[0]]), dtype=bool)
    for _ in range(ATTEMPTS):
        c = _condition(rng)
        a, b = _leaf(rng), _leaf(rng)
        if a == b or show(c) in taken:
            continue
        if MIN_BRANCH <= _share(c, everywhere) <= 1 - MIN_BRANCH:
            taken.add(show(c))
            return if_(c, a, b)
    raise RuntimeError("no atom found")


def _residuals(rng: np.random.Generator, rung: int) -> Program:
    taken: set = set()
    atoms = [_atom(rng, taken) for _ in range(rung)]
    out = atoms[-1]
    for atom in reversed(atoms[:-1]):
        out = add(atom, out)
    return out


def _cases(rng: np.random.Generator, rung: int) -> Program:
    """k conditions, k + 1 leaves; each branch holds on at least MIN_BRANCH
    of the world, and the conditions leave enough for the branches below."""
    for _ in range(ATTEMPTS):
        region = np.ones(len(_states()[VARIABLES[0]]), dtype=bool)
        conditions: List[Program] = []
        for level in range(rung):
            found: Optional[Program] = None
            for _ in range(200):
                c = _condition(rng)
                true = (_vector(c, _states()) != 0) & region
                left = region & ~true
                if (np.mean(true) >= MIN_BRANCH
                        and np.mean(left) >= MIN_BRANCH * (rung - level)):
                    found = c
                    region = left
                    break
            if found is None:
                break
            conditions.append(found)
        if len(conditions) < rung:
            continue
        leaves: List[Program] = []
        while len(leaves) < rung + 1:
            leaf = _leaf(rng)
            if not leaves or leaf != leaves[-1]:
                leaves.append(leaf)
        out = leaves[-1]
        for c, leaf in zip(reversed(conditions), reversed(leaves[:-1])):
            out = if_(c, leaf, out)
        return out
    raise RuntimeError("no nested cases found")


def _mixed(rng: np.random.Generator, rung: int) -> Program:
    """Around an atom, rung - 1 levels alternating between a residual and a
    case, from a random start. A case's condition holds on at least
    MIN_BRANCH of the world and fails on at least as much."""
    taken: set = set()
    out = _atom(rng, taken)
    kind = int(rng.integers(2))
    everywhere = np.ones(len(_states()[VARIABLES[0]]), dtype=bool)
    for _ in range(rung - 1):
        if kind == 0:
            out = add(_atom(rng, taken), out)
        else:
            for _ in range(ATTEMPTS):
                c = _condition(rng)
                share = _share(c, everywhere)
                if show(c) not in taken and MIN_BRANCH <= share <= 1 - MIN_BRANCH:
                    taken.add(show(c))
                    out = if_(c, _leaf(rng), out)
                    break
            else:
                raise RuntimeError("no case found")
        kind = 1 - kind
    return out


def _outside(rng: np.random.Generator) -> Program:
    """C1: if((v + w) + c < u + t, 1, 0), true on 30%..70% of the world."""
    for _ in range(ATTEMPTS):
        v, w, u, t = (VARIABLES[i] for i in rng.permutation(len(VARIABLES)))
        c = int(rng.integers(0, 5))
        condition = cmp(LESS, add(add(get(v), get(w)), const(c)), add(get(u), get(t)))
        everywhere = np.ones(len(_states()[VARIABLES[0]]), dtype=bool)
        if 0.3 <= _share(condition, everywhere) <= 0.7:
            return if_(condition, const(1), const(0))
    raise RuntimeError("no comparison found")


def truth(family: str, rung: int, seed: int) -> Program:
    """The truth of one world of the ladder: the same for the same family,
    rung and seed, whenever it is asked for."""
    rng = _rng(family, rung, seed)
    if family == "R":
        return _residuals(rng, rung)
    if family == "S":
        return _cases(rng, rung)
    if family == "M":
        return _mixed(rng, rung)
    if family == "C1":
        if rung != 1:
            raise ValueError("C1 has one rung")
        return _outside(rng)
    raise ValueError(f"no such family: {family!r}")


def world(family: str, rung: int, seed: int, noise: float = 0.0) -> World:
    p = truth(family, rung, seed)
    return World(f"{family}{rung}/{seed}", VARIABLES, LOW, HIGH,
                 lambda state, p=p: value(p, state), p, noise=noise, note=show(p))
