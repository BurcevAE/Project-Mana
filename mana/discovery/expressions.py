"""
mana.discovery.expressions — the space of experiment programs of the
contract's language, by form and size (D3-prep-1, docs/ГЛУБИНА_D3.md, 10).

An experiment is an expression mana/discovery/questions.py runs. Here that
space is defined as an object: counted without being built, enumerated in
one fixed order, and sampled uniformly. Nothing is learnt and nothing is
chosen -- what to do with a program is the probe's business.

The forms are the contract's: a program (P), a vector over the points (V),
a mask (M), a number (N), a list of programs (L). An expression is
admitted when every argument has the form its operation asks for, and the
whole has form P -- the catalogue's derivation reads X as a program.

The declared restrictions (10.1), all of them the human's choice:

    successors, flatmap, list   not here: the first needs rules from
                                experience, the others multiply lists
                                without paying for anything
    candidates(k)               at most one, never inside a lambda, k <= 6
    lambda                      only the second argument of argmin/argmax,
                                one bound variable, x, of form P; so no
                                argmin/argmax inside a lambda, which would
                                bind a second
    constants                   K below, and `share`, the node's share
    arithmetic                  a number that reads nothing may only be an
                                atom: plus(2, 3) names what a constant
                                names, and such nestings are most of the
                                space

Size is the number of nodes: a lambda, a variable and a constant are one
each. The order is by size; within a size by the order of OPS; within an
operation by the size of the left part, then by the order of the parts.
"""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The constants an expression may name: small numbers and one round one.
K = (0, 1, 2, 3, 4, 5, 6, 7, 8, 16)

#: The sizes `candidates` may ask for.
CANDIDATE_SIZES = (1, 2, 3, 4, 5, 6)

#: The ladder of sizes the probe walks, and the largest it may reach.
LADDER = (4, 6, 8, 10, 12)
MOST = 12

FORMS = ("P", "V", "M", "N", "L")

#: What an expression of each form may be, at size 1.
ATOMS: Dict[str, Tuple[tuple, ...]] = {
    "P": (("own",),),
    "V": (("target",),),
    "M": (("domain",),),
    "N": tuple(("const", k) for k in K) + (("var", "share"),),
    "L": (("leaves",), ("conditions",), ("held",)),
}

#: The bound variable of a lambda, an expression of form P inside it.
BOUND = ("var", "x")

#: (name, the form it gives, the forms it asks for) -- in the order of the
#: table of docs/ГЛУБИНА_D3.md, 10.1. "Λ" is a lambda whose body is a
#: number; "K" is one of CANDIDATE_SIZES, written as a constant.
OPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("item", "P", ("L", "N")),
    ("argmin", "P", ("L", "Λ")),
    ("argmax", "P", ("L", "Λ")),
    ("search", "P", ("V", "M", "N")),
    ("eval", "V", ("P",)),
    ("bool", "V", ("M",)),
    ("vadd", "V", ("V", "V")),
    ("vsub", "V", ("V", "V")),
    ("not", "M", ("M",)),
    ("mask", "M", ("V",)),
    ("eq", "M", ("V", "V")),
    ("neq", "M", ("V", "V")),
    ("and", "M", ("M", "M")),
    ("or", "M", ("M", "M")),
    ("count", "N", ("M",)),
    ("bits", "N", ("P",)),
    ("distinct", "N", ("V", "M")),
    ("plus", "N", ("N", "N")),
    ("minus", "N", ("N", "N")),
    ("div", "N", ("N", "N")),
    ("probe", "N", ("V", "M", "N")),
    ("candidates", "L", ("K",)),
    ("take", "L", ("N", "L")),
)

#: Inside a lambda: no candidates (restriction 4) and no second bound
#: variable (restriction 5).
NOT_IN_LAMBDA = ("argmin", "argmax", "candidates")

#: Arithmetic over numbers. A number that reads nothing -- arithmetic over
#: constants alone -- may only be an atom (restriction 7): plus(2, 3) names
#: what a constant names, and its nestings are most of the space.
ARITHMETIC = ("plus", "minus", "div")


def _constant(e) -> bool:
    """Reads nothing: under restriction 7 that is a constant and nothing else."""
    return e[0] == "const"


def _admitted(name: str, parts: Sequence[tuple]) -> bool:
    return name not in ARITHMETIC or not all(_constant(p) for p in parts)


def _atoms(form: str, lam: bool) -> Tuple[tuple, ...]:
    if form == "P" and lam:
        return ATOMS["P"] + (BOUND,)
    return ATOMS[form]


# -- counting: how large the space is, without building it --------------------

_COUNTS: Dict[Tuple[str, int, bool], Tuple[int, int]] = {}


def count(form: str, size: int, lam: bool = False) -> Tuple[int, int]:
    """How many expressions of this form and size there are: without a
    `candidates` in them, and with exactly one."""
    if size < 1:
        return 0, 0
    key = (form, size, lam)
    if key in _COUNTS:
        return _COUNTS[key]
    without, with_one = (len(_atoms(form, lam)), 0) if size == 1 else (0, 0)
    for name, gives, asks in OPS:
        if gives != form or (lam and name in NOT_IN_LAMBDA):
            continue
        if name == "candidates":
            if size == 2:
                with_one += len(CANDIDATE_SIZES)
            continue
        for parts in _cuts(asks, size - 1):
            counted = [_count_arg(a, n, lam) for a, n in zip(asks, parts)]
            whole = 1
            for a, _ in counted:
                whole *= a
            if name in ARITHMETIC and all(n == 1 for n in parts):
                whole -= len(K) ** len(parts)              # restriction 7
            without += whole
            for i, (_, b) in enumerate(counted):
                if b:
                    other = 1
                    for j, (a, _) in enumerate(counted):
                        if j != i:
                            other *= a
                    with_one += b * other
    _COUNTS[key] = (without, with_one)
    return _COUNTS[key]


def _count_arg(asks: str, size: int, lam: bool) -> Tuple[int, int]:
    if asks == "K":
        return (len(CANDIDATE_SIZES), 0) if size == 1 else (0, 0)
    if asks == "Λ":
        return (count("N", size - 1, True)[0], 0) if size >= 2 else (0, 0)
    return count(asks, size, lam)


def _cuts(asks: Sequence[str], room: int) -> Iterator[Tuple[int, ...]]:
    """The sizes of the parts, the left one first -- the declared order."""
    if len(asks) == 1:
        if room >= 1:
            yield (room,)
        return
    for first in range(1, room):
        for rest in _cuts(asks[1:], room - first):
            yield (first,) + rest


def total(size: int, form: str = "P", lam: bool = False) -> int:
    a, b = count(form, size, lam)
    return a + b


# -- enumeration: the one fixed order ------------------------------------------

def generate(size: int, form: str = "P") -> Iterator[tuple]:
    """Every expression of this form and size, in the declared order."""
    for expression, _ in _of(form, size, False, True):
        yield expression


def _of(form: str, size: int, lam: bool, free: bool) -> Iterator[Tuple[tuple, bool]]:
    """(expression, whether it holds a `candidates`); `free` says one may."""
    if size < 1:
        return
    if size == 1:
        for atom in _atoms(form, lam):
            yield atom, False
        return
    for name, gives, asks in OPS:
        if gives != form or (lam and name in NOT_IN_LAMBDA):
            continue
        if name == "candidates":
            if size == 2 and free:
                for k in CANDIDATE_SIZES:
                    yield ("candidates", ("const", k)), True
            continue
        for parts in _cuts(asks, size - 1):
            yield from _made(name, asks, parts, lam, free)


def _made(name, asks, parts, lam, free) -> Iterator[Tuple[tuple, bool]]:
    made: List[tuple] = []
    taken = False

    def onwards(i: int) -> Iterator[Tuple[tuple, bool]]:
        nonlocal taken
        if i == len(asks):
            if _admitted(name, made):
                yield (name,) + tuple(made), taken
            return
        was = taken
        for part, used in _arg(asks[i], parts[i], lam, free and not taken):
            made.append(part)
            taken = was or used
            yield from onwards(i + 1)
            made.pop()
            taken = was

    yield from onwards(0)


def _arg(asks: str, size: int, lam: bool, free: bool) -> Iterator[Tuple[tuple, bool]]:
    if asks == "K":
        if size == 1:
            for k in CANDIDATE_SIZES:
                yield ("const", k), False
        return
    if asks == "Λ":
        if size >= 2:
            for body, _ in _of("N", size - 1, True, False):
                yield ("lambda", "x", body), False
        return
    yield from _of(asks, size, lam, free)


# -- a uniform sample of the same space, for the diagnostic baseline ----------

def sample(size: int, rng: random.Random, form: str = "P") -> tuple:
    """One expression of this form and size, uniformly among them all."""
    return _sampled(form, size, False, True, rng)


def _weight(form: str, size: int, lam: bool, free: bool) -> int:
    a, b = count(form, size, lam)
    return a + (b if free else 0)


def _weight_arg(asks: str, size: int, lam: bool, free: bool) -> int:
    if asks == "K":
        return len(CANDIDATE_SIZES) if size == 1 else 0
    if asks == "Λ":
        return count("N", size - 1, True)[0] if size >= 2 else 0
    return _weight(asks, size, lam, free)


def _pick(weighted: Sequence[Tuple[int, object]], rng: random.Random):
    whole = sum(w for w, _ in weighted)
    if whole <= 0:
        raise ValueError("пустое пространство")
    point = rng.randrange(whole)
    for w, what in weighted:
        if point < w:
            return what
        point -= w
    raise AssertionError


def _sampled(form: str, size: int, lam: bool, free: bool, rng: random.Random) -> tuple:
    ways: List[Tuple[int, object]] = []
    if size == 1:
        ways += [(1, atom) for atom in _atoms(form, lam)]
    for name, gives, asks in OPS:
        if gives != form or (lam and name in NOT_IN_LAMBDA):
            continue
        if name == "candidates":
            if size == 2 and free:
                ways += [(1, ("candidates", ("const", k))) for k in CANDIDATE_SIZES]
            continue
        for parts in _cuts(asks, size - 1):
            weight = 1
            for a, n in zip(asks, parts):
                weight *= _weight_arg(a, n, lam, free)
            if name in ARITHMETIC and all(n == 1 for n in parts):
                weight -= len(K) ** len(parts)             # restriction 7
            if weight > 0:
                ways.append((weight, (name, asks, parts)))
    chosen = _pick(ways, rng)
    if type(chosen) is not tuple or len(chosen) != 3 or chosen[0] not in dict(
            (name, True) for name, _, _ in OPS):
        return chosen
    name, asks, parts = chosen
    while True:
        made, taken = [], False
        for a, n in zip(asks, parts):
            part = _sampled_arg(a, n, lam, free and not taken, rng)
            taken = taken or _holds_candidates(part)
            made.append(part)
        if _admitted(name, made):
            return (name,) + tuple(made)


def _sampled_arg(asks: str, size: int, lam: bool, free: bool, rng: random.Random):
    if asks == "K":
        return ("const", rng.choice(CANDIDATE_SIZES))
    if asks == "Λ":
        return ("lambda", "x", _sampled("N", size - 1, True, False, rng))
    return _sampled(asks, size, lam, free, rng)


def _holds_candidates(e) -> bool:
    if type(e) is not tuple or not e:
        return False
    return e[0] == "candidates" or any(_holds_candidates(part) for part in e[1:])
