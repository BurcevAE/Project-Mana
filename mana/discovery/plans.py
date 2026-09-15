"""
mana.discovery.plans — the catalogue of plans for research questions, as
data (docs/ГЛУБИНА_ВОПРОС_ИССЛЕДОВАНИЯ.md). mana/discovery/questions.py runs
any of them and knows none of them: take a plan out of the list it is given,
and what the plan did is gone.

    D1B          D1b's three branches -- X is the question's own answer. Run
                 through the interpreter in this order they are D1b-A
    DIAGNOSTIC   A and C of docs/ГЛУБИНА_ЭКСПЕРИМЕНТ.md: diagnostic candidates
                 of equal standing, none presumed right (owner's decision)
    p2_step(k)   one exploratory step of P2's class: the best successor, by
                 the rules bound to `ahead`, of the first k states held
"""
from __future__ import annotations

from .language import add, if_, sub
from .questions import Entry, Plan

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


def v(name: str) -> tuple:
    return ("var", name)


def c(value) -> tuple:
    return ("const", value)


def h(name: str, item=None) -> tuple:
    return ("?", name) if item is None else ("?", name, item)


SHARE = v("share")
HALF = ("div", SHARE, c(2))
OTHER_HALF = ("minus", SHARE, HALF)
TARGET, DOMAIN = ("target",), ("domain",)


def _right(program: tuple) -> tuple:
    return ("eq", ("eval", program), TARGET)


def _residual(name: str, experiment: tuple, entry: str, minus: bool) -> Plan:
    asked = ("vsub", ("eval", v("X")), TARGET) if minus else ("vsub", TARGET, ("eval", v("X")))
    rebuild = sub(h("X"), h(entry)) if minus else add(h("X"), h(entry))
    return Plan(name, experiment, (Entry(entry, ("question", asked, DOMAIN), SHARE),), rebuild)


def _misses_then_cases(name: str, experiment: tuple, rest: str, choice: str) -> Plan:
    return Plan(name, experiment, (
        Entry(rest, ("question", TARGET, ("not", _right(v("X")))), HALF),
        Entry(choice, ("question", ("bool", _right(v("X"))),
                       ("neq", _right(v("X")), _right(v(rest)))), OTHER_HALF),
    ), if_(h(choice), h("X"), h(rest)))


#: D1b's branches, in D0's table order.
D1B = (
    _residual("остаток", ("own",), "остаток", minus=False),
    _residual("остаток-", ("own",), "остаток-", minus=True),
    _misses_then_cases("промахи и случаи", ("own",), "промахи", "случаи"),
)

_FEW = c(16)
_PROBE = c(2000)

#: A: X after which the rest is simplest -- of the first states held.
A_REST = _residual(
    "A: остаток проще",
    ("argmin", ("take", _FEW, ("held",)),
     ("lambda", "x", ("plus", ("bits", v("x")),
                      ("probe", ("vsub", TARGET, ("eval", v("x"))), DOMAIN, _PROBE)))),
    "A остаток", minus=False)

#: A: X right on the most points -- of the first states held.
A_COVER = _misses_then_cases(
    "A: точен на большей части",
    ("argmax", ("take", _FEW, ("held",)), ("lambda", "x", ("count", _right(v("x"))))),
    "A промахи", "A случаи")

#: A: a condition after which both sides are simplest -- of the first conditions.
A_SPLIT = Plan(
    "A: развилка",
    ("argmin", ("take", _FEW, ("conditions",)),
     ("lambda", "k", ("plus", ("probe", TARGET, ("mask", ("eval", v("k"))), _PROBE),
                      ("probe", TARGET, ("not", ("mask", ("eval", v("k")))), _PROBE)))),
    (Entry("A да", ("question", TARGET, ("mask", ("eval", v("X")))), HALF),
     Entry("A нет", ("question", TARGET, ("not", ("mask", ("eval", v("X"))))), OTHER_HALF)),
    if_(h("X"), h("A да"), h("A нет")))

#: C: the points where the target equals the leaf that matches it most.
_LEAF = ("argmax", ("leaves",), ("lambda", "l", ("count", _right(v("l")))))
C_MASK = Plan(
    "C: где цель совпадает с листом",
    ("list", _LEAF, _right(_LEAF)),
    (Entry("C вне", ("question", TARGET, ("not", ("item", v("X"), c(1)))), HALF),
     Entry("C где", ("question", ("bool", ("item", v("X"), c(1))),
                     ("neq", ("item", v("X"), c(1)), _right(v("C вне")))), OTHER_HALF)),
    if_(h("C где"), h("X", 0), h("C вне")))

DIAGNOSTIC = (A_REST, A_COVER, A_SPLIT, C_MASK)


def p2_step(k: int) -> Plan:
    """One exploratory step of P2's class, then misses and cases from its X."""
    return _misses_then_cases(
        f"P2: шаг на {k}",
        ("argmin", ("flatmap", ("take", c(k), ("held",)),
                    ("lambda", "s", ("successors", v("s"), v("ahead")))),
         ("lambda", "y", ("bits", v("y")))),
        "P2 промахи", "P2 случаи")
