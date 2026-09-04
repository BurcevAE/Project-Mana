"""
mana.substrates — brains that are not language models.

The audit for MANA 3.0 found that the cheap substrate was already built
and living in the wrong place: `verify_arithmetic` computes arithmetic
exactly through an AST with no model involved, and `run_code` executes
Python in a sandbox. Both sat in `tools`, a hierarchy the brain router
cannot see, so the cheapest possible answer to "what is 4821 * 37 + 145"
was unreachable from the thing whose job is choosing how to answer.

This module does not build an algorithmic brain. It makes the one that
exists addressable, by giving the pool a way to call something other
than an HTTP endpoint.

Refusal is not failure
----------------------
A language model answers everything, wrongly if necessary. An
algorithmic brain must decline what it cannot do exactly -- and that
decline has to be a different outcome from a malfunction, or the router
will mark a correctly-behaving brain unhealthy for doing its job. Hence
`BrainRefusal`: not an error, a statement of applicability.

The bar this raises
-------------------
An algorithmic brain that guesses is worse than useless: it is a wrong
answer with no uncertainty attached, and downstream nothing will doubt
it. So each handler here answers only where it can be exact, and where
it cannot, it says so.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Optional

from .core import cost as cost_units

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.5"


class BrainRefusal(Exception):
    """This task is outside what this brain can answer exactly.

    Distinct from every other exception a brain can raise. A refusal
    costs almost nothing, damages no reputation, and simply means the
    cascade should try the next substrate.
    """


#: A handler answers a prompt exactly, or refuses. It never approximates.
Handler = Callable[[str], str]

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------

def arithmetic_answer(prompt: str, verifier: Any = None) -> str:
    """Evaluate the arithmetic in a prompt exactly, or refuse.

    Uses the expression extraction and safe evaluator that already exist
    in `LocalVerifier`, rather than writing a second arithmetic parser.
    Two parsers for one job disagree eventually, and the day they do,
    the verifier and the brain contradict each other about the same
    expression -- which is precisely the situation the verifier exists
    to prevent.
    """
    from .verifier import LocalVerifier

    expression = LocalVerifier.extract_expression(prompt)
    if not expression:
        raise BrainRefusal("в задаче нет арифметического выражения")

    if verifier is None:
        result = LocalVerifier.evaluate_expression(expression)
    else:
        result = verifier.verify_expression(expression)
    if not result.get("ok"):
        # Parsed as arithmetic and did not evaluate -- overflow, division
        # by zero, something outside the safe node set. Refusing is right:
        # the brain has nothing exact to say, and the next substrate may.
        raise BrainRefusal(f"выражение не вычисляется: {result.get('error', '?')}")
    return str(result["value"])


# ---------------------------------------------------------------------------
# code execution
# ---------------------------------------------------------------------------

def code_output(prompt: str, verifier: Any = None) -> str:
    """Run the Python in the prompt and return what it printed, or refuse.

    Answers exactly one question -- "what does this code output?" -- and
    refuses everything else, including "write code that does X", which is
    a generation task and belongs to a model.
    """
    from .verifier import LocalVerifier

    match = _CODE_BLOCK.search(prompt or "")
    if not match:
        raise BrainRefusal("в задаче нет блока кода")
    code = match.group(1).strip()
    if not code:
        raise BrainRefusal("блок кода пуст")
    if verifier is None:
        raise BrainRefusal("песочница недоступна")
    result = verifier.verify_code(code)
    if not result.get("ok"):
        raise BrainRefusal(f"код не выполнился: {result.get('error', '?')}")
    output = (result.get("stdout") or "").strip()
    if not output:
        raise BrainRefusal("код ничего не вывел")
    return output


# ---------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------

_SEQUENCE = re.compile(r"последовательность\s*:\s*([0-9,\s\-]+)")


def _differences(values):
    return [b - a for a, b in zip(values, values[1:])]


def sequence_answer(prompt: str, verifier: Any = None) -> str:
    """Continue the sequence exactly, or refuse.

    Tries the shapes that can be *proved* from the terms given -- constant
    differences, constant ratios, constant second differences, and a
    first-order linear recurrence -- and refuses everything else. A
    sequence solver that guesses is the worst kind of brain: it returns a
    number with the same confidence whether it found the rule or invented
    one.
    """
    match = _SEQUENCE.search(prompt or "")
    if not match:
        raise BrainRefusal("в задаче нет последовательности")
    try:
        values = [int(part) for part in match.group(1).replace(" ", "").split(",")
                  if part not in ("", "-")]
    except ValueError:
        raise BrainRefusal("члены последовательности не целые")
    if len(values) < 4:
        raise BrainRefusal(f"слишком мало членов ({len(values)}), правило недоказуемо")

    first = _differences(values)
    if len(set(first)) == 1:
        return str(values[-1] + first[-1])

    if all(v != 0 for v in values[:-1]):
        ratios = [b / a for a, b in zip(values, values[1:])]
        if len(set(round(r, 9) for r in ratios)) == 1 and float(ratios[0]).is_integer():
            return str(int(values[-1] * ratios[0]))

    second = _differences(first)
    if len(set(second)) == 1:
        return str(values[-1] + first[-1] + second[-1])

    # a(n) = x*a(n-1) + y, solved from two consecutive pairs and then
    # CHECKED against every remaining term. Fitting two points and
    # trusting the result is how a solver invents a rule.
    if len(values) >= 4 and values[1] != values[0]:
        denominator = values[1] - values[0]
        if (values[2] - values[1]) % denominator == 0:
            x = (values[2] - values[1]) // denominator
            y = values[1] - x * values[0]
            if all(values[i + 1] == x * values[i] + y for i in range(len(values) - 1)):
                return str(x * values[-1] + y)

    raise BrainRefusal("правило последовательности не доказано")


# ---------------------------------------------------------------------------
# text operations
# ---------------------------------------------------------------------------

_TEXT_BODY = re.compile(r"текст\s*:\s*(.+?)\n", re.I | re.S)
_LETTER_COUNT = re.compile(r"сколько\s+раз\s+буква\s*[«\"']?(\w)[»\"']?", re.I)
_LONGEST_WORD = re.compile(r"какое\s+слово.*самое\s+длинное", re.I | re.S)
_WORD_COUNT = re.compile(r"сколько\s+(?:в нём\s+)?слов", re.I)


def text_ops_answer(prompt: str, verifier: Any = None) -> str:
    """Count, measure or select over a given text -- exactly, or refuse.

    Every question here has one right answer computable from the text.
    The tie-breaking rule for the longest word is read out of the prompt
    rather than assumed, because "если таких несколько" means the
    generator has a rule and a brain that picks a different one is
    confidently wrong.
    """
    body_match = _TEXT_BODY.search(prompt or "")
    if not body_match:
        raise BrainRefusal("в задаче нет текста")
    words = body_match.group(1).split()
    if not words:
        raise BrainRefusal("текст пуст")

    letter = _LETTER_COUNT.search(prompt)
    if letter:
        target = letter.group(1).lower()
        return str(sum(w.lower().count(target) for w in words))

    if _LONGEST_WORD.search(prompt):
        longest = max(len(w) for w in words)
        candidates = [w for w in words if len(w) == longest]
        if len(candidates) > 1 and "перв" not in (prompt or "").lower():
            raise BrainRefusal("несколько самых длинных слов, правило выбора не задано")
        return candidates[0]

    if _WORD_COUNT.search(prompt):
        return str(len(words))

    raise BrainRefusal("вопрос о тексте не распознан")


# ---------------------------------------------------------------------------
# ordering logic
# ---------------------------------------------------------------------------

_EARLIER = re.compile(r"[-•]\s*(\w+)\s+стоит\s+раньше,?\s+чем\s+(\w+)", re.I)
_POSITION = re.compile(r"на\s+позиции\s+(\d+)", re.I)


def logic_answer(prompt: str, verifier: Any = None) -> str:
    """Resolve an ordering from "before" constraints, or refuse.

    Answers only when the constraints determine a single order. A
    topological sort with more than one valid ordering has no single
    right answer, and returning one of them would be a coin toss wearing
    a proof's clothes.
    """
    pairs = _EARLIER.findall(prompt or "")
    if not pairs:
        raise BrainRefusal("в задаче нет ограничений порядка")
    position = _POSITION.search(prompt or "")
    if not position:
        raise BrainRefusal("не сказано, какая позиция нужна")

    names = []
    for before, after in pairs:
        for name in (before, after):
            if name not in names:
                names.append(name)
    after_count = {name: 0 for name in names}
    edges = {name: [] for name in names}
    for before, after in pairs:
        edges[before].append(after)
        after_count[after] += 1

    order = []
    available = [n for n in names if after_count[n] == 0]
    while available:
        if len(available) > 1:
            raise BrainRefusal("порядок не определён однозначно")
        current = available.pop()
        order.append(current)
        for nxt in edges[current]:
            after_count[nxt] -= 1
            if after_count[nxt] == 0:
                available.append(nxt)
    if len(order) != len(names):
        raise BrainRefusal("в ограничениях цикл")

    index = int(position.group(1)) - 1
    if not 0 <= index < len(order):
        raise BrainRefusal(f"позиции {index + 1} нет среди {len(order)} имён")
    return order[index]


#: Every algorithmic brain the pool can address, by brain id. A new one
#: is a line here plus a catalog entry -- deliberately, so that adding a
#: substrate is visible rather than discovered by scanning.
HANDLERS: Dict[str, Callable[..., str]] = {
    "arithmetic": arithmetic_answer,
    "code-exec": code_output,
    "sequence-solver": sequence_answer,
    "text-ops": text_ops_answer,
    "order-logic": logic_answer,
}


def call(brain_id: str, prompt: str, verifier: Any = None) -> str:
    handler = HANDLERS.get(brain_id)
    if handler is None:
        raise BrainRefusal(f"нет обработчика для {brain_id}")
    return handler(prompt, verifier)


def cost_of(elapsed: float) -> Dict[str, Any]:
    """What an algorithmic call cost.

    Tokens are genuinely zero, not unmeasured: no tokens exist. Memory
    is left unmeasured rather than reported as a delta -- around a
    computation this short the allocator noise is larger than the signal,
    and a number that is mostly noise reads as precision that is not
    there. Phase 17 will have a substrate where it is worth measuring.
    """
    return cost_units.one_call(
        substrate=cost_units.ALGORITHMIC, wall_seconds=elapsed,
        tokens_in=0, tokens_out=0).as_dict()
