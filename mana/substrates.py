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
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import cost as cost_units

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.7"


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

#: Runs of comma-separated integers, wherever they appear. The previous
#: version required the literal word "последовательность:" and scored
#: zero the moment the holdout said "Дан ряд чисел" instead -- a solver
#: matched to one sentence, not to the problem.
_NUMBER_RUN = re.compile(r"-?\d+(?:\s*,\s*-?\d+){3,}")


def _differences(values):
    return [b - a for a, b in zip(values, values[1:])]


def _next_term(values):
    """The next term, if a rule can be PROVED from the terms given."""
    first = _differences(values)
    if len(set(first)) == 1:
        return values[-1] + first[-1]

    if all(v != 0 for v in values[:-1]):
        ratios = [b / a for a, b in zip(values, values[1:])]
        if len(set(round(r, 9) for r in ratios)) == 1 and float(ratios[0]).is_integer():
            return int(values[-1] * ratios[0])

    second = _differences(first)
    if len(set(second)) == 1:
        return values[-1] + first[-1] + second[-1]

    # a(n) = x*a(n-1) + y, solved from two pairs and then CHECKED against
    # every remaining term. Fitting two points and trusting the result is
    # how a solver invents a rule.
    if len(values) >= 4 and values[1] != values[0]:
        denominator = values[1] - values[0]
        if (values[2] - values[1]) % denominator == 0:
            x = (values[2] - values[1]) // denominator
            y = values[1] - x * values[0]
            if all(values[i + 1] == x * values[i] + y for i in range(len(values) - 1)):
                return x * values[-1] + y
    return None


def sequence_answer(prompt: str, verifier: Any = None) -> str:
    """Continue the sequence exactly, or refuse.

    Finds the numbers by shape rather than by the sentence around them,
    so a rewording does not silence it. Still refuses whenever the rule
    cannot be proved: a sequence solver that guesses returns a number
    with the same confidence whether it found the rule or invented one.
    """
    runs = _NUMBER_RUN.findall(prompt or "")
    if not runs:
        raise BrainRefusal("в задаче нет ряда чисел")
    longest = max(runs, key=lambda r: r.count(","))
    try:
        values = [int(part.strip()) for part in longest.split(",")]
    except ValueError:                                  # pragma: no cover
        raise BrainRefusal("члены ряда не целые")
    if len(values) < 4:
        raise BrainRefusal(f"слишком мало членов ({len(values)}), правило недоказуемо")
    nxt = _next_term(values)
    if nxt is None:
        raise BrainRefusal("правило ряда не доказано")
    return str(nxt)


# ---------------------------------------------------------------------------
# text operations
# ---------------------------------------------------------------------------

#: Question types recognised by the WORDS present, not by a sentence
#: template. "Сколько раз буква «р»" and "Посчитай, сколько раз символ «р»"
#: are the same question; the old version knew only the first.
_COUNT_MARKERS = ("скольк", "посчита", "количеств")
_LETTER_MARKERS = ("букв", "символ")
_WORD_MARKERS = ("слов",)
_LONGEST_MARKERS = ("длинн", "наибольш")
_QUOTED_CHAR = re.compile(r"[«\"'‘“]\s*(\w)\s*[»\"'’”]")
#: A line of plain lowercase words is the text being asked about.
_WORD_LINE = re.compile(r"^[а-яёa-z\s]+$", re.M)


def _text_body(prompt: str) -> List[str]:
    """The words the question is about, found by shape.

    The longest line that is nothing but lowercase words. Labels differ
    between wordings ("Текст:", "Дана строка:") and questions are not
    all-lowercase because they contain punctuation and quotes.
    """
    best: List[str] = []
    for line in (prompt or "").splitlines():
        candidate = line.strip()
        if ":" in candidate:
            candidate = candidate.split(":", 1)[1].strip()
        if not candidate or not _WORD_LINE.fullmatch(candidate):
            continue
        words = candidate.split()
        if len(words) > len(best):
            best = words
    return best


def text_ops_answer(prompt: str, verifier: Any = None) -> str:
    """Count, measure or select over a given text -- exactly, or refuse."""
    lowered = (prompt or "").lower()
    words = _text_body(prompt)
    if not words:
        raise BrainRefusal("в задаче нет текста")

    asks_count = any(m in lowered for m in _COUNT_MARKERS)
    quoted = _QUOTED_CHAR.search(prompt or "")

    if asks_count and quoted and any(m in lowered for m in _LETTER_MARKERS):
        target = quoted.group(1).lower()
        return str(sum(w.lower().count(target) for w in words))

    if any(m in lowered for m in _LONGEST_MARKERS) and any(
            m in lowered for m in _WORD_MARKERS):
        longest = max(len(w) for w in words)
        candidates = [w for w in words if len(w) == longest]
        if len(candidates) > 1 and not ("перв" in lowered or "любо" in lowered):
            raise BrainRefusal("несколько самых длинных слов, правило выбора не задано")
        return candidates[0]

    if asks_count and any(m in lowered for m in _WORD_MARKERS):
        return str(len(words))

    raise BrainRefusal("вопрос о тексте не распознан")


# ---------------------------------------------------------------------------
# ordering logic
# ---------------------------------------------------------------------------

#: Words that place one name before another, and words that place it
#: after. The relation is what the solver looks for; the sentence around
#: it is not its business. The previous version matched the single phrase
#: "X стоит раньше, чем Y" and went silent on "Y стоит позже, чем X" --
#: the same constraint with the operands swapped.
_BEFORE_WORDS = ("раньше", "перед", "до ", "сначала", "прежде")
_AFTER_WORDS = ("позже", "после", "затем", "потом")
_NAME = re.compile(r"\b([А-ЯЁ][а-яё]+)\b")
_POSITION_MARKERS = ("позици", "мест", "номер")


def logic_answer(prompt: str, verifier: Any = None) -> str:
    """Resolve an ordering from constraints, however they are worded.

    Answers only when the constraints determine a single order. A
    topological sort with more than one valid ordering has no single
    right answer, and returning one of them would be a coin toss wearing
    a proof's clothes.
    """
    text = prompt or ""
    pairs: List[Tuple[str, str]] = []
    for line in text.splitlines():
        names = _NAME.findall(line)
        if len(names) != 2:
            continue
        lowered = line.lower()
        before = any(w in lowered for w in _BEFORE_WORDS)
        after = any(w in lowered for w in _AFTER_WORDS)
        if before == after:
            continue                    # neither, or contradictory
        pairs.append((names[0], names[1]) if before else (names[1], names[0]))
    if not pairs:
        raise BrainRefusal("в задаче нет ограничений порядка")

    position = None
    for line in text.splitlines():
        if any(m in line.lower() for m in _POSITION_MARKERS):
            found = re.search(r"(\d+)", line)
            if found:
                position = int(found.group(1))
                break
    if position is None:
        raise BrainRefusal("не сказано, какая позиция нужна")

    names: List[str] = []
    for before_name, after_name in pairs:
        for name in (before_name, after_name):
            if name not in names:
                names.append(name)
    incoming = {name: 0 for name in names}
    edges: Dict[str, List[str]] = {name: [] for name in names}
    for before_name, after_name in pairs:
        if after_name in edges[before_name]:
            continue                    # a repeated constraint is not a new one
        edges[before_name].append(after_name)
        incoming[after_name] += 1

    order: List[str] = []
    available = [n for n in names if incoming[n] == 0]
    while available:
        if len(available) > 1:
            raise BrainRefusal("порядок не определён однозначно")
        current = available.pop()
        order.append(current)
        for nxt in edges[current]:
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                available.append(nxt)
    if len(order) != len(names):
        raise BrainRefusal("в ограничениях цикл")

    index = position - 1
    if not 0 <= index < len(order):
        raise BrainRefusal(f"позиции {position} нет среди {len(order)} имён")
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
