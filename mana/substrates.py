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
__version__ = "2.0"


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


#: Every algorithmic brain the pool can address, by brain id. A new one
#: is a line here plus a catalog entry -- deliberately, so that adding a
#: substrate is visible rather than discovered by scanning.
HANDLERS: Dict[str, Callable[..., str]] = {
    "arithmetic": arithmetic_answer,
    "code-exec": code_output,
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
