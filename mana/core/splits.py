"""
mana.core.splits — TRAIN / DEV / HIDDEN / TRANSFER, and why HIDDEN is not
a list you can get.

The problem this solves
-----------------------
MANA's old holdout was `BenchmarkSuite.holdout_tasks()` -- a function in
the same file as the training tasks, callable by anything, returning the
tasks themselves. Everything downstream could read it, and
`code_evolution` did: its regression oracle was train + generalization +
holdout, all three. A "held out" set that the optimiser can read is a
training set with a different name.

The design here is one idea: **the hidden set is never returned as data.**
There is no `hidden_tasks()`. The only way to interact with it is
`hidden_score(answer_fn)` -- you hand in a function, it hands back
aggregate numbers. The tasks exist only inside that call. An agent cannot
overfit to a set it has never seen, and no amount of care by the caller is
required for that to hold.

The splits
----------
  TRAIN     visible; use freely for search and tuning
  DEV       visible; for choosing between candidates during development
  HIDDEN    never visible; the only score an acceptance gate may trust
  TRANSFER  never visible; different domains entirely, for testing whether
            something found in one place works in another

TRANSFER is separate from HIDDEN because they answer different questions.
HIDDEN asks "does this generalize beyond what you tuned on?"; TRANSFER
asks "does this work where you have never been?" A mechanism can pass the
first and fail the second, and that difference is the point of the
project.

Budget, not honour system
-------------------------
Every hidden evaluation is counted and capped. Unlimited access to a
hidden set is the same leak as reading it, only slower: enough queries and
the optimiser fits the set through the score alone. `remaining_budget()`
reports what is left; exceeding it raises rather than degrading quietly.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import oracle
from .tasks import DOMAINS, Task, generate_mixed

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

#: Seeds are constants inside the immutable core, not configuration. A
#: configurable hidden seed is a hidden set an agent can re-derive.
_SEED_TRAIN = 1001
_SEED_DEV = 2002
_SEED_HIDDEN = 3003
_SEED_TRANSFER = 4004

#: Domains a strategy is developed on, and the ones kept back to test
#: whether it travels. The split is by domain, not by sampling, because
#: two samples of the same distribution measure generalization, not
#: transfer.
DEVELOPMENT_DOMAINS = ("arithmetic", "sequence", "code")
TRANSFER_DOMAINS = ("logic", "text_ops")

#: How many hidden evaluations are allowed per process. Deliberately small:
#: the hidden set answers "may this be accepted", which is a question that
#: should be asked a handful of times per session, not in a loop.
DEFAULT_HIDDEN_BUDGET = 50

_lock = threading.RLock()
_hidden_calls = 0
_hidden_log: List[Dict[str, Any]] = []


def train_tasks(per_domain: int = 40) -> List[Task]:
    return generate_mixed(per_domain, _SEED_TRAIN, DEVELOPMENT_DOMAINS)


def dev_tasks(per_domain: int = 20) -> List[Task]:
    return generate_mixed(per_domain, _SEED_DEV, DEVELOPMENT_DOMAINS)


def public_train(per_domain: int = 40) -> List[Dict[str, Any]]:
    """Training tasks as an agent may see them -- prompts, no answers.

    Even the visible split hands out `public()` rather than `Task`: an
    answer that is merely inconvenient to reach is an answer that will
    eventually be reached.
    """
    return [t.public() for t in train_tasks(per_domain)]


@dataclass(frozen=True)
class HiddenResult:
    """What a hidden evaluation returns. Scores and shapes of failure --
    never a task, never an answer, never a per-task record that could be
    reassembled into the set."""
    accuracy: float
    graded: int
    ungradable: int
    format_failures: int
    by_domain: Dict[str, float]
    evaluations_used: int
    evaluations_left: int
    elapsed: float

    def as_dict(self) -> Dict[str, Any]:
        return {"accuracy": self.accuracy, "graded": self.graded,
                "ungradable": self.ungradable, "format_failures": self.format_failures,
                "by_domain": dict(self.by_domain), "evaluations_used": self.evaluations_used,
                "evaluations_left": self.evaluations_left, "elapsed": round(self.elapsed, 2)}


class HiddenBudgetExceeded(RuntimeError):
    """Raised rather than silently degrading. A gate that quietly stops
    measuring is worse than one that stops working, because the run
    continues and the numbers keep being believed."""


def _run_hidden(tasks: Sequence[Task], answer_fn: Callable[[Dict[str, Any]], str],
                verifier: Any, label: str, budget: int) -> HiddenResult:
    global _hidden_calls
    with _lock:
        if _hidden_calls >= budget:
            raise HiddenBudgetExceeded(
                f"hidden evaluation budget exhausted ({_hidden_calls}/{budget}). "
                "Unlimited queries against a hidden set fit it through the score alone.")
        _hidden_calls += 1
        used = _hidden_calls

    started = time.perf_counter()
    responses: List[str] = []
    for task in tasks:
        try:
            responses.append(str(answer_fn(task.public())))
        except Exception as exc:
            # A caller that crashes on one task must not lose the whole
            # evaluation -- and an exception is a wrong answer, not an
            # ungradable one.
            responses.append(f"<error: {type(exc).__name__}: {exc}>")
    summary = oracle.score(tasks, responses, verifier)
    elapsed = time.perf_counter() - started

    with _lock:
        _hidden_log.append({"label": label, "at": time.time(), "accuracy": summary["accuracy"],
                            "graded": summary["graded"], "elapsed": elapsed})

    return HiddenResult(
        accuracy=summary["accuracy"], graded=summary["graded"],
        ungradable=summary["ungradable"], format_failures=summary["format_failures"],
        by_domain={d: v["accuracy"] for d, v in summary["by_domain"].items()},
        evaluations_used=used, evaluations_left=max(0, budget - used), elapsed=elapsed,
    )


def hidden_score(answer_fn: Callable[[Dict[str, Any]], str], *, verifier: Any = None,
                 per_domain: int = 25, label: str = "",
                 budget: int = DEFAULT_HIDDEN_BUDGET) -> HiddenResult:
    """Evaluate against the hidden holdout.

    `answer_fn` receives one public task dict and returns a string. It
    never sees a `Task`, so it never sees an answer, and the caller never
    receives the tasks either -- only the aggregate.

    This is the single function in MANA permitted to touch the hidden set,
    and it is the reason there is no `hidden_tasks()` to import.
    """
    tasks = generate_mixed(per_domain, _SEED_HIDDEN, DEVELOPMENT_DOMAINS)
    return _run_hidden(tasks, answer_fn, verifier, label or "hidden", budget)


def transfer_score(answer_fn: Callable[[Dict[str, Any]], str], *, verifier: Any = None,
                   per_domain: int = 25, label: str = "",
                   budget: int = DEFAULT_HIDDEN_BUDGET) -> HiddenResult:
    """Evaluate on domains the development splits never contained.

    Separate from `hidden_score` because it answers a different question.
    Passing hidden means "you did not merely fit the training sample".
    Passing transfer means "this works somewhere you have never been" --
    the claim the whole project rests on, and the one a paraphrased
    holdout cannot support.
    """
    tasks = generate_mixed(per_domain, _SEED_TRANSFER, TRANSFER_DOMAINS)
    return _run_hidden(tasks, answer_fn, verifier, label or "transfer", budget)


def remaining_budget(budget: int = DEFAULT_HIDDEN_BUDGET) -> int:
    with _lock:
        return max(0, budget - _hidden_calls)


def audit_log() -> List[Dict[str, Any]]:
    """Every hidden/transfer evaluation performed in this process.

    Scores and timings only. Kept so that "how many times was the holdout
    consulted before this was accepted?" is answerable after the fact --
    a number that matters as much as the score itself.
    """
    with _lock:
        return list(_hidden_log)


def _reset_for_tests() -> None:
    """Clear the counter. Named for what it is: tests need a fresh budget,
    and a general-purpose `reset()` would be an obvious way around the cap."""
    global _hidden_calls
    with _lock:
        _hidden_calls = 0
        _hidden_log.clear()
