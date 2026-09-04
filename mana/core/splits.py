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
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import oracle
from .tasks import DOMAINS, Task, generate, generate_mixed

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.6"

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


@dataclass(frozen=True)
class Holdout:
    """A named, frozen version of the hidden set.

    Versioned rather than extended because a number is only meaningful
    together with what it was measured on. Adding two domains to the
    existing set would retroactively claim that every past hidden score
    had covered five domains, which is false. V0 stays exactly as it was;
    V1 is a different set, and results carry which one they came from.
    """
    name: str
    domains: Tuple[str, ...]
    seed: int
    surface: str

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "domains": list(self.domains),
                "surface": self.surface}


#: The historical holdout. Frozen: every hidden score recorded before
#: phase 18 was measured against exactly this, and it must keep meaning
#: what it meant.
HOLDOUT_V0 = Holdout("v0", DEVELOPMENT_DOMAINS, 3003, "canonical")

#: The full holdout. Five domains, and -- more importantly -- variant
#: surfaces. A holdout drawn from the same generator AND the same wording
#: as the training tasks is not independent of a solver written against
#: that generator: an algorithmic brain matching one template scores
#: perfectly on hidden instances of the same template and the number says
#: nothing. Measured: of four solvers, three scored zero once the wording
#: changed, and the fourth was unaffected because it parses structure.
HOLDOUT_V1 = Holdout("v1", ("arithmetic", "sequence", "code", "logic", "text_ops"),
                     5005, "variant")

HOLDOUTS = {"v0": HOLDOUT_V0, "v1": HOLDOUT_V1}

EVALUATED = "EVALUATED"
NOT_EVALUATED = "NOT_EVALUATED"
INSUFFICIENT_POWER = "INSUFFICIENT_POWER"
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
    #: Which holdout version produced this. A hidden score without it is
    #: a number whose meaning depends on when it was taken.
    holdout: str = "v0"
    #: Correct out of everything ATTEMPTED, counting an ungradable answer
    #: as not correct. `accuracy` deliberately excludes ungradable ones so
    #: that a missing sandbox does not read as a capability deficit -- but
    #: that makes refusing free, and a brain with narrow applicability
    #: then gets scored on fewer tasks than the model it is compared
    #: against. Two numbers over different denominators are not a
    #: comparison. Both are reported; comparisons should use this one.
    strict_accuracy: float = 0.0
    strict_by_domain: Dict[str, float] = field(default_factory=dict)
    attempted: int = 0
    correct: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"holdout": self.holdout,
                "accuracy": self.accuracy, "strict_accuracy": self.strict_accuracy,
                "graded": self.graded, "attempted": self.attempted,
                "correct": self.correct,
                "ungradable": self.ungradable, "format_failures": self.format_failures,
                "by_domain": dict(self.by_domain),
                "strict_by_domain": dict(self.strict_by_domain),
                "evaluations_used": self.evaluations_used,
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

    attempted = summary["graded"] + summary["ungradable"]
    strict_by_domain = {}
    for domain, values in summary["by_domain"].items():
        tried = values["total"] + values["ungradable"]
        strict_by_domain[domain] = values["correct"] / tried if tried else 0.0

    return HiddenResult(
        accuracy=summary["accuracy"], graded=summary["graded"],
        ungradable=summary["ungradable"], format_failures=summary["format_failures"],
        by_domain={d: v["accuracy"] for d, v in summary["by_domain"].items()},
        strict_accuracy=summary["correct"] / attempted if attempted else 0.0,
        strict_by_domain=strict_by_domain,
        attempted=attempted, correct=summary["correct"],
        evaluations_used=used, evaluations_left=max(0, budget - used), elapsed=elapsed,
    )


def hidden_score(answer_fn: Callable[[Dict[str, Any]], str], *, verifier: Any = None,
                 per_domain: int = 25, label: str = "",
                 budget: int = DEFAULT_HIDDEN_BUDGET,
                 holdout: Holdout = HOLDOUT_V0,
                 per_domain_counts: Optional[Dict[str, int]] = None) -> HiddenResult:
    """Evaluate against the hidden holdout.

    `answer_fn` receives one public task dict and returns a string. It
    never sees a `Task`, so it never sees an answer, and the caller never
    receives the tasks either -- only the aggregate.

    This is the single function in MANA permitted to touch the hidden set,
    and it is the reason there is no `hidden_tasks()` to import.
    """
    # Per-domain counts, because one number for every domain is the wrong
    # shape: how many trials a domain needs follows from the smallest
    # effect that domain must be able to show, and those differ. The
    # significance bar does not move -- only the sample size does.
    tasks: List[Task] = []
    for domain in holdout.domains:
        count = (per_domain_counts or {}).get(domain, per_domain)
        if count <= 0:
            continue
        tasks.extend(generate(domain, count, holdout.seed, surface=holdout.surface))
    result = _run_hidden(tasks, answer_fn, verifier,
                         label or f"hidden-{holdout.name}", budget)
    return replace(result, holdout=holdout.name)


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
