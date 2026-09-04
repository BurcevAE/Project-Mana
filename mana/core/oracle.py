"""
mana.core.oracle — grading, with no model in the loop.

Every function here answers "is this answer correct?" by computing, not by
asking. That is what makes the score trustworthy, and it is also what
makes an experiment budget of thousands of trials affordable: a grader
that costs an LLM call doubles the price of every measurement and puts a
model's opinion inside the definition of success.

The grader is strict on purpose
-------------------------------
Every prompt states its output format ("одним числом, без пояснений"), and
an answer that ignores the format is wrong. A lenient extractor -- pull
the last number out of any prose -- would reintroduce the fuzziness this
module exists to remove, and would silently reward a model that produced
three candidate numbers and let the grader pick the flattering one.

There is exactly one concession, and it is bounded: leading and trailing
whitespace, surrounding quotes, and a single trailing period are ignored,
because those are formatting noise rather than a different answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .tasks import Task

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_TRIM = " \t\n\r\"'«»`.,;:"


@dataclass(frozen=True)
class Grade:
    """The verdict, plus why. `reason` exists so a failure is diagnosable
    without re-running: "format" and "wrong" are different problems and
    lead to different fixes."""
    correct: bool
    reason: str = ""
    expected: Any = None
    got: Any = None

    def as_dict(self) -> Dict[str, Any]:
        return {"correct": self.correct, "reason": self.reason,
                "expected": self.expected, "got": self.got}


def _clean(text: str) -> str:
    return (text or "").strip().strip(_TRIM).strip()


def _grade_number(task: Task, response: str) -> Grade:
    raw = _clean(response).replace(" ", "").replace(" ", "").replace(",", ".")
    if not _NUM_RE.match(raw):
        return Grade(False, "format", task.answer, _clean(response)[:80])
    try:
        got: Any = int(raw) if "." not in raw else float(raw)
    except ValueError:
        return Grade(False, "format", task.answer, raw[:80])
    expected = task.answer
    if isinstance(expected, int) and isinstance(got, float) and got.is_integer():
        got = int(got)          # 391.0 is the same answer as 391
    if isinstance(expected, float) or isinstance(got, float):
        ok = abs(float(got) - float(expected)) < 1e-9
    else:
        ok = got == expected
    return Grade(ok, "" if ok else "wrong", expected, got)


def _grade_text(task: Task, response: str) -> Grade:
    got = _clean(response).lower()
    expected = str(task.answer).strip().lower()
    # A single word was asked for; a sentence containing it is a different
    # thing and is graded as a format failure, not as correct-with-noise.
    if len(got.split()) > 1:
        return Grade(False, "format", task.answer, _clean(response)[:80])
    return Grade(got == expected, "" if got == expected else "wrong", task.answer, got)


def _grade_sequence(task: Task, response: str) -> Grade:
    parts = [p.strip().strip(_TRIM).lower() for p in re.split(r"[,\n;]+", _clean(response)) if p.strip()]
    expected = [str(x).strip().lower() for x in task.answer]
    ok = parts == expected
    return Grade(ok, "" if ok else "wrong", expected, parts)


def _grade_set(task: Task, response: str) -> Grade:
    parts = {p.strip().strip(_TRIM).lower() for p in re.split(r"[,\n;]+", _clean(response)) if p.strip()}
    expected = {str(x).strip().lower() for x in task.answer}
    ok = parts == expected
    return Grade(ok, "" if ok else "wrong", sorted(expected), sorted(parts))


def _grade_code_tests(task: Task, response: str, verifier: Any = None) -> Grade:
    """Run the hidden cases against the submitted function.

    Needs a LocalVerifier with execution enabled. Without one the answer
    is ungradable and must be reported as such -- `correct=False,
    reason="ungradable"` -- rather than silently counted as a failure,
    which would make a machine without the sandbox look like a machine
    where the agent is bad at code.
    """
    if verifier is None:
        return Grade(False, "ungradable", "code_tests", "no sandbox available")
    code = _strip_fences(response)
    if not code.strip():
        return Grade(False, "format", "code_tests", "empty")
    lines = [code, ""]
    for i, (call, expected) in enumerate(task.answer):
        lines.append(f"__r{i} = {call}")
        lines.append(f"assert __r{i} == {expected!r}, 'case {i}: ' + repr(__r{i})")
    lines.append("print('ALL_OK')")
    result = verifier.verify_code("\n".join(lines))
    if result.get("policy_blocked"):
        return Grade(False, "blocked", "code_tests", result.get("error", "")[:120])
    if result.get("sandbox_missing"):
        return Grade(False, "ungradable", "code_tests", result.get("error", "")[:120])
    ok = bool(result.get("ok")) and "ALL_OK" in (result.get("stdout") or "")
    return Grade(ok, "" if ok else "wrong", "code_tests",
                 (result.get("stderr") or result.get("error") or "")[:160])


def _strip_fences(text: str) -> str:
    """Models wrap code in ``` no matter how the prompt asks them not to.
    Unwrapping is formatting noise, not leniency about the answer."""
    body = (text or "").strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    return body.strip()


def grade(task: Task, response: str, verifier: Any = None) -> Grade:
    """Grade one response. Never raises: a grader that can crash would
    take down the experiment that called it, and an ungradable answer is
    information, not an error."""
    try:
        if task.checker == "number":
            return _grade_number(task, response)
        if task.checker == "text":
            return _grade_text(task, response)
        if task.checker == "sequence":
            return _grade_sequence(task, response)
        if task.checker == "set":
            return _grade_set(task, response)
        if task.checker == "code_tests":
            return _grade_code_tests(task, response, verifier)
        return Grade(False, "unknown_checker", task.checker, None)
    except Exception as exc:
        return Grade(False, "grader_error", task.answer, f"{type(exc).__name__}: {exc}")


def score(tasks: Sequence[Task], responses: Sequence[str], verifier: Any = None) -> Dict[str, Any]:
    """Aggregate over a task set.

    `ungradable` is counted apart from `wrong` and excluded from the
    accuracy denominator. Mixing them would let a missing sandbox read as
    a capability deficit -- a measurement error disguised as a finding,
    which is the specific failure this whole layer exists to prevent.
    """
    grades = [grade(t, r, verifier) for t, r in zip(tasks, responses)]
    gradable = [g for g in grades if g.reason != "ungradable"]
    correct = sum(1 for g in gradable if g.correct)
    by_domain: Dict[str, Dict[str, int]] = {}
    for task, g in zip(tasks, grades):
        bucket = by_domain.setdefault(task.domain, {"correct": 0, "total": 0, "ungradable": 0})
        if g.reason == "ungradable":
            bucket["ungradable"] += 1
            continue
        bucket["total"] += 1
        bucket["correct"] += int(g.correct)
    return {
        "accuracy": correct / len(gradable) if gradable else 0.0,
        "correct": correct,
        "graded": len(gradable),
        "ungradable": len(grades) - len(gradable),
        "format_failures": sum(1 for g in grades if g.reason == "format"),
        "by_domain": {d: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
                      for d, v in by_domain.items()},
        "grades": [g.as_dict() for g in grades],
    }
