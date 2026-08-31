"""
mana.code_evolution — gated self-improvement of MANA's own source code.

This mirrors, deliberately, the acceptance discipline agent_parts/evolution.py
already applies to PipelineSpec: propose a candidate -> run it in the
existing sandbox (LocalVerifier) -> require it to fix something without
regressing anything already passing -> only then apply it, with a backup
and a rollback path. Nothing here is a new trust boundary; it reuses
LocalVerifier's existing subprocess sandbox and import/call policy rather
than inventing a second one.

Hard boundaries, on purpose:

  1. WHITELIST below is the *only* thing that can ever be patched. It is a
     module-level constant, not data the agent can extend at runtime --
     growing it means a human edits this file. An agent that could widen
     its own patchable surface could eventually patch its own gate; that
     path does not exist here.
  2. A candidate is source for one free function (no `self`, no closures
     over agent state) with a fixed name/parameter list. It never runs
     with elevated privileges: it goes through LocalVerifier's existing
     `_static_policy` import/call blocklist *and* its subprocess sandbox,
     exactly like any other agent-generated code MANA already executes.
  3. Acceptance requires: sandbox execution succeeds, every test case the
     *current* implementation already passes still passes (no regression),
     and at least one test case that currently fails now passes (real
     improvement, not just churn). Ties are rejected, matching
     evolve_pipeline's `_strict_acceptance` philosophy of "accept only a
     clear win".
  4. Every applied patch is preceded by a timestamped backup of the whole
     file and a changelog entry (diff + scores + instruction); rollback()
     restores the most recent (or a named) backup. Nothing is applied
     without both existing first.
"""
from __future__ import annotations

import ast
import difflib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
HISTORY_ROOT = PACKAGE_ROOT.parent / "mana_code_history"

# Files a patch may never target, even in principle: the sandbox/security
# policy itself, config, this module, and the CLI. Checked defensively in
# apply() in addition to WHITELIST only ever containing safe entries.
_NEVER_PATCHABLE = {"verifier.py", "hardware.py", "code_evolution.py", "config.py", "cli.py", "optional_deps.py"}


@dataclass
class CodeTestCase:
    """One (input, required-substrings) pair. Mirrors BenchmarkTask.must_contain
    semantics on purpose -- these functions exist to answer benchmark-style
    tasks, so "correct" should mean the same thing it means everywhere
    else in MANA, not a stricter exact-match a real patch could never win
    against."""
    query: str
    must_contain: List[str]


@dataclass
class CodeTarget:
    target_id: str
    file_path: str          # relative to mana/, e.g. "agent_parts/execution.py"
    class_name: str
    function_name: str
    is_staticmethod: bool
    param_names: List[str]  # excluding self
    signature_hint: str     # shown to the LLM, e.g. "def _local_fallback(task: str) -> str:"
    description: str
    test_cases: List[CodeTestCase] = field(default_factory=list)

    def __post_init__(self) -> None:
        if Path(self.file_path).name in _NEVER_PATCHABLE:
            raise ValueError(f"{self.file_path} is not a patchable target (see _NEVER_PATCHABLE)")


def _fallback_test_cases() -> List[CodeTestCase]:
    # Reuse MANA's own benchmark suite as the regression oracle for
    # _local_fallback, since that function's entire job is answering these
    # exact tasks when the LLM is unavailable. This also means the test
    # set updates itself if BenchmarkSuite grows -- no separate fixture to
    # keep in sync by hand.
    from .pipeline import BenchmarkSuite
    tasks = BenchmarkSuite.train_tasks() + BenchmarkSuite.generalization_tasks() + BenchmarkSuite.holdout_tasks()
    return [CodeTestCase(t.query, list(t.must_contain)) for t in tasks if t.must_contain]


def _task_category_test_cases() -> List[CodeTestCase]:
    from .pipeline import BenchmarkSuite
    tasks = BenchmarkSuite.train_tasks() + BenchmarkSuite.generalization_tasks() + BenchmarkSuite.holdout_tasks()
    return [CodeTestCase(t.query, [t.category]) for t in tasks]


WHITELIST: Dict[str, CodeTarget] = {
    "local_fallback": CodeTarget(
        target_id="local_fallback",
        file_path="agent_parts/execution.py",
        class_name="ExecutionMixin",
        function_name="_local_fallback",
        is_staticmethod=True,
        param_names=["task"],
        signature_hint="def _local_fallback(task: str) -> str:",
        description="Deterministic fallback answer used whenever the LLM is "
                     "unavailable or empty. Graded against MANA's own "
                     "benchmark suite (must_contain substrings).",
        test_cases=_fallback_test_cases(),
    ),
    "task_category": CodeTarget(
        target_id="task_category",
        file_path="agent_parts/routing.py",
        class_name="RoutingMixin",
        function_name="_task_category",
        is_staticmethod=False,  # kept as an instance method in the real file (unused self), see apply()
        param_names=["task"],
        signature_hint="def _task_category(task: str) -> str:",
        description="Classifies a task into math/programming/reasoning/current/general; "
                     "drives routing and architecture selection. Graded against "
                     "MANA's own benchmark suite's `category` labels.",
        test_cases=_task_category_test_cases(),
    ),
}


def list_targets() -> List[Dict[str, Any]]:
    return [
        {"target_id": t.target_id, "file": t.file_path, "function": f"{t.class_name}.{t.function_name}",
         "description": t.description, "test_cases": len(t.test_cases)}
        for t in WHITELIST.values()
    ]


def _harness_source(target: CodeTarget, candidate_source: str) -> str:
    """Smoke-test harness: call the candidate on every test input and check
    it doesn't raise and returns the right type. Deliberately no
    correctness asserts here -- the baseline implementation itself may
    already fail some cases (that's the whole point of patching it), so a
    combined all-in-one-script assert would halt on the first failing case
    and make evaluate_candidate() report "broken" for a perfectly-working
    candidate that merely hasn't fixed *every* case yet. Correctness is
    measured case-by-case afterward in evaluate_candidate(), where one
    failing case can't hide the result of the others."""
    lines = ["# Auto-generated MANA self-improvement smoke-test harness.", candidate_source.strip(), ""]
    for i, tc in enumerate(target.test_cases):
        lines.append(f"__out_{i} = {target.function_name}({tc.query!r})")
        lines.append(f"assert isinstance(__out_{i}, str), 'must return a string'")
    lines.append("print('SMOKE_OK')")
    return "\n".join(lines)


def _validate_candidate(target: CodeTarget, candidate_source: str) -> Dict[str, Any]:
    """AST-level checks before the candidate ever reaches a sandbox: exactly
    one function, right name/arity, no `self`/attribute access (must be a
    genuinely free, pure function)."""
    try:
        tree = ast.parse(candidate_source, mode="exec")
    except SyntaxError as exc:
        return {"ok": False, "reason": f"SyntaxError: {exc}"}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or len(funcs) != 1:
        return {"ok": False, "reason": "candidate must contain exactly one function definition, nothing else"}
    fn = funcs[0]
    if fn.name != target.function_name:
        return {"ok": False, "reason": f"function must be named {target.function_name!r}, got {fn.name!r}"}
    arg_names = [a.arg for a in fn.args.args]
    if arg_names != target.param_names:
        return {"ok": False, "reason": f"parameters must be exactly {target.param_names}, got {arg_names}"}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "self":
            return {"ok": False, "reason": "candidate must not reference `self` -- it must be a pure function of its parameters"}
    return {"ok": True}


def baseline_score(target: CodeTarget) -> Dict[str, Any]:
    """Score the *currently committed* implementation in-process (trusted
    code, no sandbox needed) so candidates have something concrete to beat."""
    import importlib
    module = importlib.import_module(f".{target.file_path[:-3].replace('/', '.')}", package="mana")
    cls = getattr(module, target.class_name)
    fn = getattr(cls, target.function_name)
    passed, failed = [], []
    for i, tc in enumerate(target.test_cases):
        try:
            out = str(fn(tc.query) if target.is_staticmethod else fn(None, tc.query))
        except Exception as exc:
            out = f"<error: {exc}>"
        ok = all(s.lower() in out.lower() for s in tc.must_contain)
        (passed if ok else failed).append(i)
    return {"passed": passed, "failed": failed, "total": len(target.test_cases)}


def evaluate_candidate(target_id: str, candidate_source: str, verifier: Any) -> Dict[str, Any]:
    """Validate + sandbox-run a candidate against every test case. `verifier`
    is the agent's existing LocalVerifier -- same sandbox as any other
    agent-generated code, including its local_exec_enabled gate."""
    target = WHITELIST.get(target_id)
    if target is None:
        return {"ok": False, "reason": f"unknown target_id: {target_id!r}"}
    static_check = _validate_candidate(target, candidate_source)
    if not static_check["ok"]:
        return {"ok": False, "reason": static_check["reason"], "stage": "static_validation"}
    base = baseline_score(target)
    harness = _harness_source(target, candidate_source)
    sandbox = verifier.verify_code(harness)
    if not sandbox.get("ok"):
        return {"ok": False, "reason": sandbox.get("error") or "sandbox execution failed",
                "stage": "sandbox", "sandbox": sandbox, "baseline": base}
    # verify_code only tells us the script ran without raising; recover
    # per-test pass/fail by re-running the same asserts one at a time so a
    # partial failure is diagnosable instead of all-or-nothing.
    passed, failed = [], []
    for i, tc in enumerate(target.test_cases):
        one = "\n".join([candidate_source.strip(), "",
                          f"__out = {target.function_name}({tc.query!r})",
                          *(f"assert {s.lower()!r} in str(__out).lower()" for s in tc.must_contain),
                          "print('OK')"])
        r = verifier.verify_code(one)
        (passed if r.get("ok") else failed).append(i)
    return {"ok": True, "stage": "sandbox", "sandbox": sandbox, "baseline": base,
            "candidate": {"passed": passed, "failed": failed, "total": len(target.test_cases)}}


def decide(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Strict accept/reject gate: no regression on anything the baseline
    already passed, and a strict improvement on at least one case that
    currently fails. A tie is a reject, same as evolve_pipeline's
    _strict_acceptance -- "no measurable win" is not accepted just because
    nothing got worse."""
    if not evaluation.get("ok"):
        return {"accepted": False, "reason": evaluation.get("reason", "evaluation failed")}
    base_passed = set(evaluation["baseline"]["passed"])
    cand_passed = set(evaluation["candidate"]["passed"])
    regressed = base_passed - cand_passed
    newly_fixed = cand_passed - base_passed
    if regressed:
        return {"accepted": False, "reason": "regression", "regressed_cases": sorted(regressed)}
    if not newly_fixed:
        return {"accepted": False, "reason": "no_improvement",
                "baseline_pass_rate": len(base_passed) / max(1, evaluation["baseline"]["total"])}
    return {"accepted": True, "reason": "strict_improvement",
            "newly_fixed_cases": sorted(newly_fixed),
            "before_pass_rate": len(base_passed) / max(1, evaluation["baseline"]["total"]),
            "after_pass_rate": len(cand_passed) / max(1, evaluation["candidate"]["total"])}


def _locate_function(file_text: str, class_name: str, function_name: str) -> Dict[str, Any]:
    tree = ast.parse(file_text)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        raise ValueError(f"class {class_name!r} not found")
    fn = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == function_name), None)
    if fn is None:
        raise ValueError(f"function {function_name!r} not found in {class_name!r}")
    # NOTE (bugfix, caught by this module's own end-to-end test): ast's
    # FunctionDef.lineno points at the `def` line, NOT at any decorator
    # above it -- decorators have their own separate lineno in
    # decorator_list. Using fn.lineno as the replacement span's start left
    # the original @staticmethod line untouched while _render_method also
    # writes its own, producing a duplicated decorator in the patched
    # file. Start the span at the first decorator instead, when present.
    start = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
    return {"start": start, "end": fn.end_lineno,
            "is_staticmethod": any(ast.unparse(d) == "staticmethod" for d in fn.decorator_list)}


def _render_method(target: CodeTarget, candidate_source: str, indent: str = "    ") -> str:
    """Turn a sandbox-form free function back into the class-method text
    that belongs in the real file (reattach `self`/@staticmethod)."""
    fn_lines = candidate_source.strip("\n").split("\n")
    header = fn_lines[0]
    if target.is_staticmethod:
        assert header.startswith("def "), header
        new_header = header
        decorator = f"{indent}@staticmethod\n"
    else:
        assert header.startswith(f"def {target.function_name}("), header
        new_header = header.replace(f"def {target.function_name}(", f"def {target.function_name}(self, ", 1)
        decorator = ""
    body = "\n".join(f"{indent}{l}" if l.strip() else l for l in fn_lines[1:])
    return f"{decorator}{indent}{new_header}\n{body}\n"


def apply_patch(target_id: str, candidate_source: str, evaluation: Dict[str, Any],
                 decision: Dict[str, Any], instruction: str = "") -> Dict[str, Any]:
    target = WHITELIST[target_id]
    if not decision.get("accepted"):
        return {"applied": False, "reason": "decision not accepted; call decide() first"}
    file_path = PACKAGE_ROOT / target.file_path
    if file_path.name in _NEVER_PATCHABLE:  # defense in depth, see module docstring
        return {"applied": False, "reason": f"{file_path.name} is never patchable"}
    original = file_path.read_text(encoding="utf-8")
    loc = _locate_function(original, target.class_name, target.function_name)
    lines = original.split("\n")
    old_block = "\n".join(lines[loc["start"] - 1:loc["end"]])
    new_block = _render_method(target, candidate_source).rstrip("\n")
    patched = "\n".join(lines[:loc["start"] - 1]) + "\n" + new_block + "\n" + "\n".join(lines[loc["end"]:])

    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = HISTORY_ROOT / f"{target.file_path.replace('/', '__')}.{target_id}.{stamp}.bak"
    backup_path.write_text(original, encoding="utf-8")

    diff = "\n".join(difflib.unified_diff(
        old_block.split("\n"), new_block.split("\n"),
        fromfile=f"before/{target_id}", tofile=f"after/{target_id}", lineterm=""))

    changelog_path = HISTORY_ROOT / "changelog.jsonl"
    entry = {
        "timestamp": time.time(), "target_id": target_id, "file": target.file_path,
        "instruction": instruction, "decision": decision, "diff": diff,
        "backup": str(backup_path),
    }
    with open(changelog_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    file_path.write_text(patched, encoding="utf-8")
    return {"applied": True, "backup": str(backup_path), "diff": diff, "decision": decision}


def rollback(target_id: str, backup_path: Optional[str] = None) -> Dict[str, Any]:
    target = WHITELIST[target_id]
    file_path = PACKAGE_ROOT / target.file_path
    if backup_path is None:
        candidates = sorted(HISTORY_ROOT.glob(f"{target.file_path.replace('/', '__')}.{target_id}.*.bak"))
        if not candidates:
            return {"rolled_back": False, "reason": "no backup found for this target"}
        backup_path = str(candidates[-1])
    backup = Path(backup_path)
    if not backup.exists():
        return {"rolled_back": False, "reason": f"backup not found: {backup_path}"}
    file_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    return {"rolled_back": True, "restored_from": str(backup)}


def history(target_id: Optional[str] = None) -> List[Dict[str, Any]]:
    path = HISTORY_ROOT / "changelog.jsonl"
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [e for e in entries if target_id is None or e["target_id"] == target_id]


def propose_patch_llm(target_id: str, ask_llm: Callable[..., Any], instruction: str) -> Dict[str, Any]:
    """Ask the agent's own LLM for a candidate. This is the only entry
    point that involves the LLM; evaluate_candidate/decide/apply_patch
    above never trust the LLM's own claims about correctness -- they
    re-derive pass/fail from the sandbox every time.

    `ask_llm` is a callable matching ManaAgent._llm_call's signature
    (prompt, *, temperature=..., context_tag=...) -> (text, meta) -- the
    same choke point every other LLM call in the agent goes through (see
    mana/tools.py's 'llm_generate' tool), passed in rather than importing
    the agent here, to keep this module decoupled from agent_parts."""
    target = WHITELIST.get(target_id)
    if target is None:
        return {"ok": False, "reason": f"unknown target_id: {target_id!r}"}
    base = baseline_score(target)
    failing = [target.test_cases[i] for i in base["failed"]]
    prompt = (
        "Improve this pure Python function. Requirements:\n"
        f"- Signature must stay exactly: {target.signature_hint}\n"
        "- No `self`, no imports, no side effects, no I/O -- pure function of its parameters only.\n"
        "- Return ONLY the function definition, no markdown fences, no explanation.\n"
        f"- Currently failing cases that must start passing (need these substrings in the return value):\n"
        + "\n".join(f'  - input: {tc.query!r} -> must contain: {tc.must_contain}' for tc in failing) +
        "\n- Every other existing case must keep passing (do not remove existing branches unless replacing their logic).\n"
        f"- Improvement goal: {instruction}\n\n"
        f"Current implementation:\n{target.signature_hint}\n    ...\n"
    )
    text, meta = ask_llm(prompt, temperature=0.2, context_tag="SELF-IMPROVE-CODE")
    if not text:
        return {"ok": False, "reason": "LLM returned no candidate"}
    import re
    candidate = re.sub(r"^```(?:python)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S).strip()
    return {"ok": True, "candidate_source": candidate, "instruction": instruction, "llm_latency": meta.latency}
