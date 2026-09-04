"""
mana.verifier — local arithmetic/code verification sandbox (no network, no elevated privileges).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pickle
import random
import re
import statistics
import sys
import threading
import time
import subprocess
import tempfile
import shutil
import platform
import ast
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .paths import sandbox_python, sandbox_python_available

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.10"


# ---------------------------------------------------------------------------
# Local verification / execution sandbox (v4.6)
# ---------------------------------------------------------------------------

class LocalVerifier:
    """Conservative local verifier.

    Safe arithmetic is evaluated in-process using AST. Arbitrary Python execution is
    disabled unless Config.local_exec_enabled=True. When enabled, code runs in a
    temporary working directory without shell=True, with a short timeout and bounded
    output. The child inherits the current user's OS permissions; MANA does not elevate
    privileges or request administrator rights.
    """
    def __init__(self, config: Config, vlog=None):
        self.config = config
        self.vlog = vlog
        self.stats = {"checks": 0, "arithmetic": 0, "code": 0, "ok": 0, "failed": 0, "timeouts": 0}
        self.lock = threading.Lock()

    def _log(self, msg: str) -> None:
        if self.vlog:
            self.vlog(msg)

    #: Ceiling on how large an intermediate integer may get. Exact
    #: arithmetic removes the float rounding, but it also removes the
    #: overflow that used to cap the work: 9**9**9 is a valid expression
    #: whose exact value would occupy gigabytes. Digits, not magnitude,
    #: because that is what actually costs time and memory.
    MAX_INT_DIGITS = 4000

    @staticmethod
    def _guard_int(value: Any) -> Any:
        if isinstance(value, int) and value.bit_length() > LocalVerifier.MAX_INT_DIGITS * 4:
            raise ValueError("intermediate integer too large")
        return value

    @staticmethod
    def _safe_math_node(node: ast.AST) -> Any:
        """Evaluate a constant arithmetic expression EXACTLY.

        This used to coerce everything to float, which quietly made the
        one component MANA treats as ground truth wrong on integers:

            999999999999999999 * 3  ->  3e+18   (correct: 2999999999999999997)

        and it reported ok:true. A verifier that is confidently wrong is
        worse than no verifier, because every layer above it -- the trust
        levels, the acceptance gates, the fitness that drives evolution --
        is built on believing this number.

        Integers now stay integers, and float appears only where the
        expression genuinely produces one (true division, a float
        literal). The result type is therefore int|float rather than
        float; callers already stringify it, and `verify()` compares
        against both the exact form and the integral form.
        """
        if isinstance(node, ast.Expression):
            return LocalVerifier._safe_math_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            raise ValueError("unsupported expression")     # True/False are ints in Python
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return LocalVerifier._guard_int(node.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, float) and math.isfinite(node.value):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = LocalVerifier._safe_math_node(node.operand)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            a = LocalVerifier._safe_math_node(node.left)
            b = LocalVerifier._safe_math_node(node.right)
            if isinstance(node.op, ast.Add): return LocalVerifier._guard_int(a + b)
            if isinstance(node.op, ast.Sub): return LocalVerifier._guard_int(a - b)
            if isinstance(node.op, ast.Mult): return LocalVerifier._guard_int(a * b)
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.FloorDiv): return a // b
            if isinstance(node.op, ast.Mod): return a % b
            if isinstance(node.op, ast.Pow):
                # The old bound compared against 1e6 on the base, which a
                # float comparison made approximate. Exact arithmetic lets
                # the limit be stated in the terms that matter: how big
                # can the answer get.
                if not isinstance(b, int) or b < 0:
                    raise ValueError("unsafe exponent")
                if b > 64:
                    raise ValueError("unsafe exponent")
                base_bits = a.bit_length() if isinstance(a, int) else 64
                if base_bits * max(1, b) > LocalVerifier.MAX_INT_DIGITS * 4:
                    raise ValueError("unsafe exponent")
                return LocalVerifier._guard_int(a ** b)
        raise ValueError("unsupported expression")

    def verify_expression(self, expr: str) -> Dict[str, Any]:
        expr=(expr or "").strip()
        with self.lock: self.stats["checks"] += 1; self.stats["arithmetic"] += 1
        try:
            tree=ast.parse(expr, mode="eval")
            value=self._safe_math_node(tree)
            # float(value) on a large exact integer raises OverflowError,
            # which the old code could never hit because everything was
            # already a float. An int is finite by construction.
            ok = True if isinstance(value, int) else math.isfinite(value)
            result={"kind":"arithmetic","ok":ok,"value":value,"expression":expr,"executor":"ast"}
        except Exception as exc:
            result={"kind":"arithmetic","ok":False,"expression":expr,"error":f"{type(exc).__name__}: {exc}","executor":"ast"}
        with self.lock: self.stats["ok" if result["ok"] else "failed"] += 1
        return result

    def _static_policy(self, code: str) -> Dict[str, Any]:
        """Reject obviously dangerous Python before sandbox execution."""
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return {"ok": False, "reason": f"SyntaxError: {exc}"}
        blocked_modules = {"os", "subprocess", "socket", "pathlib", "shutil", "ctypes", "sys", "multiprocessing", "pickle", "importlib", "requests", "urllib", "http", "ftplib", "builtins"}
        blocked_calls = {"eval", "exec", "compile", "open", "input", "breakpoint", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                names = [a.name.split(".")[0] for a in getattr(node, "names", [])]
                roots = ([mod.split(".")[0]] if mod else []) + names
                for root in roots:
                    if root in blocked_modules:
                        return {"ok": False, "reason": f"blocked import: {root}"}
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if name in blocked_calls:
                    return {"ok": False, "reason": f"blocked call: {name}"}
            if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
                return {"ok": False, "reason": "dunder attribute access is blocked"}
        return {"ok": True}

    def verify_code(self, code: str) -> Dict[str, Any]:
        if not self.config.local_exec_enabled:
            return {"kind":"code","ok":False,"enabled":False,"error":"local execution disabled; start with --enable-local-exec"}
        code=(code or "").strip()
        if not code or len(code)>self.config.local_exec_max_code_chars:
            return {"kind":"code","ok":False,"enabled":True,"error":"empty or code too large"}
        policy = self._static_policy(code)
        if not policy.get("ok"):
            return {"kind":"code","ok":False,"enabled":True,"policy_blocked":True,"error":policy.get("reason","blocked by policy")}
        # The interpreter, not `sys.executable`. In a frozen build the
        # latter is MANA.exe: this line would have re-launched the whole
        # agent instead of running the snippet, and the "verification"
        # would have reported whatever the agent printed. Refusing up
        # front is the only safe answer -- a missing sandbox interpreter
        # must look like a missing sandbox, not like a passing test.
        interpreter = sandbox_python()
        if not sandbox_python_available():
            return {"kind": "code", "ok": False, "enabled": True, "sandbox_missing": True,
                    "error": f"sandbox interpreter not found at {interpreter}"}
        with self.lock: self.stats["checks"] += 1; self.stats["code"] += 1
        root=Path(self.config.local_exec_workdir); root.mkdir(parents=True, exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix="mana_", dir=str(root)))
        script=work/"check.py"; script.write_text(code, encoding="utf-8")
        env={"PYTHONIOENCODING":"utf-8","PYTHONDONTWRITEBYTECODE":"1","PATH":os.environ.get("PATH","")}
        start=time.perf_counter(); timed_out=False
        creationflags=0
        preexec = None
        if os.name=="nt":
            creationflags=getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)
        else:
            import resource
            def preexec():
                resource.setrlimit(resource.RLIMIT_CPU,(self.config.local_exec_cpu_seconds,self.config.local_exec_cpu_seconds))
                mem=self.config.local_exec_memory_mb*1024*1024
                resource.setrlimit(resource.RLIMIT_AS,(mem,mem))
                resource.setrlimit(resource.RLIMIT_NPROC,(self.config.local_exec_max_processes,self.config.local_exec_max_processes))
            creationflags=0
        try:
            cp=subprocess.run([interpreter,"-I","-S",str(script)],cwd=str(work),env=env,
                              stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,encoding="utf-8",errors="replace",timeout=self.config.local_exec_timeout,
                              shell=False,creationflags=creationflags,preexec_fn=preexec if os.name != "nt" else None)
            out=(cp.stdout or "")[:self.config.local_exec_max_output]
            err=(cp.stderr or "")[:self.config.local_exec_max_output]
            ok=(cp.returncode==0)
            result={"kind":"code","ok":ok,"enabled":True,"returncode":cp.returncode,"stdout":out,"stderr":err,
                    "latency":time.perf_counter()-start,"workdir":str(work)}
        except subprocess.TimeoutExpired as exc:
            timed_out=True
            if os.name=="nt":
                try: subprocess.run(["taskkill","/F","/T","/PID",str(getattr(exc,"pid",0))],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=1)
                except Exception: pass
            result={"kind":"code","ok":False,"enabled":True,"timeout":True,"error":"execution timeout",
                    "latency":time.perf_counter()-start,"workdir":str(work)}
        except Exception as exc:
            result={"kind":"code","ok":False,"enabled":True,"error":f"{type(exc).__name__}: {exc}",
                    "latency":time.perf_counter()-start,"workdir":str(work)}
        finally:
            if timed_out:
                with self.lock: self.stats["timeouts"] += 1
            with self.lock: self.stats["ok" if result.get("ok") else "failed"] += 1
            shutil.rmtree(work, ignore_errors=True)
        return result

    def verify_code_with_tests(self, code: str, tests: str) -> Dict[str, Any]:
        """Run generated code plus assertions in the same restricted sandbox."""
        if not self.config.local_exec_enabled:
            return {"kind":"code_tests","ok":False,"enabled":False,"error":"local execution disabled; start with --enable-local-exec"}
        combined = (code or "").strip() + "\n\n# --- MANA TESTS ---\n" + (tests or "").strip() + "\n"
        result = self.verify_code(combined)
        result["kind"] = "code_tests"
        result["tests_included"] = bool((tests or "").strip())
        return result

    # Word-form arithmetic normalization (bugfix, found on real hardware:
    # "сколько будет 17 умножить на 23" produced kind:"none" -- the patterns
    # below only ever matched symbolic operators, so a naturally-phrased
    # arithmetic question was silently unverifiable and a wrong LLM answer
    # went out unchecked). Longer phrases must come before shorter ones that
    # are their prefixes ("разделить на" before "разделить"), or the shorter
    # one matches first and leaves a dangling remainder.
    _WORD_OPS: Tuple[Tuple[str, str], ...] = (
        (r"\bумножить\s+на\b", "*"), (r"\bумноженное\s+на\b", "*"), (r"\bумножено\s+на\b", "*"),
        (r"\bпомножить\s+на\b", "*"), (r"\bразделить\s+на\b", "/"), (r"\bподелить\s+на\b", "/"),
        (r"\bделённое\s+на\b", "/"), (r"\bделенное\s+на\b", "/"),
        (r"\bприбавить\s+к\b", "+"), (r"\bприбавить\b", "+"), (r"\bплюс\b", "+"),
        (r"\bотнять\s+от\b", "-"), (r"\bвычесть\s+из\b", "-"), (r"\bвычесть\b", "-"), (r"\bминус\b", "-"),
        (r"\bв\s+степени\b", "**"),
        (r"\bmultiplied\s+by\b", "*"), (r"\btimes\b", "*"),
        (r"\bdivided\s+by\b", "/"), (r"\bplus\b", "+"), (r"\bminus\b", "-"),
    )

    @staticmethod
    def _normalize_word_arithmetic(text: str) -> str:
        """Rewrite spelled-out operators into symbols so the expression
        patterns in verify() can find them. Purely textual and reversible in
        meaning -- it never evaluates anything itself, and a phrase it
        doesn't recognize is simply left alone (verify() then falls through
        to kind:"none" exactly as before)."""
        out = text or ""
        for pattern, symbol in LocalVerifier._WORD_OPS:
            out = re.sub(pattern, symbol, out, flags=re.I)
        # "17 в квадрате" / "5 в кубе" -> explicit powers
        out = re.sub(r"(\d+(?:\.\d+)?)\s*в\s+квадрате\b", r"\1**2", out, flags=re.I)
        out = re.sub(r"(\d+(?:\.\d+)?)\s*в\s+кубе\b", r"\1**3", out, flags=re.I)
        return re.sub(r"\s{2,}", " ", out)

    @staticmethod
    def _expected_forms(value: Any) -> List[str]:
        """Every spelling of the answer an answer might legitimately use.

        `391` and `391.0` are the same result; so are `2999999999999999997`
        and its comma-grouped form. This replaced a float round-trip that
        raised OverflowError once the verifier started computing exactly.
        """
        forms: List[str] = [str(value)]
        if isinstance(value, int):
            forms.append(f"{value}.0")
        elif isinstance(value, float) and value.is_integer():
            forms.append(str(int(value)))
        return [f for f in forms if f]

    #: Characters an arithmetic expression may be built from. Anything
    #: else -- a letter, a colon, a comma -- ends the expression.
    _EXPR_CHARS = re.compile(r"[0-9.\s+\-*/%()]+")
    #: A run must contain a real operator, or "12" alone would parse as an
    #: expression and every task with a number in its wording would be
    #: claimed by the arithmetic path.
    _HAS_OPERATOR = re.compile(r"[+\-*/%]")

    #: Words that introduce the expression being asked about. The
    #: anchor matters more than it looks: a real prompt carries a system
    #: preamble, a date and instructions, and the longest run of
    #: arithmetic characters in it is not the question.
    _ASKS = re.compile(
        r"(вычисли|найди\s+значение|сколько\s+будет|чему\s+равн\w*|посчитай|calculate)",
        re.I)

    @classmethod
    def extract_expression(cls, task: str) -> str:
        r"""The arithmetic expression this task is ASKING about, or "".

        Three rules, each of which exists because breaking it produced a
        wrong answer with no uncertainty attached:

        **Whole, not truncated.** The first version matched only after a
        keyword and could not see parentheses, so "Вычисли: (987 + 33) * 11"
        yielded `987 + 33` and answered 1020. Because `verify` shares this
        code, the arithmetic verifier then called the correct answer wrong
        and stamped the wrong one INDEPENDENTLY_VERIFIED.

        **Anchored, not longest.** Replacing that with "the longest
        balanced run" fixed truncation and created a worse failure: in a
        real agent prompt -- system preamble, today's date, tool list --
        the longest run is the date. `2026-09-04` was refused only because
        a leading zero is a syntax error in Python; `2026-9-4` evaluates
        to 2013, and the brain would have answered 2013 to an arithmetic
        question, exactly.

        **Ambiguous means refuse.** Where no asking word anchors it and
        several runs would evaluate, there is no way to tell which one was
        the question, and picking one is a guess wearing a proof's
        clothes.
        """
        text = cls._normalize_word_arithmetic((task or "").strip().lower())
        candidates = []
        for match in cls._EXPR_CHARS.finditer(text):
            candidate = match.group(0).strip()
            if not cls._HAS_OPERATOR.search(candidate):
                continue
            if candidate.count("(") != candidate.count(")"):
                continue
            if not cls.evaluate_expression(candidate).get("ok"):
                continue
            candidates.append((match.start(), candidate))
        if not candidates:
            return ""

        asked = cls._ASKS.search(text)
        if asked:
            after = [c for start, c in candidates if start >= asked.end()]
            if after:
                return max(after, key=len)
        if len(candidates) == 1:
            return candidates[0][1]
        return ""

    @classmethod
    def evaluate_expression(cls, expr: str) -> Dict[str, Any]:
        """Evaluate without an instance, for callers that have no config.

        Same safe node set as `verify_expression`; that one additionally
        keeps per-instance statistics, which a stateless caller neither
        has nor needs.
        """
        expr = (expr or "").strip()
        try:
            tree = ast.parse(expr, mode="eval")
            value = cls._safe_math_node(tree)
            ok = True if isinstance(value, int) else math.isfinite(value)
            return {"kind": "arithmetic", "ok": ok, "value": value,
                    "expression": expr, "executor": "ast"}
        except Exception as exc:
            return {"kind": "arithmetic", "ok": False, "expression": expr,
                    "error": f"{type(exc).__name__}: {exc}", "executor": "ast"}

    def verify(self, task: str, answer: str, category: str) -> Dict[str, Any]:
        """Verify simple claims. It intentionally refuses to execute arbitrary answer text.

        Uses `extract_expression` rather than its own patterns. It used to
        carry a second copy, and the copies disagreed: the one here could
        not see parentheses, so it verified a fragment and reported the
        wrong number as the truth.
        """
        a=(answer or "").strip()
        expression = self.extract_expression(task)
        if expression:
            vr=self.verify_expression(expression)
            if vr.get("ok"):
                compact = re.sub(r"\s+", "", a.lower())
                expected_tokens = LocalVerifier._expected_forms(vr["value"])
                answer_mentions=any(x and x in compact for x in expected_tokens)
                vr["answer_mentions_value"]=answer_mentions; vr["verified"]=answer_mentions
                return vr
            return vr
        return {"kind":"none","verified":False,"available":self.config.local_exec_enabled}
