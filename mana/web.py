"""
mana.web — web search backend with health/circuit-breaker management.
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
from .optional_deps import DDGS, WEB_BACKEND, HAS_WEB

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------

class WebHealthManager:
    def __init__(self, config: Config):
        self.config = config
        self.failures = 0
        self.calls = 0
        self.errors = 0
        self.successes = 0
        self.last_error = None
        self.opened_at = 0.0
        self.state = "closed"
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            if self.state == "closed":
                return True
            if time.time() - self.opened_at >= self.config.web_cooldown_seconds:
                self.state = "half_open"
                return True
            return False

    def record(self, ok: bool, error: Optional[str] = None) -> None:
        with self.lock:
            self.calls += 1
            if ok:
                self.successes += 1
                self.failures = 0
                self.last_error = None
                self.state = "closed"
            else:
                self.errors += 1
                self.failures += 1
                self.last_error = error
                if self.failures >= self.config.web_failure_limit:
                    self.state = "open"
                    self.opened_at = time.time()

    def health(self) -> float:
        with self.lock:
            if self.calls <= 0:
                return 1.0
            return self.successes / self.calls


class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


# Failure taxonomy. The old code collapsed every exception into
# "transport_error", so "the network hiccuped" and "the search engine
# refused us as a bot" were indistinguishable -- and both were retried
# immediately, which makes rate limiting worse rather than better.
#
# Honest limitation, stated up front: from a client's side it is IMPOSSIBLE
# to be certain. A silent block can look exactly like an honest empty
# result. What follows narrows the uncertainty and, above all, stops
# reporting "unknown" as "fine" -- it does not eliminate the ambiguity.
REASON_OK = "ok"
REASON_NO_RESULTS = "no_results"          # response arrived, list was empty -- cause unknown
REASON_RATE_LIMITED = "rate_limited"      # HTTP 429 / explicit throttling
REASON_BLOCKED = "blocked"                # HTTP 403 / captcha / bot detection
REASON_TRANSPORT = "transport_error"      # genuine network/parse failure
REASON_TIMEOUT = "timeout"

_RATE_LIMIT_MARKERS = ("429", "too many requests", "rate limit", "ratelimit", "throttl")
_BLOCKED_MARKERS = ("403", "forbidden", "captcha", "challenge", "blocked", "denied",
                    "access denied", "unusual traffic", "bot detect")
_TIMEOUT_MARKERS = ("timeout", "timed out")


def classify_web_failure(exc: BaseException) -> str:
    """Map an exception to one of the REASON_* codes above.

    Matching is on the textual form of the exception because `ddgs` wraps
    HTTP errors in its own exception types and does not expose a status
    code uniformly -- there is no structured field to read. This is a
    heuristic over error text, so it can misclassify; when nothing matches
    we deliberately fall back to REASON_TRANSPORT rather than guessing
    "blocked", so an ordinary network glitch is never reported as
    censorship.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return REASON_RATE_LIMITED
    if status in (401, 403):
        return REASON_BLOCKED
    if any(m in text for m in _RATE_LIMIT_MARKERS):
        return REASON_RATE_LIMITED
    if any(m in text for m in _BLOCKED_MARKERS):
        return REASON_BLOCKED
    if any(m in text for m in _TIMEOUT_MARKERS):
        return REASON_TIMEOUT
    return REASON_TRANSPORT


#: Reasons where retrying immediately makes the situation worse rather
#: than better -- the server is already telling us to back off.
NON_RETRYABLE_REASONS = frozenset({REASON_RATE_LIMITED, REASON_BLOCKED})


class WebSearcher:
    def __init__(self, config: Config, vlog=None):
        self.config = config
        self.enabled = bool(config.enable_web and HAS_WEB)
        self.health_manager = WebHealthManager(config)
        self.vlog = vlog
        self.retries = 0
        self.transport_errors = 0
        self.no_results = 0
        self.blocked_responses = 0
        self._request_lock = threading.RLock()

    def _log(self, message: str) -> None:
        if self.vlog:
            self.vlog(message)

    def search(self, query: str, max_results: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.enabled or DDGS is None:
            return [], {"attempted": False, "ok": False, "reason": "disabled", "tool_health": 1.0, "retries": 0, "transport_errors": 0}
        if not self.health_manager.allow():
            return [], {"attempted": False, "ok": False, "reason": "circuit_open", "tool_health": self.health_manager.health(), "retries": 0, "transport_errors": 0}
        limit = int(max_results or self.config.max_web_results)
        max_retries = max(0, int(getattr(self.config, "web_max_retries", 2)))
        started = time.perf_counter()
        attempts = 0
        transport_errors = 0

        def _do_request():
            ddgs = DDGS()
            try:
                return list(ddgs.text(query, region=self.config.web_region, max_results=limit) or [])
            except TypeError:
                return list(ddgs.text(query, max_results=limit) or [])

        lock = self._request_lock if getattr(self.config, "web_serialized_requests", True) else _NullContext()
        self._log(f"WEB START | query={query!r} | limit={limit} | backend={WEB_BACKEND} | retries={max_retries}")
        with lock:
            for attempt in range(max_retries + 1):
                attempts += 1
                try:
                    rows = (_do_request() or [])[:limit]
                    elapsed = time.perf_counter() - started
                    if rows:
                        self.health_manager.record(True)
                        if attempt:
                            self.retries += attempt
                        self._log(f"WEB OK | results={len(rows)} | attempts={attempts} | time={elapsed:.2f}s")
                        return rows, {"attempted": True, "ok": True, "count": len(rows), "reason": REASON_OK, "tool_health": self.health_manager.health(), "latency": elapsed, "attempts": attempts, "retries": attempt, "transport_errors": transport_errors, "technical_failure": False}
                    self.no_results += 1
                    self.health_manager.record(False, "no_results")
                    self._log(f"WEB NO_RESULTS | attempts={attempts} | time={elapsed:.2f}s")
                    # Do NOT retry an empty result set. An honest "nothing
                    # matched" will not become non-empty 0.35s later, and if
                    # the emptiness is actually a silent block, retrying is
                    # exactly the wrong move. (Previously this retried up to
                    # max_retries times.)
                    return [], {"attempted": True, "ok": False, "count": 0, "reason": REASON_NO_RESULTS,
                                "tool_health": self.health_manager.health(), "latency": elapsed,
                                "attempts": attempts, "retries": attempt,
                                "transport_errors": transport_errors, "technical_failure": False,
                                "possibly_blocked": True,
                                "note": "empty result set -- an honest miss and a silent block are "
                                        "indistinguishable from the client side"}
                except Exception as exc:
                    transport_errors += 1
                    self.transport_errors += 1
                    reason = classify_web_failure(exc)
                    err_text = f"{type(exc).__name__}: {exc}"
                    self._log(f"WEB {reason.upper()} | attempt={attempt+1}/{max_retries+1} | {err_text}")
                    self.health_manager.record(False, err_text)
                    if reason in NON_RETRYABLE_REASONS:
                        # Being told to back off and immediately asking again
                        # is what turns a soft throttle into a hard ban.
                        self.blocked_responses += 1
                        elapsed = time.perf_counter() - started
                        return [], {"attempted": True, "ok": False, "count": 0, "reason": reason,
                                    "error": str(exc), "tool_health": self.health_manager.health(),
                                    "latency": elapsed, "attempts": attempts, "retries": max(0, attempts-1),
                                    "transport_errors": transport_errors, "technical_failure": True,
                                    "possibly_blocked": True}
                    if attempt < max_retries:
                        self.retries += 1
                        time.sleep(max(0.0, float(getattr(self.config, "web_retry_delay_seconds", 0.35))))
                        continue
                    elapsed = time.perf_counter() - started
                    return [], {"attempted": True, "ok": False, "count": 0, "reason": reason,
                                "error": str(exc), "tool_health": self.health_manager.health(),
                                "latency": elapsed, "attempts": attempts, "retries": max(0, attempts-1),
                                "transport_errors": transport_errors, "technical_failure": True,
                                "possibly_blocked": False}
        return [], {"attempted": True, "ok": False, "count": 0, "reason": "unknown", "tool_health": self.health_manager.health(), "latency": time.perf_counter()-started, "attempts": attempts, "retries": max(0, attempts-1), "transport_errors": transport_errors, "technical_failure": True}

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": WEB_BACKEND,
            "state": self.health_manager.state,
            "calls": self.health_manager.calls,
            "errors": self.health_manager.errors,
            "successes": self.health_manager.successes,
            "health": self.health_manager.health(),
            "last_error": self.health_manager.last_error,
            "retries": self.retries,
            "transport_errors": self.transport_errors,
            "no_results": self.no_results,
            "blocked_responses": self.blocked_responses,
            "serialized_requests": bool(getattr(self.config, "web_serialized_requests", True)),
        }
