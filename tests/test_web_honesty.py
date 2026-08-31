"""
tests/test_web_honesty.py — audit #16 plus the bot-detection questions it
raised: the router must not learn from attempts, and "we don't know" must
not be reported as "fine".

Two defects, one theme (a learning signal that does not reflect reality):

  1. execution_success counted `web_attempted` as success, so a search that
     was refused, throttled, or returned nothing still trained the router
     as though the web arm had worked.
  2. every exception collapsed into "transport_error" and was retried
     immediately -- so being throttled looked like a network glitch, and
     the retry made the throttling worse.

Honest limitation encoded here on purpose: an empty result set is
genuinely ambiguous from the client side (honest miss vs silent block).
The tests assert we FLAG that ambiguity, not that we resolve it.
"""
from __future__ import annotations

import pytest

from mana.agent_parts.routing import RoutingMixin
from mana.web import (NON_RETRYABLE_REASONS, REASON_BLOCKED, REASON_RATE_LIMITED,
                      REASON_TIMEOUT, REASON_TRANSPORT, classify_web_failure)


# --- 1. attempting the web is not the same as the web working -----------

def test_attempted_but_failed_web_is_not_success():
    """THE audit #16 case."""
    r = RoutingMixin.evaluate_route_execution("web", web_attempted=True, web_ok=False)
    assert r["execution_success"] is False, "an attempt that returned nothing is not a success"
    assert r["degraded"] is True
    assert r["status"] == "degraded_no_evidence"


def test_successful_web_is_success():
    r = RoutingMixin.evaluate_route_execution("web", web_attempted=True, web_ok=True)
    assert r["execution_success"] is True
    assert r["degraded"] is False


@pytest.mark.parametrize("route", ["web", "mixed"])
def test_degraded_is_reported_separately_from_plain_failure(route):
    """Degraded (tried, failed, answered anyway) must be distinguishable
    from never having tried -- they call for different responses."""
    degraded = RoutingMixin.evaluate_route_execution(route, web_attempted=True, web_ok=False)
    not_tried = RoutingMixin.evaluate_route_execution(route, web_attempted=False, web_ok=False)
    assert degraded["execution_success"] is not_tried["execution_success"] is False
    assert degraded["status"] != not_tried["status"]
    assert degraded["degraded"] is True and not_tried["degraded"] is False


def test_local_route_touching_the_web_is_flagged():
    r = RoutingMixin.evaluate_route_execution("local", web_attempted=True, web_ok=True)
    assert r["execution_success"] is False
    assert r["status"] == "unexpected_web_use"


def test_local_route_without_web_is_success():
    r = RoutingMixin.evaluate_route_execution("local", web_attempted=False, web_ok=False)
    assert r["execution_success"] is True


def test_disabled_web_is_not_counted_as_a_route_success():
    r = RoutingMixin.evaluate_route_execution("web", web_attempted=False, web_ok=False, web_enabled=False)
    assert r["execution_success"] is False
    assert r["status"] == "tool_unavailable"


def test_both_code_paths_share_one_evaluator(isolated_agent):
    """routing.py and execution.py must not drift apart on this rule --
    the same bug class we just hit with the two task classifiers."""
    import inspect
    from mana.agent_parts import execution, routing
    assert "evaluate_route_execution" in inspect.getsource(execution.ExecutionMixin)
    assert "evaluate_route_execution" in inspect.getsource(routing.RoutingMixin)


# --- 2. blocking is distinguishable from a network glitch ---------------

@pytest.mark.parametrize("message,expected", [
    ("429 Too Many Requests", REASON_RATE_LIMITED),
    ("rate limit exceeded", REASON_RATE_LIMITED),
    ("403 Forbidden", REASON_BLOCKED),
    ("Captcha challenge required", REASON_BLOCKED),
    ("detected unusual traffic from your network", REASON_BLOCKED),
    ("Read timed out", REASON_TIMEOUT),
    ("Connection reset by peer", REASON_TRANSPORT),
    ("failed to parse html", REASON_TRANSPORT),
])
def test_failure_classification(message, expected):
    assert classify_web_failure(Exception(message)) == expected


def test_http_status_attribute_wins_over_text():
    class Resp:
        status_code = 429
    exc = Exception("something opaque")
    exc.response = Resp()
    assert classify_web_failure(exc) == REASON_RATE_LIMITED


def test_unknown_errors_default_to_transport_not_blocked():
    """Never cry 'blocked' on an ordinary failure -- over-reporting bot
    detection would be its own kind of dishonest signal."""
    assert classify_web_failure(Exception("weird unexpected thing")) == REASON_TRANSPORT


def test_throttling_and_blocking_are_not_retried():
    """Retrying something that just told us to back off is what turns a
    soft throttle into a hard ban."""
    assert REASON_RATE_LIMITED in NON_RETRYABLE_REASONS
    assert REASON_BLOCKED in NON_RETRYABLE_REASONS
    assert REASON_TRANSPORT not in NON_RETRYABLE_REASONS
    assert REASON_TIMEOUT not in NON_RETRYABLE_REASONS


# --- 3. retry behaviour actually changed, not just the labels ------------

def _run_search(isolated_config, monkeypatch, backend_call):
    """Run the REAL WebSearcher.search against a fake DDGS backend, counting
    how many times the backend is actually hit. The retry/branching logic
    under test is the genuine one -- only the network call is replaced."""
    from mana import web as web_mod
    isolated_config.enable_web = True
    isolated_config.web_max_retries = 2
    isolated_config.web_retry_delay_seconds = 0.0
    calls = {"n": 0}

    class FakeDDGS:
        def text(self, *a, **k):
            calls["n"] += 1
            return backend_call()

    monkeypatch.setattr(web_mod, "HAS_WEB", True)
    monkeypatch.setattr(web_mod, "DDGS", FakeDDGS)
    searcher = web_mod.WebSearcher(isolated_config)
    rows, meta = searcher.search("anything")
    return rows, meta, calls["n"]


def test_empty_results_are_not_retried(isolated_config, monkeypatch):
    """An honest miss won't become non-empty 0.35s later, and if the
    emptiness is a silent block, retrying is exactly wrong."""
    rows, meta, n = _run_search(isolated_config, monkeypatch, lambda: [])
    assert rows == []
    assert n == 1, f"empty result set was retried {n} times"
    assert meta["reason"] == "no_results"
    assert meta["possibly_blocked"] is True, "ambiguity must be flagged, not hidden"


def test_rate_limited_is_not_retried(isolated_config, monkeypatch):
    def boom():
        raise Exception("429 Too Many Requests")
    rows, meta, n = _run_search(isolated_config, monkeypatch, boom)
    assert n == 1, f"a throttled request was retried {n} times"
    assert meta["reason"] == REASON_RATE_LIMITED
    assert meta["possibly_blocked"] is True


def test_blocked_is_not_retried(isolated_config, monkeypatch):
    def boom():
        raise Exception("403 Forbidden")
    rows, meta, n = _run_search(isolated_config, monkeypatch, boom)
    assert n == 1
    assert meta["reason"] == REASON_BLOCKED


def test_ordinary_transport_error_is_still_retried(isolated_config, monkeypatch):
    """The back-off only applies where the server asked for it."""
    def boom():
        raise Exception("Connection reset by peer")
    rows, meta, n = _run_search(isolated_config, monkeypatch, boom)
    assert n == 3, f"expected 1 try + 2 retries, got {n}"
    assert meta["reason"] == REASON_TRANSPORT
    assert meta["possibly_blocked"] is False


def test_successful_search_reports_ok(isolated_config, monkeypatch):
    rows, meta, n = _run_search(
        isolated_config, monkeypatch,
        lambda: [{"title": "t", "href": "https://x", "body": "b"}])
    assert n == 1 and meta["ok"] is True and meta["reason"] == "ok"


# --- 4. the person can actually see it ----------------------------------

def test_route_execution_is_reported_on_every_result(isolated_agent):
    """The whole point of #16 is lost if the outcome is computed but never
    shown -- the same oversight that hid verification_trust at first. The
    interactive loop renders this, so it must always be present."""
    result = isolated_agent.solve_task("какие последние новости про ИИ")
    rex = result.get("route_execution")
    assert rex is not None, "route_execution missing from result"
    assert rex["web_required"] is True, "a current-events question should require the web"
    assert rex["execution_success"] is False, "web is disabled in this fixture"
    assert rex["status"] == "tool_unavailable"


def test_local_question_reports_a_local_success(isolated_agent):
    result = isolated_agent.solve_task("Сколько будет 20 плюс 22?")
    rex = result.get("route_execution") or {}
    assert rex.get("web_required") is False
    assert rex.get("execution_success") is True
