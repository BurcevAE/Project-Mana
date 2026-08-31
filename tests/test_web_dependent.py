"""
tests/test_web_dependent.py — needs real network access (ddgs/
duckduckgo_search backend). Skipped unless MANA_TEST_WEB=1. UNVERIFIED by
me -- the sandbox this suite was authored in has no network egress.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MANA_TEST_WEB") != "1",
    reason="set MANA_TEST_WEB=1 (and have real network access) to run these",
)


@pytest.fixture
def isolated_agent_web(isolated_config):
    isolated_config.enable_web = True
    from mana import ManaAgent
    return ManaAgent(isolated_config)


def test_web_search_tool_returns_results(isolated_agent_web):
    assert isolated_agent_web._tool_available("web_search"), (
        "web_search reports unavailable -- is ddgs/duckduckgo_search installed? "
        "`pip install ddgs`"
    )
    result = isolated_agent_web.tools.call("web_search", query="Python programming language", max_results=3)
    print("WEB_SEARCH_META:", result.meta)
    print("WEB_SEARCH_RESULT_COUNT:", len(result.output or []))
    if result.output:
        print("FIRST_RESULT:", result.output[0])
    assert result.meta.get("attempted") is True


def test_web_search_circuit_breaker_does_not_trip_on_first_call(isolated_agent_web):
    """A single successful/failed call should not exhaust
    web_failure_limit -- this just documents current health after one
    real call, for your log.

    NOTE: health() lives on WebHealthManager, not on WebSearcher itself;
    WebSearcher exposes status(), which includes the health figure. An
    earlier version of this test called isolated_agent_web.web.health()
    and failed with AttributeError on the first real-network run -- it was
    written without network access to verify it against."""
    isolated_agent_web.tools.call("web_search", query="test query", max_results=1)
    status = isolated_agent_web.web.status()
    print("WEB_STATUS_AFTER_ONE_CALL:", status)
    assert isinstance(status, dict)


def test_build_context_includes_web_block_for_current_events_query(isolated_agent_web):
    from mana.pipeline import PipelineSpec
    spec = PipelineSpec(web_mode="always", use_web=True).normalize(isolated_agent_web.config)
    context, trace = isolated_agent_web._build_context("Какие последние новости о разработке ИИ?", spec)
    print("WEB_TRACE:", {k: v for k, v in trace.items() if "web" in k})
    print("CONTEXT_HAS_WEB_BLOCK:", "[WEB]" in context)
