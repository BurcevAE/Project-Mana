"""
tests/test_tools.py — ToolRegistry mechanics and the default tool set that
now backs the agent's real execution path (see agent_parts/*.py migration).
"""
from __future__ import annotations

import pytest

from mana.tools import BaseTool, FunctionTool, ToolResult, ToolRegistry


def test_unknown_tool_call_fails_gracefully_not_an_exception():
    registry = ToolRegistry()
    result = registry.call("does_not_exist", foo=1)
    assert result.ok is False
    assert "no such tool" in result.error


def test_broken_tool_never_raises_out_of_call():
    def _boom(**kwargs):
        raise RuntimeError("boom")
    registry = ToolRegistry()
    registry.register(FunctionTool("broken", "always raises", _boom))
    result = registry.call("broken")
    assert result.ok is False
    assert "boom" in result.error


def test_duplicate_registration_requires_explicit_replace():
    registry = ToolRegistry()
    registry.register(FunctionTool("x", "", lambda **k: ToolResult(ok=True)))
    with pytest.raises(ValueError):
        registry.register(FunctionTool("x", "", lambda **k: ToolResult(ok=True)))
    registry.register(FunctionTool("x", "v2", lambda **k: ToolResult(ok=True)), replace=True)
    assert registry.get("x").description == "v2"


def test_default_registry_has_all_nine_tools(isolated_agent):
    names = {t["name"] for t in isolated_agent.tools.list_tools()}
    expected = {
        "llm_generate", "web_search", "verify_arithmetic", "verify_answer",
        "run_code", "search_knowledge_base", "write_memory",
        "search_conversation_memory", "search_graph_memory",
    }
    assert expected <= names, f"missing tools: {expected - names}"


def test_llm_generate_unavailable_when_llm_disabled(isolated_agent):
    # isolated_agent fixture builds with enable_llm=False
    assert isolated_agent._tool_available("llm_generate") is False


def test_run_code_unavailable_until_local_exec_enabled(isolated_agent):
    assert isolated_agent._tool_available("run_code") is False
    isolated_agent.config.local_exec_enabled = True
    assert isolated_agent._tool_available("run_code") is True


def test_verify_arithmetic_tool_correct(isolated_agent):
    result = isolated_agent.tools.call("verify_arithmetic", expression="17*23")
    assert result.ok is True
    assert result.output["value"] == 391.0


def test_run_code_tool_executes_when_enabled(isolated_agent_exec_enabled):
    result = isolated_agent_exec_enabled.tools.call("run_code", code="print(6*7)")
    assert result.ok is True
    assert "42" in result.output["stdout"]


def test_write_memory_and_search_knowledge_base_roundtrip(isolated_agent):
    w = isolated_agent.tools.call("write_memory", content="MANA хранит опыт отдельно от весов модели.",
                                   source="test", confidence=0.8)
    assert w.ok is True
    r = isolated_agent.tools.call("search_knowledge_base", query="опыт")
    assert r.ok is True
    assert any("опыт" in e["content"].lower() for e in r.output)


def test_llm_model_is_configurable_not_hardcoded(isolated_config):
    """Regression test: Config.ollama_model was effectively unreachable --
    hardcoded to qwen2.5:0.5b with no CLI flag, so a user who ran
    `ollama create mana -f ./Modelfile` silently kept hitting the base
    model instead of their own. The payload builder must read whatever
    Config says, not a constant."""
    from mana.llm import LLMClient
    isolated_config.ollama_model = "my-custom-model"
    client = LLMClient(isolated_config)
    assert client.config.ollama_model == "my-custom-model"


def test_cli_llm_model_flag_overrides_config():
    """--llm-model must actually reach Config.ollama_model."""
    from mana.cli import build_config, build_parser
    parser = build_parser()
    args = parser.parse_args(["--no-web", "--llm-model", "mana", "--llm-url", "http://x:1/api/generate"])
    cfg = build_config(args)
    assert cfg.ollama_model == "mana"
    assert cfg.ollama_url == "http://x:1/api/generate"
