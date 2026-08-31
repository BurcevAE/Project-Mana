"""
tests/test_llm_dependent.py — tests that need a REAL, reachable LLM backend
(Ollama by default; set MANA_LLM_BACKEND/MANA_OLLAMA_MODEL env vars to
point elsewhere). Skipped unless MANA_TEST_LLM=1 is set, because nothing
in the sandbox this suite was authored in has a live LLM -- these are
UNVERIFIED by me. Run them on your machine and send me the output.

Every test here that could mutate real files (self_improve_code) rolls
back in a finally block, so a failed assertion never leaves your
checkout modified.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MANA_TEST_LLM") != "1",
    reason="set MANA_TEST_LLM=1 (and have a real LLM backend reachable) to run these",
)


@pytest.fixture
def isolated_agent_llm(isolated_config):
    """Same isolation as isolated_agent, but with the LLM turned on --
    uses whatever mana.config.Config defaults to (Ollama at
    http://localhost:11434 by default; override via env vars your Config
    already reads, e.g. set config.ollama_model directly below if needed)."""
    isolated_config.enable_llm = True
    from mana import ManaAgent
    agent = ManaAgent(isolated_config)
    yield agent
    try:
        agent.persistent_memory.close()
    except Exception:
        pass
    try:
        agent.experience.close()
    except Exception:
        pass


def test_llm_generate_tool_returns_real_text(isolated_agent_llm):
    assert isolated_agent_llm._tool_available("llm_generate"), (
        "LLM tool reports unavailable -- is Ollama actually running and "
        "reachable at config.ollama_url? Check `ollama list` / `ollama serve`."
    )
    result = isolated_agent_llm.tools.call("llm_generate", prompt="Скажи одно слово: тест.",
                                            temperature=0.0, context_tag="DIAG")
    print("LLM_GENERATE_OUTPUT:", repr(result.output))
    print("LLM_GENERATE_META:", result.meta)
    assert result.ok is True
    assert result.output


def test_full_adaptive_answer_with_real_llm(isolated_agent_llm):
    from dataclasses import asdict
    from mana.pipeline import PipelineSpec
    spec = PipelineSpec(**asdict(isolated_agent_llm.pipeline))
    spec.route_mode = "auto"
    spec = spec.normalize(isolated_agent_llm.config)
    result = isolated_agent_llm.answer("Объясни в одном предложении, что такое рекурсия.",
                                        spec=spec, save_memory=False, context_tag="DIAG")
    print("ANSWER:", result["answer"])
    print("ADAPTIVE_TRACE:", result.get("adaptive"))
    assert result["llm_ok"] is True
    assert result["fallback"] is False


def test_critic_repair_loop_with_real_llm(isolated_agent_llm):
    from mana.pipeline import PipelineSpec
    spec = PipelineSpec(use_critic=True, critic_threshold=0.9).normalize(isolated_agent_llm.config)
    repaired, score, trace = isolated_agent_llm._critic(
        "Сколько будет 2+2?", "5", spec, "DIAG")
    print("CRITIC_REPAIRED_ANSWER:", repaired)
    print("CRITIC_SCORE:", score, "TRACE:", trace)
    assert trace["called"] is True


def test_graph_memory_llm_distillation_quality(isolated_agent_llm):
    """Compares the LLM-based distiller against the heuristic fallback on
    the same rambling input -- prints both so you can eyeball the
    difference; no hard quality assertion since 'better' is subjective."""
    from mana import graph_memory as gm
    rambling = ("Ну короче слушай, я тут подумал, в общем-то это важно: "
                "надо не забыть, что дедлайн по проекту переносится на пятницу, "
                "и ещё нужно созвониться с командой насчёт бюджета.")
    heuristic = gm.extractive_distill(rambling)
    llm_based = gm.distill_turn("", rambling, llm_ask=isolated_agent_llm._llm_ask_plain)
    print("HEURISTIC DISTILL:", heuristic)
    print("LLM DISTILL:", llm_based)
    assert llm_based


def test_self_improve_code_with_real_llm_proposal_always_rolls_back(isolated_agent_llm):
    """CRITICAL WARNING: self_improve_code's apply_patch() writes to the
    REAL installed package (mana/agent_parts/routing.py) and the REAL
    mana_code_history/ next to it -- code_evolution.py's paths are NOT
    redirectable via Config (see its module docstring), so the
    isolated_agent_llm fixture does NOT sandbox this call the way it
    sandboxes everything else. That's why this test needs its own opt-in
    flag on top of MANA_TEST_LLM, and why the rollback below is
    unconditional (runs even if the assertion fails)."""
    if os.environ.get("MANA_TEST_LLM_CODE_EVOLUTION") != "1":
        pytest.skip("set MANA_TEST_LLM_CODE_EVOLUTION=1 to run this -- it can write to your real "
                    "mana/agent_parts/ files (with rollback), see the docstring above")
    isolated_agent_llm.config.local_exec_enabled = True
    history_before = isolated_agent_llm.code_history("task_category")
    try:
        report = isolated_agent_llm.self_improve_code(
            "task_category", "improve accuracy on ambiguous cases without breaking existing ones")
        print("SELF_IMPROVE_CODE_REPORT:", {k: v for k, v in report.items() if k != "candidate_source"})
        print("PROPOSED_CANDIDATE_SOURCE:\n", report.get("candidate_source"))
        assert "decision" in report
    finally:
        if isolated_agent_llm.code_history("task_category") != history_before:
            latest = isolated_agent_llm.code_history("task_category")[-1]
            rollback_result = isolated_agent_llm.rollback_code("task_category", latest["backup"])
            print("ROLLED BACK:", rollback_result)
