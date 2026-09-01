"""
tests/test_temporal_grounding.py — audit #60, approached as a missing
INPUT rather than as a new detector module.

Observed failure: asked for "последние новости про ИИ", MANA answered
"новости за последние два дня" directly above a snippet dated 18 June,
while the actual date was 31 August. The critic did not catch it.

The tempting fix is a freshness-detector module, or asking the model
"is this stale?". Both are wrong for the same reason: the model was never
told what day it is. A judgement about time cannot follow from data that
was never supplied -- asking for more self-assessment would just produce
a confident guess, which is precisely the MODEL_TESTED failure mode
established in test_verification_trust.py.

So these tests assert that the REASONER IS GIVEN THE FACTS: today's date
in both the answer and critic prompts, and a retrieval timestamp plus an
honest "this is a snippet, not page content" marker on web evidence. What
the model then concludes is still only MODEL_TESTED -- see
test_critic_verdict_is_still_not_proof below.
"""
from __future__ import annotations

import time
from dataclasses import asdict

from mana.agent_parts.execution import ExecutionMixin
from mana.pipeline import PipelineSpec


def _today() -> str:
    return time.strftime("%Y-%m-%d")


# --- the answer prompt knows what day it is ------------------------------

def test_answer_prompt_states_todays_date(isolated_agent):
    spec = PipelineSpec(**asdict(isolated_agent.pipeline)).normalize(isolated_agent.config)
    prompt = isolated_agent._compose_prompt("какие последние новости про ИИ", "", spec)
    assert _today() in prompt, "the model has no clock unless we give it one"


def test_answer_prompt_forbids_unsupported_freshness_claims(isolated_agent):
    spec = PipelineSpec(**asdict(isolated_agent.pipeline)).normalize(isolated_agent.config)
    prompt = isolated_agent._compose_prompt("новости", "", spec)
    lowered = prompt.lower()
    assert "свеж" in lowered and "актуальн" in lowered, (
        "prompt must tell the model not to call information current without a date")


# --- the critic knows what day it is too ---------------------------------

def test_critic_prompt_states_todays_date_and_checks_time_claims(isolated_agent, monkeypatch):
    from mana import ManaAgent
    from mana.llm import LLMCallMeta
    captured = {}

    def fake_llm_call(self, prompt, **kwargs):
        captured["prompt"] = prompt
        return "SCORE: 0.9", LLMCallMeta(ok=True)

    monkeypatch.setattr(ManaAgent, "_tool_available", lambda self, name: True)
    monkeypatch.setattr(ManaAgent, "_llm_call", fake_llm_call)

    spec = PipelineSpec(**asdict(isolated_agent.pipeline))
    spec.use_critic = True
    spec = spec.normalize(isolated_agent.config)
    isolated_agent._critic("какие последние новости", "Вот свежие новости за два дня", spec, "T")

    assert _today() in captured["prompt"], "the critic cannot judge staleness without a reference point"
    assert "свеж" in captured["prompt"].lower()


# --- web evidence carries its own timestamp and honest provenance --------

def test_web_context_block_is_timestamped_and_labelled_as_snippet(isolated_config, monkeypatch):
    from mana import ManaAgent, web as web_mod

    isolated_config.enable_web = True
    monkeypatch.setattr(web_mod, "HAS_WEB", True)

    class FakeDDGS:
        def text(self, *a, **k):
            return [{"title": "Заголовок", "body": "Текст сниппета", "href": "https://example.com"}]

    monkeypatch.setattr(web_mod, "DDGS", FakeDDGS)
    agent = ManaAgent(isolated_config)

    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.web_mode = "always"
    spec.use_web = True
    spec.web_results = 1
    spec = spec.normalize(agent.config)

    context, trace = agent._build_context("какие последние новости про ИИ", spec)
    assert "[WEB" in context, f"no web block produced; trace={trace}"
    assert _today() in context, "web evidence must record WHEN it was fetched"
    assert "сниппет" in context, (
        "the model must be told this is a search-result snippet, not the page itself")
    assert trace.get("web_retrieved_at"), "retrieval time must be in the trace too"


# --- and none of this promotes the critic to an oracle -------------------

def test_critic_verdict_is_still_not_proof():
    """Guard the boundary this change must NOT cross: giving the critic
    better inputs makes its judgement better-founded, not authoritative.
    A model reviewing an answer stays MODEL_TESTED."""
    verdict = {"kind": "code", "mode": "generated_tests", "verified": True, "ok": True}
    assert ExecutionMixin.verification_trust_level(verdict) == ExecutionMixin.TRUST_MODEL_TESTED
    assert (ExecutionMixin.TRUST_QUALITY_CAP[ExecutionMixin.TRUST_MODEL_TESTED]
            < ExecutionMixin.TRUST_QUALITY_CAP[ExecutionMixin.TRUST_INDEPENDENTLY_VERIFIED])


def test_prompt_forbids_leaking_evidence_labels_into_the_answer(isolated_agent):
    """Observed on real hardware: the model emitted "USER CLAIM:",
    "SOURCE EVIDENCE:" and "CONCLUSION:" as literal headings to the user.
    Those labels are internal context markup; the prompt described them
    but never said they are not an output format, and a 7B model followed
    the description literally."""
    from dataclasses import asdict

    from mana.pipeline import PipelineSpec

    spec = PipelineSpec(**asdict(isolated_agent.pipeline)).normalize(isolated_agent.config)
    prompt = isolated_agent._compose_prompt("какие новости", "", spec)
    assert "служебная разметка" in prompt
    assert "НЕ формат ответа" in prompt


def test_prompt_says_a_snippet_date_describes_only_the_snippet(isolated_agent):
    """Observed on real hardware: MANA insisted a site had nothing newer
    than four days, citing a search snippet -- while the user was looking
    at articles published that day. The snippet's date says nothing about
    the site; the page was never opened."""
    from dataclasses import asdict

    from mana.pipeline import PipelineSpec

    spec = PipelineSpec(**asdict(isolated_agent.pipeline)).normalize(isolated_agent.config)
    prompt = isolated_agent._compose_prompt("какие новости", "", spec)
    assert "только сам сниппет" in prompt.lower()
    assert "страницу ты не открывал" in prompt.lower()
