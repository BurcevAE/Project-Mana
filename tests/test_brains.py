"""
tests/test_brains.py — the Brain Pool: selection, failover, quotas, consensus.

Everything here runs offline. `BrainPool(transport=...)` replaces the HTTP
adapters with a callable, which is what makes it possible to assert on
*routing decisions* rather than on whether some provider happened to be up
-- the previous LLM layer could only be tested by either mocking `requests`
or hitting the network, so its failover order was never actually verified.
"""
from __future__ import annotations

import time

import pytest

from mana.brains import (BrainPool, BrainSpec, answer_similarity, classify_error,
                         tier_rank)


def make_pool(config, specs, transport=None):
    """A pool containing ONLY `specs` -- the built-in catalog is cleared so
    a test never depends on which API keys happen to be in the developer's
    environment."""
    pool = BrainPool(config, transport=transport or (lambda **kw: "ok"))
    pool.brains.clear()
    pool.health.clear()
    for spec in specs:
        pool.add(spec)
    return pool


def spec(brain_id, **kw):
    kw.setdefault("provider", "openai_chat")
    kw.setdefault("model", f"model-{brain_id}")
    kw.setdefault("base_url", f"https://example.invalid/{brain_id}")
    return BrainSpec(brain_id=brain_id, **kw)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def test_hard_task_prefers_a_large_brain_over_the_small_local_one(isolated_config):
    """The core regression this module exists for: before 5.10 a hard task
    went to whatever was first in a fixed order (ollama), regardless of
    whether a stronger brain was available."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [
        spec("small-local", tier="small", local=True, strengths=("general",)),
        spec("big-remote", tier="large", strengths=("reasoning", "general")),
    ])
    chosen = pool.select(kind="reasoning", difficulty=0.9)
    assert chosen[0] == "big-remote"


def test_easy_task_prefers_the_cheap_local_brain(isolated_config):
    """The other half of the same decision: spending a scarce free-tier
    call on '2+2' is exactly the waste the router has to avoid."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [
        spec("small-local", tier="small", local=True, strengths=("general", "math")),
        spec("big-remote", tier="large", strengths=("reasoning",)),
    ])
    assert pool.select(kind="math", difficulty=0.05)[0] == "small-local"


def test_strengths_beat_tier_for_the_matching_kind(isolated_config):
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [
        spec("generalist", tier="large", strengths=("general",)),
        spec("coder", tier="large", strengths=("programming",)),
    ])
    assert pool.select(kind="programming", difficulty=0.6)[0] == "coder"


def test_paid_brains_are_excluded_unless_explicitly_allowed(isolated_config):
    """Cost is not a soft preference here: a user who never opted in must
    not discover a paid API in their billing because a router 'preferred'
    it."""
    isolated_config.enable_llm = True
    isolated_config.brain_allow_paid = False
    pool = make_pool(isolated_config, [
        spec("paid", tier="large", free=False, strengths=("reasoning",)),
        spec("free", tier="medium", free=True, strengths=("reasoning",)),
    ])
    assert pool.select(kind="reasoning", difficulty=0.9) == ["free"]
    isolated_config.brain_allow_paid = True
    assert pool.select(kind="reasoning", difficulty=0.9)[0] == "paid"


def test_no_external_brains_leaves_only_local(isolated_config):
    """--no-external-brains must be absolute, not a ranking preference."""
    isolated_config.enable_llm = True
    isolated_config.brain_external_enabled = False
    pool = make_pool(isolated_config, [
        spec("local", tier="small", local=True),
        spec("remote", tier="large"),
    ])
    assert pool.available() == ["local"]


def test_a_brain_without_its_api_key_is_simply_absent(isolated_config, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [
        spec("keyed", api_key_env="NOPE_KEY"),
        spec("open", tier="medium"),
    ])
    assert "keyed" not in pool.configured()
    assert "open" in pool.configured()


def test_round_robin_distributes_across_brains(isolated_config):
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [spec("a"), spec("b"), spec("c")])
    picks = [pool.select(policy="round_robin")[0] for _ in range(6)]
    assert len(set(picks)) == 3, f"round_robin stayed on {set(picks)}"


# ---------------------------------------------------------------------------
# failover, circuit breaking, quotas
# ---------------------------------------------------------------------------

def test_failover_moves_to_the_next_brain_and_reports_both_attempts(isolated_config):
    isolated_config.enable_llm = True

    def transport(spec, **kw):
        if spec.brain_id == "broken":
            raise RuntimeError("upstream 500")
        return "real answer"

    pool = make_pool(isolated_config, [
        spec("broken", tier="large", strengths=("reasoning",)),
        spec("working", tier="large", strengths=("reasoning",)),
    ], transport=transport)
    res = pool.ask("hard question", kind="reasoning", difficulty=0.9)
    assert res["ok"] is True
    assert res["text"] == "real answer"
    assert res["brain"] == "working"
    assert res["attempts"] == ["broken", "working"], res["attempts"]


def test_repeated_failures_trip_the_breaker_and_take_the_brain_out(isolated_config):
    isolated_config.enable_llm = True
    isolated_config.brain_failure_limit = 2
    pool = make_pool(isolated_config, [spec("flaky")],
                     transport=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    for _ in range(2):
        pool.ask_brain("flaky", "q")
    assert pool.ready("flaky") is False
    assert pool.available() == []


def test_rate_limit_uses_retry_after_and_does_not_trip_the_breaker(isolated_config):
    """A 429 means 'full', not 'broken'. Treating it as a failure would
    apply the exponential backoff meant for outages and would eventually
    disable a perfectly healthy free tier."""
    isolated_config.enable_llm = True
    isolated_config.brain_failure_limit = 2

    class Resp:
        status_code = 429
        headers = {"Retry-After": "30"}

    def transport(**kw):
        exc = RuntimeError("429 Too Many Requests")
        exc.response = Resp()
        raise exc

    pool = make_pool(isolated_config, [spec("limited")], transport=transport)
    pool.ask_brain("limited", "q")
    health = pool.health["limited"]
    assert health.rate_limited == 1
    assert health.consecutive_failures == 0, "a 429 must not count toward the breaker"
    assert 25 <= health.cooldown_until - time.time() <= 35


def test_daily_quota_removes_a_brain_when_exhausted(isolated_config):
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [spec("capped", rpd=2), spec("uncapped")])
    for _ in range(2):
        pool.ask_brain("capped", "q")
    assert pool.ready("capped") is False
    assert pool.ready("uncapped") is True


def test_minute_quota_expires_on_its_own(isolated_config):
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [spec("rpm1", rpm=1)])
    pool.ask_brain("rpm1", "q")
    assert pool.ready("rpm1") is False
    # Age the window instead of sleeping 60s.
    pool.health["rpm1"].minute_window[0] -= 61.0
    assert pool.ready("rpm1") is True


def test_pool_with_no_ready_brain_returns_a_clean_failure(isolated_config):
    """Callers treat this exactly like 'LLM unavailable' and fall back --
    it must never raise, or every call site would need a try/except."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [])
    res = pool.ask("anything")
    assert res["ok"] is False
    assert res["text"] is None
    assert res["error"] == "no brain available"


def test_outcome_feedback_changes_future_selection(isolated_config):
    """Reputation must actually move the ranking; otherwise record_outcome
    is a metric nobody reads."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [
        spec("a", tier="large", strengths=("general",)),
        spec("b", tier="large", strengths=("general",)),
    ])
    first = pool.select(kind="general", difficulty=0.8)[0]
    other = "b" if first == "a" else "a"
    for _ in range(12):
        pool.record_outcome(first, 0.0)
        pool.record_outcome(other, 1.0)
    assert pool.select(kind="general", difficulty=0.8)[0] == other


# ---------------------------------------------------------------------------
# consensus
# ---------------------------------------------------------------------------

def test_consensus_picks_the_medoid_not_the_outlier(isolated_config):
    """Two brains say 391, one says 17. The vote must not be decided by
    which answer arrived first or which is longest."""
    isolated_config.enable_llm = True
    answers = {"a": "Ответ: 391", "b": "391", "c": "Скорее всего 17, но не уверен"}
    pool = make_pool(isolated_config,
                     [spec("a"), spec("b"), spec("c")],
                     transport=lambda spec, **kw: answers[spec.brain_id])
    res = pool.ask_consensus("17*23?", n=3)
    assert res["ok"] is True
    assert "391" in res["text"]
    assert res["brain"] in {"a", "b"}
    assert res["agreement"] > 0.0


def test_consensus_flags_disagreement(isolated_config):
    isolated_config.enable_llm = True
    answers = {"a": "результат равен 391", "b": "получается 512"}
    pool = make_pool(isolated_config, [spec("a"), spec("b")],
                     transport=lambda spec, **kw: answers[spec.brain_id])
    res = pool.ask_consensus("17*23?", n=2)
    assert res["disagreement"] is True, res


def test_consensus_with_one_ready_brain_is_marked_single_not_agreed(isolated_config):
    """The failure mode this guards against: reporting agreement=1.0 after
    asking the same model once, which would be manufactured corroboration."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [spec("only")], transport=lambda **kw: "391")
    res = pool.ask_consensus("17*23?", n=3)
    assert res["single"] is True
    assert res["agreement"] == 0.0
    assert res["disagreement"] is False


def test_consensus_survives_one_brain_failing(isolated_config):
    isolated_config.enable_llm = True

    def transport(spec, **kw):
        if spec.brain_id == "dead":
            raise RuntimeError("down")
        return "391"

    pool = make_pool(isolated_config, [spec("dead"), spec("alive")], transport=transport)
    res = pool.ask_consensus("q", n=2)
    assert res["ok"] is True
    assert res["text"] == "391"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_answer_similarity_weights_numbers_over_wording():
    same_number = answer_similarity("391", "по моим расчётам получается 391")
    diff_number = answer_similarity("391", "по моим расчётам получается 512")
    assert same_number > diff_number


def test_classify_error_detects_timeouts_and_429s():
    class ReadTimeout(Exception):
        pass

    assert classify_error(ReadTimeout("read timed out"))[0] is True
    assert classify_error(RuntimeError("HTTP 429 rate limit"))[1] is True
    assert classify_error(RuntimeError("bad gateway")) == (False, False, 0.0)


def test_legacy_provider_names_still_resolve(isolated_config):
    """PipelineSpec.llm_provider values evolved before 5.10 live in users'
    state pickles; they must keep meaning something."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [
        BrainSpec(brain_id="ollama", provider="ollama", model="qwen", local=True),
        spec("openrouter-free"),
    ])
    assert pool.resolve_alias("ollama") == "ollama"
    assert pool.resolve_alias("openrouter") == "openrouter-free"
    assert pool.resolve_alias("local") == "ollama"
    assert pool.resolve_alias("nonsense") == ""


def test_difficulty_estimator_separates_trivial_from_hard():
    easy = BrainPool.estimate_difficulty("Сколько будет 17 * 23?")
    hard = BrainPool.estimate_difficulty(
        "Сравни два подхода к архитектуре агента и объясни, почему один "
        "масштабируется лучше другого, с учётом trade-off по стоимости.")
    assert hard > easy
    assert BrainPool.difficulty_to_tier(hard) == "large"
    assert BrainPool.difficulty_to_tier(easy) == "small"


def test_tier_rank_is_ordered_and_tolerates_garbage():
    assert tier_rank("small") < tier_rank("medium") < tier_rank("large")
    assert tier_rank("nonsense") == 0


# ---------------------------------------------------------------------------
# the facade the rest of MANA still calls
# ---------------------------------------------------------------------------

def test_llm_client_contract_is_unchanged(isolated_config):
    """Every call site in MANA calls ask_detailed(...) -> (text, meta).
    That contract is what let the pool be introduced without touching
    them, so it is worth a test of its own."""
    from mana.llm import LLMClient
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [spec("only")], transport=lambda **kw: "готово")
    client = LLMClient(isolated_config, pool=pool)
    text, meta = client.ask_detailed("вопрос")
    assert text == "готово"
    assert meta.ok is True
    assert meta.provider == "only" and meta.brain == "only"
    assert meta.attempts == ("only",)


def test_llm_client_is_enabled_when_only_a_remote_brain_exists(isolated_config):
    """Regression: `enabled` used to mean 'the local backend is on', so a
    machine with a free remote key and no Ollama reported llm=off and fell
    back to canned text."""
    from mana.llm import LLMClient
    isolated_config.enable_llm = False          # no local backend
    pool = make_pool(isolated_config, [spec("remote", tier="large")],
                     transport=lambda **kw: "ok")
    client = LLMClient(isolated_config, pool=pool)
    assert client.enabled is True


# ---------------------------------------------------------------------------
# independent critic (5.10.1)
# ---------------------------------------------------------------------------

def test_avoid_routes_to_a_different_brain(isolated_config):
    """`avoid` is what lets the critic be someone other than the author."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [
        spec("author", tier="large", strengths=("reasoning", "general")),
        spec("other", tier="large", strengths=("reasoning", "general")),
    ], transport=lambda spec, **kw: f"from {spec.brain_id}")
    res = pool.ask("судить черновик", kind="reasoning", avoid=["author"])
    assert res["brain"] == "other"
    assert res["avoided"] is True


def test_avoid_is_a_preference_not_a_constraint(isolated_config):
    """With one brain there is nobody else to ask. The call must still go
    through -- an unavailable second opinion is not a reason to produce no
    critique at all -- but it must report avoided=False so the weaker
    self-review is not recorded as an independent check."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [spec("only", tier="large")],
                     transport=lambda **kw: "ok")
    res = pool.ask("судить черновик", avoid=["only"])
    assert res["ok"] is True
    assert res["brain"] == "only"
    assert res["avoided"] is False


# ---------------------------------------------------------------------------
# found by the first live run against two real models (5.12.1)
# ---------------------------------------------------------------------------

def test_an_unreachable_brain_steps_aside_after_one_refusal(isolated_config):
    """With no Ollama server installed, the local brain still ranked first
    (it gets the local bonus) and was still picked for a two-brain
    consensus. The call failed, one opinion came back instead of two.
    Waiting for brain_failure_limit refusals means wasting an attempt on
    every call until then -- and silently turning N opinions into N-1."""
    isolated_config.enable_llm = True
    isolated_config.brain_failure_limit = 3

    def transport(**_kw):
        raise ConnectionError(
            "HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded")

    pool = make_pool(isolated_config, [spec("phantom", local=True)], transport=transport)
    pool.ask_brain("phantom", "q")
    assert pool.ready("phantom") is False, "one refused connection is enough evidence"


def test_a_merely_flaky_brain_still_gets_its_retries(isolated_config):
    """The other half of the same decision: a 500 is not a missing server,
    and treating it as one would drop a working brain on one bad call."""
    isolated_config.enable_llm = True
    isolated_config.brain_failure_limit = 3
    pool = make_pool(isolated_config, [spec("flaky")],
                     transport=lambda **kw: (_ for _ in ()).throw(RuntimeError("upstream 500")))
    pool.ask_brain("flaky", "q")
    assert pool.ready("flaky") is True


def test_same_verdict_different_wording_counts_as_agreement():
    """The live failure: two models both answered "Нет" to the same
    question, explained it differently, and scored 0.29 -- reported as
    disagreement. They agreed on the only thing being asked."""
    a = "Нет. Ответы языковых моделей — вероятностные предсказания, а не гарантированная истина."
    b = "Нет. Модель генерирует текст по статистике обучающих данных и не проверяет источники."
    assert answer_similarity(a, b) >= 0.8


def test_opposite_verdicts_are_not_rescued_by_shared_vocabulary():
    a = "Да. Ответ языковой модели можно считать проверенным фактом."
    b = "Нет. Ответ языковой модели нельзя считать проверенным фактом."
    assert answer_similarity(a, b) <= 0.2, "an explicit contradiction must not read as agreement"


def test_a_verdict_word_inside_an_explanation_is_not_a_verdict():
    """Only the opening counts. A 'нет' buried mid-sentence would otherwise
    make two opposite answers look identical."""
    from mana.brains import verdict_of
    assert verdict_of("Нет. Это неверно.") == "no"
    assert verdict_of("Да, безусловно.") == "yes"
    assert verdict_of("Это утверждение неверно, поэтому нет.") is None
    assert verdict_of("391") is None


def test_numeric_agreement_still_works_where_there_is_no_verdict():
    assert answer_similarity("391", "получается 391") > answer_similarity("391", "получается 512")


def test_consensus_tops_up_when_a_selected_brain_fails(isolated_config):
    """Asking for two opinions and accepting one because a brain was dead
    weakens exactly what consensus is for. If another brain is ready, ask
    it. (Live run: the local brain had no server behind it, and a
    two-brain consensus silently became `single`.)"""
    isolated_config.enable_llm = True

    def transport(spec, **kw):
        if spec.brain_id == "dead":
            raise ConnectionError("Max retries exceeded")
        return f"Нет. Объяснение от {spec.brain_id}."

    pool = make_pool(isolated_config, [
        spec("dead", tier="large", local=True),
        spec("live-a", tier="large"),
        spec("live-b", tier="large"),
    ], transport=transport)
    res = pool.ask_consensus("вопрос", n=2)
    assert res["single"] is False, "a ready third brain should have been asked"
    assert len(res["brains"]) == 2
    assert "dead" not in res["brains"]


def test_top_up_stops_when_the_pool_is_exhausted(isolated_config):
    """The loop must terminate on a pool where nothing works, rather than
    hunting for a second opinion that does not exist."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, [spec("a"), spec("b")],
                     transport=lambda **kw: (_ for _ in ()).throw(ConnectionError("refused")))
    res = pool.ask_consensus("вопрос", n=2)
    assert res["ok"] is False
    assert len(res["responses"]) <= 4
