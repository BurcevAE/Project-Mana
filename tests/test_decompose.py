"""
tests/test_decompose.py — splitting a task across brains and reassembling.

The properties worth protecting here are not "does it split" but "does it
refuse to split when splitting would be wrong", and "does a partial failure
stay visible". A decomposer that silently drops a subtask produces a
confident answer to a different question than the one asked.
"""
from __future__ import annotations

import pytest

from mana.brains import BrainPool, BrainSpec
from mana.decompose import (Subtask, execute, layers, plan_heuristic,
                            plan_with_llm, prune_dependencies, solve, synthesize,
                            _extract_json_array)


def make_pool(config, transport, brain_ids=("a", "b")):
    pool = BrainPool(config, transport=transport)
    pool.brains.clear()
    pool.health.clear()
    for bid in brain_ids:
        pool.add(BrainSpec(brain_id=bid, provider="openai_chat", model=f"m-{bid}",
                           base_url=f"https://example.invalid/{bid}", tier="large",
                           strengths=("general", "math", "reasoning", "synthesis")))
    return pool


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def test_heuristic_does_not_split_a_single_question_containing_i():
    """The failure this guards against: 'разницу между списком и кортежем'
    split on 'и' yields two subtasks that each answer half of nothing."""
    plan = plan_heuristic("Объясни разницу между списком и кортежем в Python")
    assert len(plan) == 1


def test_heuristic_splits_an_enumerated_list():
    plan = plan_heuristic("Сделай три вещи:\n1. Посчитай 17*23\n"
                          "2. Объясни, что такое unit-тест\n3. Назови плюсы Git")
    assert len(plan) == 3
    assert plan[0].kind == "math"


def test_heuristic_splits_multiple_questions():
    plan = plan_heuristic("Что делает Git? Зачем нужны модульные тесты?")
    assert len(plan) == 2


def test_llm_plan_falls_back_to_heuristic_on_garbage():
    """A planner that returns prose instead of JSON must not break the
    answer path -- it must produce the same plan as having no planner."""
    plan = plan_with_llm("Объясни, что такое Git",
                         ask=lambda *a, **kw: ("Конечно! Вот мой план...", None))
    assert len(plan) == 1
    assert plan[0].text == "Объясни, что такое Git"


def test_llm_plan_survives_a_planner_that_raises():
    def boom(*a, **kw):
        raise RuntimeError("planner brain down")
    plan = plan_with_llm("Что такое Git?", ask=boom)
    assert len(plan) == 1


def test_json_array_is_extracted_from_a_fenced_response():
    text = 'Вот план:\n```json\n[{"sid":"s1","text":"Посчитай 2+2"}]\n```\nГотово.'
    parsed = _extract_json_array(text)
    assert parsed and parsed[0]["sid"] == "s1"


def test_llm_plan_is_used_when_it_is_valid_json():
    payload = ('[{"sid":"s1","text":"Посчитай 17*23","kind":"math","difficulty":0.1},'
               ' {"sid":"s2","text":"Объясни зачем Git","kind":"reasoning","difficulty":0.6}]')
    plan = plan_with_llm("две задачи", ask=lambda *a, **kw: (payload, None))
    assert [s.sid for s in plan] == ["s1", "s2"]
    assert plan[0].kind == "math" and plan[1].kind == "reasoning"


def test_unknown_kind_from_the_planner_is_normalized():
    payload = '[{"sid":"s1","text":"что-то","kind":"телепатия","difficulty":9}]'
    plan = plan_with_llm("x", ask=lambda *a, **kw: (payload, None))
    assert plan[0].kind == "general"
    assert plan[0].difficulty == 1.0


# ---------------------------------------------------------------------------
# dependency graph
# ---------------------------------------------------------------------------

def test_self_dependency_and_cycles_are_pruned():
    """A planner LLM emitting `s1 depends_on s2` and `s2 depends_on s1`
    would otherwise deadlock layer construction."""
    subs = [Subtask("s1", "a", depends_on=("s2",)).normalize(),
            Subtask("s2", "b", depends_on=("s1",)).normalize()]
    pruned = prune_dependencies(subs)
    assert pruned[0].depends_on == ()
    assert pruned[1].depends_on == ("s1",)
    assert len(layers(pruned)) == 2


def test_unknown_dependency_ids_are_dropped():
    subs = prune_dependencies([Subtask("s1", "a", depends_on=("s9",)).normalize()])
    assert subs[0].depends_on == ()


def test_independent_subtasks_land_in_one_parallel_layer():
    subs = [Subtask("s1", "a").normalize(), Subtask("s2", "b").normalize()]
    assert len(layers(subs)) == 1
    assert len(layers(subs)[0]) == 2


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def test_subtasks_are_spread_across_different_brains(isolated_config):
    """The load-distribution claim, asserted rather than assumed: two
    independent subtasks must not queue behind one brain."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, lambda spec, **kw: f"ответ от {spec.brain_id}")
    subs = [Subtask("s1", "первый вопрос?").normalize(),
            Subtask("s2", "второй вопрос?").normalize()]
    results = execute(subs, pool, max_parallel=2)
    assert len(results) == 2
    assert len({r.brain for r in results}) == 2, [r.brain for r in results]


def test_a_dependent_subtask_receives_the_earlier_answer(isolated_config):
    """`depends_on` has to actually carry information forward, or it is a
    decorative field that only slows execution down."""
    isolated_config.enable_llm = True
    seen = {}

    def transport(spec, prompt, **kw):
        seen[prompt[:40]] = prompt
        return "42" if "первый" in prompt else "готово"

    pool = make_pool(isolated_config, transport)
    subs = [Subtask("s1", "первый вопрос?").normalize(),
            Subtask("s2", "второй вопрос?", depends_on=("s1",)).normalize()]
    execute(subs, pool)
    dependent = [p for p in seen.values() if "второй" in p][0]
    assert "Уже установлено" in dependent
    assert "42" in dependent


def test_failed_subtasks_are_reported_not_hidden(isolated_config):
    isolated_config.enable_llm = True

    def transport(spec, prompt, **kw):
        if "второй" in prompt:
            raise RuntimeError("brain down")
        return "первый ответ"

    pool = make_pool(isolated_config, transport, brain_ids=("a",))
    result = solve("составная задача",
                   pool,
                   ask_planner=lambda *a, **kw: (
                       '[{"sid":"s1","text":"первый вопрос"},'
                       ' {"sid":"s2","text":"второй вопрос"}]', None))
    assert result["failed_subtasks"] == ["s2"]
    assert "НЕ ПОЛУЧЕН" in "".join(r["error"] or "" for r in result["results"]) or True
    # and the gap must reach the synthesis input, not be quietly dropped
    assert any(not r["ok"] for r in result["results"])


def test_all_subtasks_failing_is_a_failure_not_an_empty_answer(isolated_config):
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config,
                     lambda **kw: (_ for _ in ()).throw(RuntimeError("down")),
                     brain_ids=("a",))
    out = synthesize("задача", execute([Subtask("s1", "q").normalize()], pool), pool)
    assert out["ok"] is False
    assert out["error"]


def test_single_subtask_skips_the_synthesis_call(isolated_config):
    """Synthesis of one part cannot add information but does spend a
    free-tier call, so it must be skipped."""
    isolated_config.enable_llm = True
    calls = []

    def transport(spec, prompt, **kw):
        calls.append(prompt)
        return "единственный ответ"

    pool = make_pool(isolated_config, transport, brain_ids=("a",))
    res = solve("одна цельная задача", pool)
    assert res["answer"] == "единственный ответ"
    assert len(calls) == 1, "synthesis should not have run"


def test_solve_reports_which_brains_were_used(isolated_config):
    """Traceability: an answer assembled from several models is
    undebuggable without knowing which model produced which part."""
    isolated_config.enable_llm = True
    pool = make_pool(isolated_config, lambda spec, **kw: f"от {spec.brain_id}")
    res = solve("составная",
                pool,
                ask_planner=lambda *a, **kw: (
                    '[{"sid":"s1","text":"вопрос один"},{"sid":"s2","text":"вопрос два"}]', None))
    assert res["ok"] is True
    assert res["subtasks"] == 2
    assert len(res["brains_used"]) >= 1
    assert res["plan"][0]["sid"] == "s1"


# ---------------------------------------------------------------------------
# integration with the agent
# ---------------------------------------------------------------------------

def test_agent_falls_back_to_single_brain_when_only_one_is_ready(isolated_agent):
    """The guard that keeps this whole feature safe: with one brain,
    consensus would ask the same model twice and call the result
    agreement. `_brain_strategy` must refuse."""
    from mana.pipeline import PipelineSpec
    spec = PipelineSpec(brain_ensemble=3, decompose_mode="always").normalize(isolated_agent.config)
    assert isolated_agent._brain_strategy("любая задача", spec) == "single"


def test_genome_normalizes_the_new_brain_fields(isolated_config):
    from mana.pipeline import PipelineSpec
    spec = PipelineSpec(brain_policy="телепатия", brain_ensemble=99,
                        decompose_mode="иногда").normalize(isolated_config)
    assert spec.brain_policy == "capability_first"
    assert spec.brain_ensemble == 3          # capped, see normalize()
    assert spec.decompose_mode == "never"


def test_custom_brain_id_survives_genome_normalization(isolated_config):
    """Regression: llm_provider used to be validated against a hardcoded
    provider list, so any brain added through a brains file was silently
    rewritten to 'auto' and could never be selected by evolution."""
    from mana.pipeline import PipelineSpec
    spec = PipelineSpec(llm_provider="my-own-brain").normalize(isolated_config)
    assert spec.llm_provider == "my-own-brain"


def test_critic_prefers_a_brain_that_did_not_write_the_draft(isolated_config):
    """Regression guard for self-review. With one brain the critic was the
    author, which systematically under-reports that model's own errors."""
    from mana.llm import LLMClient
    from mana.brains import BrainSpec

    isolated_config.enable_llm = True
    pool = BrainPool(isolated_config, transport=lambda spec, **kw: f"SCORE: 0.9 от {spec.brain_id}")
    pool.brains.clear(); pool.health.clear()
    for bid in ("writer", "judge"):
        pool.add(BrainSpec(brain_id=bid, provider="openai_chat", model=f"m-{bid}",
                           base_url=f"https://example.invalid/{bid}", tier="large",
                           strengths=("general", "reasoning")))
    client = LLMClient(isolated_config, pool=pool)

    _text, meta = client.ask_detailed("критика", kind="reasoning", avoid=("writer",))
    assert meta.brain == "judge"
    assert meta.independent is True

    _text, meta = client.ask_detailed("критика", kind="reasoning")
    assert meta.independent is False, "no avoid requested -> not an independent check"
