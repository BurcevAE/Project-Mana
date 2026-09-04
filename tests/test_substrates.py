"""
tests/test_substrates.py — the phase where `Brain` stops meaning `LLM`.

The audit found the cheap substrate already built and in the wrong
place: exact arithmetic and a code sandbox lived in `tools`, a hierarchy
the brain router cannot see, so the cheapest possible answer to
"4821 * 37 + 145" was unreachable from the thing whose job is choosing
how to answer.

Most of these tests are about refusal, because refusal is what makes an
algorithmic brain safe to put first in the cascade.
"""
from __future__ import annotations

import pytest

from mana.brains import BrainHealth, BrainPool, BrainSpec, SUBSTRATE_PREFERENCE
from mana.core import cost as cost_units
from mana.substrates import BrainRefusal, arithmetic_answer, code_output
from mana.verifier import LocalVerifier


# ---------------------------------------------------------------------------
# exact, or nothing
# ---------------------------------------------------------------------------

def test_arithmetic_is_answered_exactly():
    assert arithmetic_answer("Вычисли: 4821 * 37 + 145") == "178522"
    assert arithmetic_answer("Сколько будет 17 * 23?") == "391"


def test_big_integers_stay_exact():
    """The reason this is a brain and not a prompt: no float, no rounding,
    no plausible-looking wrong digit in the middle."""
    assert arithmetic_answer("Вычисли: 99999999999 * 88888888888") == str(
        99999999999 * 88888888888)


def test_a_task_with_no_arithmetic_is_refused():
    with pytest.raises(BrainRefusal):
        arithmetic_answer("Объясни, почему небо синее")


def test_an_expression_that_cannot_be_evaluated_is_refused():
    """Parsed as arithmetic and did not evaluate. The brain has nothing
    exact to say, and the next substrate may."""
    with pytest.raises(BrainRefusal):
        arithmetic_answer("Вычисли: 5 / 0")


def test_the_brain_and_the_verifier_use_one_parser():
    """Two parsers for one job disagree eventually, and the day they do,
    the verifier and the brain contradict each other about the same
    expression -- which is what the verifier exists to prevent."""
    task = "Вычисли: (91767 - 690) * 86 + 8"
    expression = LocalVerifier.extract_expression(task)
    assert expression
    assert arithmetic_answer(task) == str(LocalVerifier.evaluate_expression(
        expression)["value"])


def test_code_execution_refuses_a_prompt_with_no_code():
    with pytest.raises(BrainRefusal):
        code_output("напиши функцию сортировки")


def test_code_execution_refuses_without_a_sandbox():
    with pytest.raises(BrainRefusal):
        code_output("```python\nprint(1)\n```", verifier=None)


# ---------------------------------------------------------------------------
# the pool can call something that is not a model
# ---------------------------------------------------------------------------

def test_an_algorithmic_brain_works_with_no_llm_and_no_keys(isolated_config):
    """The property that proves Brain is not LLM: no key, no network, no
    `enable_llm`, and an exact answer."""
    isolated_config.enable_llm = False
    pool = BrainPool(isolated_config)
    assert "arithmetic" in pool.available()
    result = pool.ask("Вычисли: 4821 * 37 + 145", kind="math")
    assert result["ok"] is True
    assert result["text"] == "178522"
    assert result["brain"] == "arithmetic"
    assert result["substrate"] == cost_units.ALGORITHMIC


def test_a_test_transport_does_not_answer_for_an_algorithmic_brain(isolated_config):
    """An injected transport stands in for a network, which an
    algorithmic brain does not use. Letting the double answer would test
    the double."""
    isolated_config.enable_llm = False
    pool = BrainPool(isolated_config, transport=lambda **kw: "подделка")
    assert pool.ask_brain("arithmetic", "Вычисли: 2 + 2")["text"] == "4"


def test_an_algorithmic_call_costs_no_tokens_and_they_are_measured(isolated_config):
    """Genuinely zero, not unmeasured: no tokens exist."""
    isolated_config.enable_llm = False
    pool = BrainPool(isolated_config)
    cost = pool.ask_brain("arithmetic", "Вычисли: 2 + 2")["cost"]
    assert cost["tokens_in"] == 0 and cost["tokens_out"] == 0
    assert cost["unmeasured_token_calls"] == 0
    assert cost["by_substrate"] == {"algorithmic": 1}


# ---------------------------------------------------------------------------
# refusal is not failure
# ---------------------------------------------------------------------------

def test_a_refusal_is_marked_as_one(isolated_config):
    isolated_config.enable_llm = False
    pool = BrainPool(isolated_config)
    result = pool.ask_brain("arithmetic", "Объясни, почему небо синее")
    assert result["ok"] is False
    assert result["refused"] is True


def test_a_refusal_does_not_damage_the_brains_health(isolated_config):
    """A brain that declines what it cannot answer exactly is working
    correctly. Recording that as a fault pushes a healthy brain into
    cooldown for doing its job, after which the cheap substrate stops
    being offered at all."""
    isolated_config.enable_llm = False
    pool = BrainPool(isolated_config)
    for _ in range(10):
        pool.ask_brain("arithmetic", "Объясни, почему небо синее")
    health = pool.health["arithmetic"]
    assert health.failures == 0
    assert health.consecutive_failures == 0
    assert health.cooldown_until == 0.0
    assert pool.ready("arithmetic") is True


def test_a_real_failure_still_damages_health(isolated_config):
    """The distinction has to cut both ways, or it is just a way of
    hiding faults."""
    isolated_config.enable_llm = True
    pool = BrainPool(isolated_config, transport=lambda **kw: (_ for _ in ()).throw(
        RuntimeError("boom")))
    pool.brains = {"m": BrainSpec("m", "openai_chat", "x")}
    pool.health = {"m": BrainHealth()}
    pool.ask_brain("m", "вопрос")
    assert pool.health["m"].failures == 1


# ---------------------------------------------------------------------------
# the cascade
# ---------------------------------------------------------------------------

def test_cheaper_machinery_outranks_dearer_machinery(isolated_config):
    assert (SUBSTRATE_PREFERENCE[cost_units.ALGORITHMIC] >
            SUBSTRATE_PREFERENCE[cost_units.SMALL_NEURAL] >
            SUBSTRATE_PREFERENCE[cost_units.REMOTE_LLM])


def test_arithmetic_goes_to_the_algorithmic_brain_even_with_models_present(
        isolated_config, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    isolated_config.enable_llm = True
    pool = BrainPool(isolated_config, transport=lambda **kw: "модель ответила 42")
    assert pool.language_models(), "the test needs a model present to be meaningful"
    result = pool.ask("Вычисли: 4821 * 37 + 145", kind="math")
    assert result["brain"] == "arithmetic"
    assert result["text"] == "178522"


def test_a_refusal_falls_through_to_the_next_substrate(isolated_config, monkeypatch):
    """The cascade, end to end: the cheap brain declines and the model
    answers, without anyone writing an if-statement about it."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    isolated_config.enable_llm = True
    pool = BrainPool(isolated_config, transport=lambda **kw: "небо синее из-за рассеяния")
    result = pool.ask("Объясни, почему небо синее", kind="reasoning")
    assert result["ok"] is True
    assert result["brain"] not in ("arithmetic", "code-exec")


def test_the_attempts_record_every_level_of_the_cascade(isolated_config, monkeypatch):
    """What a caller needs in order to write one observation per level
    into the self-model: the levels have to be in the result."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    isolated_config.enable_llm = True
    pool = BrainPool(isolated_config, transport=lambda **kw: "ответ")
    result = pool.ask("Объясни, почему небо синее", kind="reasoning")
    assert len(result["attempts"]) >= 1
    assert result["attempts"][-1] == result["brain"]


# ---------------------------------------------------------------------------
# a brain is not a language model
# ---------------------------------------------------------------------------

def test_an_algorithmic_brain_is_not_counted_as_a_language_model(isolated_config):
    """`llm_generate` needs something that will attempt anything. Before
    this distinction existed, an AST arithmetic evaluator made it
    advertise itself on a machine with no model at all."""
    isolated_config.enable_llm = False
    pool = BrainPool(isolated_config)
    assert "arithmetic" in pool.configured()
    assert "arithmetic" not in pool.language_models()


def test_the_llm_client_is_disabled_when_only_algorithmic_brains_exist(isolated_config):
    from mana.llm import LLMClient
    isolated_config.enable_llm = False
    pool = BrainPool(isolated_config)
    assert pool.available()
    assert LLMClient(isolated_config, pool=pool).enabled is False


def test_an_llm_brain_declares_its_substrate_without_being_told(isolated_config):
    assert BrainSpec("a", "ollama", "m", local=True).substrate == cost_units.LOCAL_LLM
    assert BrainSpec("b", "openai_chat", "m").substrate == cost_units.REMOTE_LLM


# ---------------------------------------------------------------------------
# the defect the prototype found: an exact answer to a fragment
# ---------------------------------------------------------------------------

PARENTHESISED = [
    ("Вычисли: (987 + 33) * 11", 11220),
    ("Вычисли: (61681 - 536) * 99 + 8", (61681 - 536) * 99 + 8),
    ("Вычисли: (66336 + 557 * 71) * 8", (66336 + 557 * 71) * 8),
    ("Вычисли: (19630 - 953) * 62 + 7", (19630 - 953) * 62 + 7),
]


@pytest.mark.parametrize("task,expected", PARENTHESISED)
def test_a_parenthesised_expression_is_taken_whole(task, expected):
    """The extractor used to stop at the colon, fall through to a pattern
    with no parentheses, and capture `987 + 33` out of
    `(987 + 33) * 11` -- answering 1020 with no uncertainty attached.

    An exact answer to a fragment is worse than a model's guess: nothing
    downstream doubts it.
    """
    assert arithmetic_answer(task) == str(expected)


@pytest.mark.parametrize("task,expected", PARENTHESISED)
def test_the_verifier_had_the_truth_inverted_on_these(task, expected):
    """The same extraction backs `verify`, so the arithmetic verifier --
    the thing that stamps INDEPENDENTLY_VERIFIED -- called the correct
    answer wrong and the wrong answer verified. On (987 + 33) * 11 the
    correction machinery would have rewritten 11220 into 1020.
    """
    from mana import Config
    from mana.verifier import LocalVerifier
    verifier = LocalVerifier(Config())
    right = verifier.verify(task, str(expected), "math")
    assert right["value"] == expected
    assert right["verified"] is True
    wrong = verifier.verify(task, "1020", "math")
    assert wrong["verified"] is False


def test_an_unbalanced_expression_is_refused_not_repaired():
    """"(2 + 3" could be a truncated expression or a typo, and guessing
    which turns a parse failure into a wrong answer."""
    with pytest.raises(BrainRefusal):
        arithmetic_answer("Вычисли: (2 + 3")


def test_a_bare_number_is_not_an_expression():
    """Otherwise every task with a number in its wording gets claimed by
    the arithmetic path."""
    with pytest.raises(BrainRefusal):
        arithmetic_answer("Назови 5 столиц Европы")


def test_the_longest_balanced_run_wins():
    """Generated tasks carry a trailing instruction line; the expression
    must not be lost to a fragment elsewhere in the prompt."""
    assert arithmetic_answer(
        "Вычисли: (12 + 8) * 3\nОтветь одним целым числом, без пояснений.") == "60"


def test_exactness_holds_across_the_whole_generator(isolated_config):
    """The claim an algorithmic brain lives or dies by, checked against
    ground truth rather than asserted: on generated arithmetic it is
    either exactly right or it refuses. Never confidently wrong."""
    from mana.core import oracle, tasks as core_tasks
    wrong = []
    for task in core_tasks.generate("arithmetic", 60, seed=11,
                                    difficulty_range=(0.0, 1.01)):
        try:
            answer = arithmetic_answer(task.prompt)
        except BrainRefusal:
            continue
        if not oracle.grade(task, answer).correct:
            wrong.append((task.prompt.splitlines()[0], answer, task.answer))
    assert not wrong, f"confidently wrong on {len(wrong)}: {wrong[:3]}"


# ---------------------------------------------------------------------------
# generalisation is relative to the surfaces tested
# ---------------------------------------------------------------------------

def _score(brain, domain, surface, n=40, seed=77):
    from mana.core import oracle, tasks
    right = wrong = refused = 0
    for task in tasks.generate(domain, n, seed=seed, surface=surface):
        try:
            answer = arithmetic_answer(task.prompt) if brain == "arithmetic" else None
        except BrainRefusal:
            refused += 1
            continue
        if answer is None:
            from mana import substrates
            try:
                answer = substrates.call(brain, task.prompt)
            except BrainRefusal:
                refused += 1
                continue
        if oracle.grade(task, answer).correct:
            right += 1
        else:
            wrong += 1
    return right, wrong, refused


def test_the_solvers_never_answer_wrongly_on_any_surface():
    """The property that makes every other failure here safe: when a
    wording is unrecognised they go silent, and silence falls through to
    the model. A wrong answer would not."""
    from mana.core import tasks
    for brain, domain in (("arithmetic", "arithmetic"),
                          ("sequence-solver", "sequence"),
                          ("text-ops", "text_ops"),
                          ("order-logic", "logic")):
        for surface in tasks.SURFACES:
            _right, wrong, _refused = _score(brain, domain, surface, n=20)
            assert wrong == 0, f"{brain} answered wrongly on {surface}"


def test_arithmetic_and_sequence_survive_a_surface_written_after_them():
    """The honest test: the stress surface was authored AFTER these
    solvers were rewritten structurally, and they were not touched
    afterwards. A surface written before the code it tests can shape that
    code, and the measurement is then of a fit rather than of a
    generalisation."""
    from mana.core import tasks
    assert _score("arithmetic", "arithmetic", tasks.STRESS)[0] == 40
    assert _score("sequence-solver", "sequence", tasks.STRESS)[0] >= 30


def test_text_ops_and_logic_assume_a_layout_and_that_is_recorded():
    """They survived REWORDING and still fail a change of LAYOUT: the
    ordering solver reads one constraint per line, and the text solver
    wants a label ending in a colon. The stress surface puts every
    constraint on one line and separates the label with a dash.

    Left failing on purpose. Fixing them against a surface I have already
    seen would be the same fitting the last two phases were about, and
    the honest reading is that "generalises" is bounded by the surfaces
    tested rather than being a property a solver simply has.
    """
    from mana.core import tasks
    right, wrong, refused = _score("order-logic", "logic", tasks.STRESS)
    assert right == 0 and wrong == 0 and refused == 40
    right, wrong, refused = _score("text-ops", "text_ops", tasks.STRESS)
    assert right == 0 and wrong == 0 and refused == 40


def test_every_surface_asks_the_same_question():
    """A variant that changed the answer would be a harder test, not a
    fairer one."""
    from mana.core import tasks
    for domain in tasks.DOMAINS:
        base = tasks.generate(domain, 5, seed=3)
        for surface in tasks.SURFACES:
            for a, b in zip(base, tasks.generate(domain, 5, seed=3, surface=surface)):
                assert a.answer == b.answer
                assert a.difficulty == b.difficulty
