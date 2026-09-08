"""What the invariants must find, and what they must not claim.

The negative tests carry the weight. A missed failure costs one data
point; a detector that flags healthy turns floods the report until nobody
reads it, and a `pattern` guess reported as a `mechanical` fact is the
"claim resting on nothing" this project keeps having to fix.
"""
from __future__ import annotations

import pytest

from mana.journal import Episode, ToolCall
from mana.cognition.invariants import (
    MECHANICAL, PATTERN, ACTION_TOOLS, INVARIANTS,
    Violation, scan, summarise, _stemmed_target,
    repeats_earlier_answer, claimed_action_without_acting,
    actionable_request_not_acted_on,
)

TARGETS = ["UT11-ER", "Информационная база"]


def ep(request: str, answer: str, calls=(), session="s",
       episode_id="", route="pipeline") -> Episode:
    return Episode(
        episode_id=episode_id or f"e{abs(hash((request, answer))) % 10000}",
        session=session, started=0.0, request=request, answer=answer,
        route=route, calls=[ToolCall(t, ok) for t, ok in calls])


LONG = ("Извините за путаницу. В процессе общения я автоматически добавила "
        "информацию о погоде, не спрашивая вас об этом. Если у вас есть "
        "вопросы или задачи, скажите.")


# --------------------------------------------------------------------------
# repeats_earlier_answer -- mechanical
# --------------------------------------------------------------------------

def test_it_finds_a_repeat_older_than_one_turn():
    """The real case. `echo_guard` compares only with the previous turn,
    so an answer identical to one three turns back scored 1.00 against
    that turn, 0.018 against the turn the guard looked at, and passed."""
    earlier = [
        ep("почему ты сказала о погоде?", LONG),
        ep("нет я хочу узнать рецепт хот-дога",
           "Вот рецепт классического хот-дога: сосиски, булочки, лук и кетчуп. "
           "Обжарьте сосиски, разогрейте булочки, соберите."),
    ]
    current = ep("хорошо, спасибо", LONG)

    found = repeats_earlier_answer(current, earlier)
    assert found is not None
    assert found.kind == MECHANICAL
    assert found.evidence["answer_similarity"] == 1.0
    assert found.evidence["repeated_from"] == earlier[0].episode_id


def test_the_same_question_twice_may_have_the_same_answer():
    """Getting this backwards would make MANA refuse to be consistent,
    which is worse than the bug."""
    earlier = [ep("какая погода в Воронеже?", LONG)]
    current = ep("какая погода в Воронеже?", LONG)
    assert repeats_earlier_answer(current, earlier) is None


def test_a_short_answer_is_not_evidence_of_anything():
    earlier = [ep("вопрос раз", "Не нашлось.")]
    assert repeats_earlier_answer(ep("вопрос два", "Не нашлось."), earlier) is None


def test_a_repeat_from_another_session_is_not_this_session_repeating_itself():
    earlier = [ep("почему про погоду?", LONG, session="other")]
    current = ep("хорошо, спасибо", LONG, session="s")
    assert repeats_earlier_answer(current, earlier) is None


def test_two_different_answers_are_left_alone():
    earlier = [ep("вопрос раз", LONG)]
    current = ep("вопрос два", "Совершенно другой ответ про совершенно "
                               "другие вещи, ничем не похожий на прошлый.")
    assert repeats_earlier_answer(current, earlier) is None


# --------------------------------------------------------------------------
# claimed_action_without_acting -- pattern
# --------------------------------------------------------------------------

def test_it_finds_the_failure_the_journal_was_built_for():
    """Measured: "Открой Notepad++" -> "Открываю Notepad++." zero calls."""
    found = claimed_action_without_acting(
        ep("Открой Notepad++", "Открываю Notepad++."), [])
    assert found is not None
    assert found.kind == PATTERN
    assert found.evidence["claim"] == "Открываю"


def test_a_claim_backed_by_a_tool_that_ran_is_not_a_violation():
    found = claimed_action_without_acting(
        ep("Открой Notepad++", "Notepad++ запущен, открываю файл.",
           calls=[("open_in_editor", True)]), [])
    assert found is None


def test_a_tool_that_failed_does_not_back_a_claim():
    """A refusal reported honestly is fine; a claim over a failed call is
    not, and the difference is the tool's `ok`."""
    found = claimed_action_without_acting(
        ep("Открой Notepad++", "Открываю Notepad++.",
           calls=[("open_in_editor", False)]), [])
    assert found is not None


def test_a_non_action_tool_does_not_back_a_claim():
    found = claimed_action_without_acting(
        ep("Открой Notepad++", "Открываю Notepad++.",
           calls=[("llm_generate", True), ("write_memory", True)]), [])
    assert found is not None


@pytest.mark.parametrize("answer", [
    "Поняла, вы хотели запустить конфигуратор информационной базы.",
    "Давайте это сделаем вместе.",
    "Чтобы открыть конфигуратор, найдите ярлык 1С на рабочем столе.",
    "Я не могу запустить эту программу.",
])
def test_describing_an_action_is_not_claiming_it(answer):
    """Infinitives excluded on purpose: "вы хотели запустить" describes
    the request, "запускаю" claims the deed."""
    assert claimed_action_without_acting(ep("запусти", answer), []) is None


# --------------------------------------------------------------------------
# actionable_request_not_acted_on -- pattern, the wide twin
# --------------------------------------------------------------------------

def test_it_finds_a_request_that_produced_no_action():
    """Both of these were misses for the actor until the recognition fix
    was adopted on 08.09.2026. The detector found them first, which is
    what a wide shadow twin is for -- and it still fires on any request
    that produced no action, whatever the actor now recognises."""
    for text in ("я хочу поработать с 1С запусти пожалуйста конфигуратор "
                 "информационной базы",
                 "я хочу что бы ты открыла конфигуратор информационной базы "
                 "на моем компьютере"):
        found = actionable_request_not_acted_on(ep(text, "инструкция"), [],
                                                TARGETS)
        assert found is not None and found.kind == PATTERN


def test_a_request_that_was_carried_out_is_not_a_violation():
    found = actionable_request_not_acted_on(
        ep("запусти 1С", "Открыто окно выбора базы.",
           calls=[("onec_launch", True)]), [], TARGETS)
    assert found is None


@pytest.mark.parametrize("text", [
    "как запустить 1С на новом компьютере?",     # asks about, not for
    "почему не запускается конфигуратор?",
    "можно ли открыть базу без пароля?",
    "расскажи что такое конфигуратор",
    "сколько записей в базе данных вообще",
])
def test_it_stays_off_messages_that_ask_for_nothing(text):
    """The first false positive found: a question about how to launch 1C
    was flagged as a request to launch it. Width is the point of this
    detector, but a report full of advice questions is one nobody reads."""
    assert actionable_request_not_acted_on(ep(text, "ответ"), [], TARGETS) is None


@pytest.mark.parametrize("text", [
    "Мана, можешь запустить 1С?",
    "запусти пожалуйста конфигуратор информационной базы",
])
def test_a_request_wearing_a_question_mark_still_counts(text):
    """Excluding on the question mark itself would lose real requests,
    which for a detector is the expensive mistake. Only an interrogative
    at the START of the message means "asking about"."""
    assert actionable_request_not_acted_on(ep(text, "ответ"), [], TARGETS) is not None


def test_a_base_name_is_matched_through_russian_inflection():
    """Nobody writes a base name in the nominative. Measured: the narrow
    matcher takes "Информационная база" and misses "информационной базы"."""
    assert _stemmed_target("запусти конфигуратор информационной базы",
                           TARGETS) == "Информационная база"
    assert _stemmed_target("открой информационную базу",
                           TARGETS) == "Информационная база"


def test_the_stem_match_does_not_fire_on_a_shared_generic_word():
    """Every word of the name must appear. The generic half is shared;
    the distinctive half is what decides."""
    assert _stemmed_target("сколько записей в базе данных", TARGETS) == ""
    assert _stemmed_target("расскажи про базы вообще", TARGETS) == ""


# --------------------------------------------------------------------------
# the two kinds must never be confused
# --------------------------------------------------------------------------

def test_every_violation_declares_which_kind_it_is():
    for violation in scan([
        ep("Открой Notepad++", "Открываю Notepad++."),
        ep("запусти 1С", "Чтобы запустить, найдите ярлык."),
    ], TARGETS):
        assert violation.kind in (MECHANICAL, PATTERN)


def test_only_the_measured_invariant_calls_itself_mechanical():
    """A guess reported as a fact is the failure this project keeps
    fixing. Anything downstream may use `mechanical` without a person
    confirming, so the label has to be earned."""
    kinds = {}
    for episode, earlier in [
        (ep("хорошо", LONG), [ep("другой вопрос", LONG)]),
        (ep("Открой Notepad++", "Открываю Notepad++."), []),
        (ep("запусти 1С", "инструкция"), []),
    ]:
        for violation in scan(list(earlier) + [episode], TARGETS):
            kinds[violation.invariant] = violation.kind
    assert kinds["repeats_earlier_answer"] == MECHANICAL
    assert kinds["claimed_action_without_acting"] == PATTERN
    assert kinds["actionable_request_not_acted_on"] == PATTERN


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def test_a_healthy_run_produces_nothing():
    """The bar that decides whether anybody keeps reading the report."""
    episodes = [
        ep("Сколько будет 17 умножить на 23?", "391",
           calls=[("llm_generate", True)]),
        ep("а если 18 на 23?", "414", calls=[("llm_generate", True)]),
        ep("Запусти 1С", "Открыто окно выбора базы; предложены UT11-ER, "
                         "Информационная база. Базу выбираете вы.",
           calls=[("onec_launch", True)], route="app_intent"),
        ep("спасибо", "Пожалуйста. Обращайтесь, если понадобится ещё."),
    ]
    assert scan(episodes, TARGETS) == []


def test_one_episode_may_violate_several_invariants():
    """"Наврал про действие" and "просили действие, не сделал" are
    different failures with different fixes; collapsing them hides one."""
    episodes = [ep("Открой Notepad++", "Открываю Notepad++.")]
    names = {v.invariant for v in scan(episodes, TARGETS)}
    assert names == {"claimed_action_without_acting",
                     "actionable_request_not_acted_on"}


def test_a_broken_invariant_does_not_stop_the_scan(monkeypatch):
    import mana.cognition.invariants as module

    def broken(episode, earlier):
        raise RuntimeError("this invariant is broken")

    monkeypatch.setattr(module, "INVARIANTS",
                        (broken, module.repeats_earlier_answer))
    # Genuinely different questions: "вопрос раз"/"вопрос два" score as
    # the same question and are correctly not a violation at all.
    episodes = [ep("почему ты сказала о погоде?", LONG),
                ep("хорошо, спасибо", LONG)]
    assert len(module.scan(episodes, TARGETS)) == 1


def test_the_summary_states_its_denominator():
    """"12 нарушений" means nothing without how many turns were looked
    at, and a rate off seven episodes is not a rate."""
    episodes = [ep("Открой Notepad++", "Открываю Notepad++."),
                ep("спасибо", "Пожалуйста, обращайтесь ещё когда угодно.")]
    summary = summarise(episodes, scan(episodes, TARGETS))
    assert summary["episodes"] == 2
    assert summary["episodes_with_a_violation"] == 1
    assert summary["violations"] == 2
    assert "rate" not in summary and "percent" not in summary


def test_every_invariant_is_registered():
    """This project's recurring failure is machinery built and connected
    to nothing -- an invariant absent from INVARIANTS never runs."""
    assert repeats_earlier_answer in INVARIANTS
    assert claimed_action_without_acting in INVARIANTS
    assert actionable_request_not_acted_on in INVARIANTS
    assert summarise([], [])["invariants_run"] == [f.__name__ for f in INVARIANTS]


def test_nothing_here_can_change_an_answer():
    """Shadow mode is the whole safety argument. If this module ever gains
    a way to act, that decision should be deliberate and visible here."""
    import inspect
    import mana.cognition.invariants as module

    source = inspect.getsource(module)
    for forbidden in ("registry.call", "solve_task", ".answer(", "subprocess"):
        assert forbidden not in source


def test_the_action_tools_are_an_allowlist():
    """A new tool is not an action until somebody says so, which fails
    towards reporting a violation rather than towards missing one."""
    assert "llm_generate" not in ACTION_TOOLS
    assert "write_memory" not in ACTION_TOOLS
    assert "onec_list_bases" not in ACTION_TOOLS      # a read, not an action
    assert "onec_launch" in ACTION_TOOLS
    assert "onec_confirm_write" in ACTION_TOOLS
