"""Turning found failures into something the existing gates can judge.

The test that matters most is `test_a_responder_that_says_nothing_is_not
_an_improvement`. It is here because the first version of this module
failed it: measured over the seven real recorded turns, a responder
answering "Не знаю." to everything scored 0.8 against a baseline of 0.6,
left the healthy turns untouched and produced zero counterexamples --
with thirty trials it would have been ACCEPTED. Every invariant is
negative, and a purely negative oracle is maximised by silence.
"""
from __future__ import annotations

import pytest

from mana.journal import Episode, ToolCall
from mana.cognition import failure_domain as fd
from mana.cognition.invariants import ACTION_TOOLS

ECHO = ("Извините за путаницу. В процессе общения я автоматически добавила "
        "информацию о погоде, не спрашивая вас об этом, и это было лишним.")

SHORT = ["хорошо", "спасибо", "ясно", "понял", "ага",
         "ладно", "отлично", "ну да", "принято", "ок"]
TOPICS = ["налоги", "склад", "отчёт", "сотрудники", "касса",
          "договор", "поставщик", "цены", "остатки", "смена"]


def substantive(index: int, topic: str) -> str:
    return (f"По теме «{topic}» в ситуации {index}: развёрнутый содержательный "
            f"ответ, который отличается от всех прочих ответов этой сессии.")


def recorded(count: int = 60) -> list:
    """A session where every third turn repeats an older answer."""
    episodes = []
    for i in range(count):
        if i % 3 == 1:
            episodes.append(Episode(f"e{i:02d}", "s", float(i),
                                    f"{SHORT[i // 3 % 10]}, {i}", ECHO))
        else:
            topic = TOPICS[i % 10]
            episodes.append(Episode(f"e{i:02d}", "s", float(i),
                                    f"расскажи про {topic} в номере {i}",
                                    substantive(i, topic)))
    return episodes


def fixed(situation) -> tuple:
    return (f"По запросу «{situation.request}»: содержательный ответ, уникальный "
            f"для ситуации {situation.situation_id}, ничего ранее сказанного "
            f"не повторяющий.", [])


# --------------------------------------------------------------------------
# building the domain
# --------------------------------------------------------------------------

def test_healthy_turns_are_in_the_domain_too():
    """A domain made only of failures is scored perfectly by a change that
    ruins everything else, and would not notice."""
    domain = fd.build(recorded(), targets=[])
    counts = domain.counts()
    assert counts["flagged"] > 0
    assert counts["clean"] > 0


def test_the_split_is_stratified():
    """With samples this small an unstratified draw often puts no failures
    in the hidden set at all, and a hidden set with nothing to find
    confirms everything."""
    domain = fd.build(recorded(), targets=[])
    for part in (domain.train, domain.hidden):
        groups = {s.flagged_by for s in part}
        assert "clean" in groups
        assert any(g != "clean" for g in groups)


def test_the_hidden_split_is_stable_across_runs():
    """Same installation, same hidden set -- otherwise scores are not
    comparable over time."""
    first = fd.build(recorded(), targets=[])
    second = fd.build(recorded(), targets=[])
    assert ([s.situation_id for s in first.hidden]
            == [s.situation_id for s in second.hidden])


def test_the_identity_says_whether_it_was_salted():
    """A score whose source cannot be told from another installation's is
    a score somebody will quietly compare with theirs."""
    salted = fd.build(recorded(), targets=[])
    plain = fd.build(recorded(), targets=[], salted=False)
    assert "@" in salted.identity
    assert plain.identity == "recorded_turns-v1"


def test_a_situation_carries_only_its_own_sessions_history():
    episodes = [Episode("a", "s1", 1.0, "вопрос", "ответ"),
                Episode("b", "s2", 2.0, "вопрос", "ответ"),
                Episode("c", "s1", 3.0, "вопрос", "ответ")]
    situations = {s.situation_id: s for s in fd.situations_from(episodes, [])}
    assert [e.episode_id for e in situations["c"].prefix] == ["a"]


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------

def test_the_oracle_is_the_invariant_set():
    situation = fd.Situation("x", "s", "Открой Notepad++")
    assert fd.grade(situation, "Открываю Notepad++.", (), []).passed is False
    assert fd.grade(situation, "Notepad++ запущен (pid 42).",
                    [ToolCall("open_in_editor", True)], []).passed is True


def test_the_calls_are_part_of_the_answer():
    """An answer graded without them cannot tell "запустил" from "сказал,
    что запустил", which is the failure this whole chain exists for."""
    situation = fd.Situation("x", "s", "Запусти 1С")
    without = fd.grade(situation, "Открываю 1С.", (), [])
    with_call = fd.grade(situation, "Открываю 1С.",
                         [ToolCall("onec_launch", True)], [])
    assert without.passed is False and with_call.passed is True


def test_a_responder_that_raises_is_a_failure_not_a_skip():
    """Dropping it would bias the sample toward what happened to work."""
    def broken(_situation):
        raise RuntimeError("candidate blew up")

    judged = fd.evaluate([fd.Situation("x", "s", "вопрос")], broken, targets=[])
    assert judged[0].passed is False
    assert judged[0].fired == ("responder_error",)


# --------------------------------------------------------------------------
# the hole this module had, and the gate that closes it
# --------------------------------------------------------------------------

def test_a_responder_that_says_nothing_is_not_an_improvement():
    """Measured on the seven real turns before this was fixed: 0.8 against
    a baseline of 0.6, healthy turns untouched, zero counterexamples. Every
    invariant is negative, so silence satisfies all of them."""
    domain = fd.build(recorded(), targets=[])
    baseline = fd.replay(domain.train, targets=[])
    silent = fd.evaluate(domain.train, lambda s: ("Не знаю.", []), targets=[])

    # It still scores better on the invariants -- that is the point.
    assert fd.pass_rate(silent) > fd.pass_rate(baseline)

    # And it is refused anyway, by the gate that already exists for this.
    out = fd.judge_change("молчать вместо ответа", domain, baseline, silent)
    assert out["verdict"]["accepted"] is False
    assert "counterexamples" in out["verdict"]["failed_gates"]
    assert out["report"]["counterexamples"]["degenerated"] > 0


@pytest.mark.parametrize("text", [
    "Не знаю.", "не могу этого сделать", "К сожалению, нет данных.",
    "", "   ", "ага",
])
def test_a_stock_refusal_is_recognised(text):
    assert fd.is_stock_refusal(text) is True


@pytest.mark.parametrize("text", [
    "Notepad++ запущен (pid 42). Файл открыт на первой строке.",
    "Не получилось: базы «UT11-ER» нет в списке. Есть: Информационная база.",
])
def test_a_real_answer_is_not_a_refusal(text):
    """A refusal reported with its reason is a substantive answer. Counting
    it as degeneration would refuse every honest failure message."""
    assert fd.is_stock_refusal(text) is False


# --------------------------------------------------------------------------
# handing it to the gates that already exist
# --------------------------------------------------------------------------

def test_a_genuine_fix_with_enough_evidence_is_accepted():
    """A pipeline that can only ever reject is indistinguishable from a
    broken one."""
    domain = fd.build(recorded(), targets=[])
    out = fd.judge_change(
        "перестать повторять прошлые ответы", domain,
        fd.replay(domain.train, targets=[]),
        fd.evaluate(domain.train, fixed, targets=[]),
        fd.replay(domain.hidden, targets=[]),
        fd.evaluate(domain.hidden, fixed, targets=[]))

    assert out["verdict"]["status"] == "ACCEPTED"
    measurements = out["verdict"]["measurements"]
    assert measurements["paired_trials"] >= 30
    assert measurements["mcnemar"]["p_value"] < 0.05
    assert out["report"]["by_group"]["clean"]["baseline"] == \
           out["report"]["by_group"]["clean"]["candidate"]


def test_too_few_situations_is_refused_on_sample_size():
    """Three flagged turns is not evidence about anything, and the honest
    answer with today's record is "not enough"."""
    domain = fd.build(recorded(9), targets=[])
    out = fd.judge_change("что угодно", domain,
                          fd.replay(domain.train, targets=[]),
                          fd.evaluate(domain.train, fixed, targets=[]))
    assert out["verdict"]["accepted"] is False
    assert "sample_size" in out["verdict"]["failed_gates"]


def test_one_situation_is_one_trial():
    """Three invariants on one episode are not three observations;
    McNemar over them would buy significance that is not there."""
    domain = fd.build(recorded(30), targets=[])
    outcomes = fd.paired(fd.replay(domain.train, targets=[]),
                         fd.evaluate(domain.train, fixed, targets=[]))
    assert len(outcomes) == len(domain.train)
    assert len({o.task_id for o in outcomes}) == len(outcomes)


def test_the_trial_domain_is_the_invariant_that_flagged_it():
    """So the existing per-domain regression gate does the cross-invariant
    check without a second gate being written here."""
    domain = fd.build(recorded(), targets=[])
    outcomes = fd.paired(fd.replay(domain.train, targets=[]),
                         fd.evaluate(domain.train, fixed, targets=[]))
    assert {o.domain for o in outcomes} == {"clean", "repeats_earlier_answer"}


def test_a_situation_missing_from_one_arm_is_not_a_trial():
    baseline = [fd.Judgement("a", "clean", True, (), "ответ раз"),
                fd.Judgement("b", "clean", True, (), "ответ два")]
    candidate = [fd.Judgement("a", "clean", True, (), "ответ раз")]
    assert [o.task_id for o in fd.paired(baseline, candidate)] == ["a"]


def test_the_verdict_comes_from_the_gates_and_not_from_here():
    """A parallel judge would let a failure domain accept on easier terms
    than everything else, which is the loophole the design closes."""
    import inspect

    source = inspect.getsource(fd.judge_change)
    assert "from ..core.gates import" in source
    for invented in ("p_value <", "MIN_PAIRED", "accepted = True"):
        assert invented not in source


def test_the_report_is_readable_when_no_verdict_is_possible():
    """The numbers are worth reading especially when the sample is too
    small: a caller that only sees a status learns nothing about how close
    it was."""
    domain = fd.build(recorded(9), targets=[])
    report = fd.report(domain, fd.replay(domain.train, targets=[]),
                       fd.evaluate(domain.train, fixed, targets=[]))
    assert report["trials"] > 0
    assert "baseline_pass_rate" in report and "candidate_pass_rate" in report
    assert "by_group" in report


def test_the_module_states_that_it_is_not_sufficient_on_its_own():
    """Measuring "stopped contradicting itself" is not measuring "answers
    well". If that caveat is ever dropped, this fails."""
    import mana.cognition.failure_domain as module

    assert "never a sufficient one" in (module.__doc__ or "")
