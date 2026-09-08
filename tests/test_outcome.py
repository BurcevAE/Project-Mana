"""What a tool did, against what it was asked to do.

Three groups. The verdict rules; the observation of a real 1C-shaped
launch, run against real processes rather than a mock of the observer;
and the wiring, which is checked end to end on purpose. The last one is
where this project keeps failing: `echo_lookback` was declared, tested,
proposed by the generator and called by nothing, and the test passed
because it checked the function instead of the path the function sits on.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from mana.apps import intent as app_intent
from mana.apps import onec_launch
from mana.journal import Journal
from mana.outcome import CONFIRMED, CONTRADICTED, UNOBSERVED, Outcome, unobserved
from mana.tools import FunctionTool, ToolRegistry, ToolResult


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

def test_everything_looked_at_and_matching_is_confirmed():
    outcome = Outcome("onec_launch", "открыть УТ",
                      expected={"running": True, "base": "УТ"},
                      observed={"running": True, "base": "УТ"})
    assert outcome.verified == CONFIRMED
    assert outcome.succeeded() and not outcome.failed()


def test_a_mismatch_on_any_axis_is_a_contradiction():
    outcome = Outcome("onec_launch", "открыть предприятие",
                      expected={"running": True, "mode": "предприятие"},
                      observed={"running": True, "mode": "конфигуратор"})
    assert outcome.verified == CONTRADICTED
    assert outcome.delta() == {
        "mode": {"expected": "предприятие", "observed": "конфигуратор"}}


def test_an_axis_nobody_looked_at_is_not_an_axis_that_came_out_right():
    # The whole reason there are three verdicts. Folding this into
    # "confirmed" is the defect the module exists to remove.
    outcome = Outcome("onec_launch", "открыть УТ",
                      expected={"running": True, "base": "УТ"},
                      observed={"running": True})
    assert outcome.verified == UNOBSERVED
    assert outcome.unobserved == ["base"]
    assert outcome.compared == ["running"]


def test_observing_nothing_at_all_is_unobserved_not_confirmed():
    assert Outcome("t", "g", expected={"running": True}).verified == UNOBSERVED
    assert Outcome("t", "g").verified == UNOBSERVED
    assert unobserved("t", "g").verified == UNOBSERVED


def test_a_contradiction_outranks_an_unobserved_axis():
    outcome = Outcome("t", "g",
                      expected={"running": True, "base": "УТ", "mode": "x"},
                      observed={"running": False})
    assert outcome.verified == CONTRADICTED


def test_values_compare_the_way_a_person_would_read_them():
    # A base name comes back from a window title the way 1С printed it.
    outcome = Outcome("t", "g", expected={"base": "УТ11-ER"},
                      observed={"base": " ут11-er "})
    assert outcome.verified == CONFIRMED


def test_the_verdict_cannot_be_set_by_a_caller():
    outcome = Outcome("t", "g", expected={"running": True},
                      observed={"running": True})
    with pytest.raises(Exception):
        outcome.verified = CONTRADICTED  # type: ignore[misc]


def test_the_verdict_travels_with_the_dict_and_is_recomputed_on_the_way_back():
    outcome = Outcome("t", "g", expected={"running": True},
                      observed={"running": False}, evidence={"exit_code": 3})
    row = outcome.as_dict()
    assert row["verified"] == CONTRADICTED and row["delta"]
    assert Outcome.from_dict(row).verified == CONTRADICTED
    # A stale verdict in the file does not survive the round trip: the
    # rule is applied again to the same evidence.
    row["verified"] = CONFIRMED
    assert Outcome.from_dict(row).verified == CONTRADICTED


def test_the_summary_says_which_of_the_three_happened():
    assert "не то, что просили" in Outcome(
        "t", "g", {"running": True}, {"running": False}).summary()
    assert "не подтверждено" in Outcome(
        "t", "g", {"base": "УТ"}, {}).summary()
    assert "проверено" in Outcome(
        "t", "g", {"running": True}, {"running": True}).summary()


# --------------------------------------------------------------------------
# the observation, against real processes
# --------------------------------------------------------------------------

pytestmark_windows = pytest.mark.skipif(
    sys.platform != "win32", reason="tasklist is a Windows program")


@pytestmark_windows
def test_a_process_that_exits_immediately_is_a_failed_launch():
    # This is the case the old code called a successful launch: Popen
    # returned, so `launched: True` was written and the process was
    # already gone by the time anybody read it.
    process = subprocess.Popen(["cmd", "/c", "exit", "3"])
    seen = onec_launch._observe(process, "УТ", settle=3.0)
    assert seen["observed"]["running"] is False
    assert seen["evidence"]["exit_code"] == 3
    outcome = Outcome("onec_launch", "открыть УТ",
                      expected={"running": True, "base": "УТ"},
                      observed=seen["observed"], evidence=seen["evidence"])
    assert outcome.verified == CONTRADICTED


@pytestmark_windows
def test_a_live_process_with_no_window_yet_is_unobserved_not_failed():
    # 1С asking for a password looks exactly like this. Reporting it as a
    # failure would be as wrong as reporting it as a success.
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(9)"])
    try:
        seen = onec_launch._observe(process, "УТ", settle=1.0)
        assert seen["observed"]["running"] is True
        assert "base" not in seen["observed"] and "mode" not in seen["observed"]
        assert seen["note"]
        outcome = Outcome("onec_launch", "открыть УТ",
                          expected={"running": True, "base": "УТ"},
                          observed=seen["observed"])
        assert outcome.verified == UNOBSERVED
    finally:
        process.kill()


@pytestmark_windows
def test_a_window_title_is_read_back_from_the_running_process_list():
    listing = subprocess.run(["tasklist", "/V", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, errors="replace")
    titled = None
    for line in listing.stdout.splitlines():
        fields = [f.strip('"') for f in line.split('","')]
        if len(fields) > 2 and fields[1].isdigit() and fields[-1].strip() not in ("N/A", ""):
            titled = (int(fields[1]), fields[-1].strip())
            break
    if titled is None:
        pytest.skip("no process on this machine has a window title")
    assert onec_launch._window_title(titled[0]) == titled[1]


def test_a_pid_that_is_not_running_has_no_title():
    assert onec_launch._window_title(999999) == ""


# --------------------------------------------------------------------------
# the wiring: does the verdict actually get anywhere
# --------------------------------------------------------------------------

def _registry(verified: str) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(FunctionTool(
        "onec_launch", "",
        lambda **kw: ToolResult(ok=True, output={"verified": verified},
                                meta={"verified": verified})))
    return registry


def test_the_journal_records_the_verdict_and_not_only_that_the_call_returned(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    registry = _registry(CONTRADICTED)
    registry.observe(journal.note_call)

    journal.open("запусти конфигуратор УТ", session="s")
    registry.call("onec_launch")
    episode = journal.finish("готово", "app_intent")

    assert episode is not None
    call = episode.calls[0]
    # Both facts, kept apart: the call returned, and the machine says the
    # action did not do what it was asked to do.
    assert call.ok is True and call.verified == CONTRADICTED
    assert episode.acted() is True
    assert episode.contradicted() == ["onec_launch"]


def test_a_tool_that_does_not_observe_leaves_the_verdict_empty(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    registry = ToolRegistry()
    registry.register(FunctionTool("quiet", "", lambda **kw: ToolResult(ok=True)))
    registry.observe(journal.note_call)
    journal.open("сделай", session="s")
    registry.call("quiet")
    episode = journal.finish("готово")
    assert episode is not None
    assert episode.calls[0].verified == ""
    assert episode.contradicted() == [] and episode.unverified() == []


def test_the_verdict_survives_the_journal_file(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    registry = _registry(UNOBSERVED)
    registry.observe(journal.note_call)
    journal.open("запусти УТ", session="s")
    registry.call("onec_launch")
    journal.finish("готово")

    restored = journal.episodes(1)[0]
    assert restored.calls[0].verified == UNOBSERVED
    assert restored.unverified() == ["onec_launch"]


# --------------------------------------------------------------------------
# the wiring: does the person hear the difference
# --------------------------------------------------------------------------

def _said(outcome: Outcome, **extra) -> str:
    data = {"base": "УТ", "kind": "файловая", "location": r"C:\1С\УТ",
            "mode": "предприятие", "outcome": outcome.as_dict()}
    data.update(extra)
    found = app_intent.Intent("launch_onec", "onec_launch", {"base": "УТ"}, "запусти УТ")
    return app_intent.describe(found, {"ok": True, "output": data})


def test_three_verdicts_are_three_different_sentences():
    confirmed = _said(Outcome("onec_launch", "открыть УТ",
                              {"running": True, "base": "УТ", "mode": "предприятие"},
                              {"running": True, "base": "УТ", "mode": "предприятие"}))
    unseen = _said(Outcome("onec_launch", "открыть УТ",
                           {"running": True, "base": "УТ", "mode": "предприятие"},
                           {"running": True},
                           note="окно ещё не появилось"))
    died = _said(Outcome("onec_launch", "открыть УТ",
                         {"running": True, "base": "УТ", "mode": "предприятие"},
                         {"running": False}, evidence={"exit_code": 1}))

    assert len({confirmed, unseen, died}) == 3
    assert "проверено" in confirmed
    assert "процесс работает" in unseen and "окно ещё не появилось" in unseen
    assert "Не открылось" in died and "код возврата 1" in died


def test_the_configurator_opening_instead_is_said_out_loud():
    # The direction the user actually complained about, and the one the
    # window title can prove: no Предприятие window says "Конфигуратор".
    said = _said(Outcome("onec_launch", "открыть УТ в предприятии",
                         {"running": True, "mode": "предприятие"},
                         {"running": True, "mode": "конфигуратор"}))
    assert "не то" in said.lower()
    assert "конфигуратор" in said and "предприятие" in said


def test_an_unconfirmed_launch_is_never_reported_as_an_open_base():
    said = _said(Outcome("onec_launch", "открыть УТ",
                         {"running": True, "base": "УТ"}, {"running": True}))
    assert "Открыта база" not in said


def test_the_reply_names_the_axis_it_could_not_check():
    # A window title that gives the base but not the mode. Saying "не
    # видно, какая база открыта" here would be false while the verdict
    # beside it was right -- which is how this was caught.
    said = _said(Outcome("onec_launch", "открыть УТ",
                         {"running": True, "base": "УТ", "mode": "предприятие"},
                         {"running": True, "base": "УТ"}))
    assert "в каком режиме открыто" in said
    assert "какая база открыта" not in said


def test_the_password_warning_survives_every_branch():
    for outcome in (Outcome("t", "g", {"running": True}, {"running": True}),
                    Outcome("t", "g", {"running": True, "base": "УТ"}, {"running": True}),
                    Outcome("t", "g", {"running": True}, {"running": False})):
        said = _said(outcome, warning="пароль передан в командной строке")
        assert "пароль передан в командной строке" in said


def test_the_goal_travels_with_the_call():
    seen = {}

    registry = ToolRegistry()
    registry.register(FunctionTool(
        "onec_launch", "",
        lambda **kw: (seen.update(kw), ToolResult(ok=True, output={}))[1]))
    found = app_intent.Intent("launch_onec", "onec_launch", {"base": "УТ"}, "запусти УТ")
    app_intent.perform(found, registry, goal="открой мне конфигуратор УТ, пожалуйста")
    assert seen["goal"] == "открой мне конфигуратор УТ, пожалуйста"


# --------------------------------------------------------------------------
# reading the record back
# --------------------------------------------------------------------------

def test_the_journal_can_list_the_turns_that_claimed_what_did_not_happen(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    registry = _registry(CONTRADICTED)
    registry.observe(journal.note_call)
    journal.open("запусти конфигуратор УТ", session="s")
    registry.call("onec_launch")
    journal.finish("Открыта база «УТ».", "app_intent")

    journal.open("привет", session="s")
    journal.finish("Здравствуйте.", "pipeline")

    found = journal.contradicted_calls()
    assert len(found) == 1
    assert found[0]["tool"] == "onec_launch"
    assert found[0]["request"] == "запусти конфигуратор УТ"
    assert found[0]["answer"] == "Открыта база «УТ»."


def test_the_reader_marks_the_three_states_apart():
    from mana.cli import _mark
    from mana.journal import ToolCall

    assert _mark(ToolCall("t", ok=False)) == "!"
    assert _mark(ToolCall("t", ok=True, verified=CONTRADICTED)) == "✗"
    assert _mark(ToolCall("t", ok=True, verified=UNOBSERVED)) == "?"
    assert _mark(ToolCall("t", ok=True, verified=CONFIRMED)) == ""
    assert _mark(ToolCall("t", ok=True)) == ""
