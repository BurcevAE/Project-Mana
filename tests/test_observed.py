"""Episode -> Outcome -> Failure -> Finding, with nobody asked.

The criterion these are written against: after a real failed action, a
record appears in the findings ledger without anyone writing it. So the
last group runs a whole agent turn and then looks in the ledger, rather
than checking that the function in the middle returns the right shape --
this project's failures have all been in the wiring, not in the pieces.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mana.cognition import invariants, observed
from mana.cognition.findings import Finding, Ledger
from mana.core.gates import NOT_EVALUATED
from mana.journal import Episode, Journal, ToolCall
from mana.outcome import CONFIRMED, CONTRADICTED, UNOBSERVED
from mana.tools import FunctionTool, ToolResult


def _episode(answer: str = "Открыта база «УТ».", verified: str = CONTRADICTED,
             request: str = "запусти конфигуратор УТ") -> Episode:
    return Episode(episode_id="e1", session="s", started=1.0,
                   request=request, answer=answer,
                   calls=[ToolCall("onec_launch", ok=True, verified=verified)])


# --------------------------------------------------------------------------
# the invariant that reads no text
# --------------------------------------------------------------------------

def test_a_tool_that_says_the_state_is_wrong_is_a_failure():
    found = invariants.acted_but_state_disagrees(_episode(), [])
    assert found is not None
    assert found.kind == invariants.MECHANICAL
    assert found.evidence["tools"] == ["onec_launch"]


def test_an_unobserved_action_is_not_a_failure():
    # A tool that could not check is not a tool that got it wrong.
    # Folding the two together would rebuild the assumption the outcome
    # layer exists to remove.
    assert invariants.acted_but_state_disagrees(
        _episode(verified=UNOBSERVED), []) is None
    assert invariants.acted_but_state_disagrees(
        _episode(verified=CONFIRMED), []) is None
    assert invariants.acted_but_state_disagrees(
        _episode(verified=""), []) is None


def test_it_does_not_depend_on_what_was_said():
    # The point of it: no word list, no phrasing, nothing to interpret.
    for answer in ("Открыта база «УТ».", "", "не получилось", "☂"):
        assert invariants.acted_but_state_disagrees(
            _episode(answer=answer), []) is not None


def test_the_new_invariant_is_actually_in_the_list():
    assert invariants.acted_but_state_disagrees in invariants.INVARIANTS


def test_check_and_scan_agree_on_one_episode():
    episode = _episode()
    assert ([v.as_dict() for v in invariants.check(episode, [], targets=[])]
            == [v.as_dict() for v in invariants.scan([episode], targets=[])])


# --------------------------------------------------------------------------
# the failure becomes a record
# --------------------------------------------------------------------------

def test_a_failure_is_written_to_the_ledger_without_being_asked(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    written = observed.record(_episode(), [], ledger=ledger, targets=[])

    assert [w["invariant"] for w in written] == ["acted_but_state_disagrees"]
    rows = ledger.findings()
    assert len(rows) == 1
    assert rows[0].verdict == NOT_EVALUATED
    assert rows[0].measurement["observed_failures"] == 1
    assert rows[0].measurement["last_episode"] == "e1"
    assert "запусти конфигуратор УТ" in rows[0].measurement["last_request"]


def test_seeing_it_again_counts_it_rather_than_starting_over(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    for _ in range(3):
        observed.record(_episode(), [], ledger=ledger, targets=[])

    standing = observed.seen_before("acted_but_state_disagrees", ledger)
    assert standing is not None
    assert standing.measurement["observed_failures"] == 3
    # One standing record, three lines of history behind it.
    assert len(ledger.findings()) == 3
    assert len(ledger.latest()) == 1


def test_a_failure_never_measured_is_never_dressed_up_as_measured(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    observed.record(_episode(), [], ledger=ledger, targets=[])
    finding = ledger.latest()[0]
    # No trials, no interval, no null -- because there was no experiment.
    # Inventing them to reach a tidier class would be fabricating a
    # measurement, which is the one thing the ledger must not hold.
    assert "trials" not in finding.measurement
    assert finding.failure.failure == "UNCLASSIFIED"
    assert "trials" in finding.failure.rule


def test_a_guess_is_not_written_automatically(tmp_path):
    # PATTERN violations match free text against a word list. They are
    # useful in a report a person reads and must not enter the ledger
    # unasked.
    episode = Episode(episode_id="e2", session="s", started=1.0,
                      request="запусти конфигуратор УТ11-ER",
                      answer="Чтобы запустить конфигуратор, откройте меню.",
                      calls=[])
    guesses = [v for v in invariants.check(episode, [], ["УТ11-ER"])
               if v.kind == invariants.PATTERN]
    assert guesses, "the fixture stopped producing a pattern violation"

    ledger = Ledger(tmp_path / "f.jsonl")
    written = observed.record(episode, [], ledger=ledger, targets=["УТ11-ER"])
    assert all(w["invariant"] != guesses[0].invariant for w in written)


def test_a_clean_turn_writes_nothing(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    episode = Episode(episode_id="e3", session="s", started=1.0,
                      request="сколько будет 2+2", answer="Четыре.")
    assert observed.record(episode, [], ledger=ledger, targets=[]) == []
    assert ledger.findings() == []


def test_an_observed_failure_does_not_look_like_a_tried_approach(tmp_path):
    # The ledger's other reader asks "has this change been tried?". An
    # observation is not a change, and must not answer yes to one.
    from mana.cognition import candidates

    ledger = Ledger(tmp_path / "f.jsonl")
    observed.record(_episode(), [], ledger=ledger, targets=[])
    asked = ledger.already_tried(
        candidates.question_for("acted_but_state_disagrees"),
        {"echo_lookback": 6})
    assert asked is None


def test_the_record_survives_a_broken_ledger(tmp_path):
    # Evidence-keeping that can break the thing it observes is not worth
    # keeping. A directory where the file should be is enough.
    broken = tmp_path / "f.jsonl"
    broken.mkdir()
    assert observed.record(_episode(), [], ledger=Ledger(broken), targets=[]) == []


# --------------------------------------------------------------------------
# the wiring: a whole turn, and then the ledger
# --------------------------------------------------------------------------

def test_the_journal_keeps_the_session_prefix_in_memory(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    for n in range(3):
        journal.open(f"вопрос {n}", session="s1")
        journal.finish(f"ответ {n}")
    journal.open("другая сессия", session="s2")
    journal.finish("ответ")

    prefix = journal.session_recent("s1")
    assert [e.request for e in prefix] == ["вопрос 0", "вопрос 1", "вопрос 2"]
    assert journal.session_recent("s1", exclude=prefix[-1].episode_id) == prefix[:2]


def test_a_real_turn_whose_action_failed_leaves_a_finding(isolated_agent):
    """The criterion, end to end. Nothing here writes a finding by hand."""
    ledger = Ledger(Path(isolated_agent.config.findings_path))
    assert ledger.findings() == []

    # A tool that runs, returns, and reports that the machine is not in
    # the state it was asked for -- what onec_launch now does when 1С
    # exits on the spot.
    isolated_agent.tools.register(FunctionTool(
        "onec_launch", "",
        lambda **kw: ToolResult(ok=True, output={"launched": False},
                                meta={"verified": CONTRADICTED})),
        replace=True)

    isolated_agent.journal.open("запусти конфигуратор", session="s")
    isolated_agent.tools.call("onec_launch")
    episode = isolated_agent.journal.finish("Открыта база.", "app_intent")
    isolated_agent._note_observed_failures(episode)

    rows = ledger.latest()
    assert len(rows) == 1
    assert rows[0].question == observed.question_for("acted_but_state_disagrees")
    assert rows[0].measurement["observed_failures"] == 1


def test_solve_task_is_the_thing_that_calls_it(isolated_agent, monkeypatch):
    """The seam itself. A check that runs nowhere is the failure this
    project keeps repeating, and it has been repeated inside the
    machinery built to catch it."""
    seen = []
    monkeypatch.setattr(isolated_agent, "_note_observed_failures",
                        lambda episode: seen.append(episode))
    isolated_agent.solve_task("Сколько будет 2 плюс 2?")
    assert len(seen) == 1 and seen[0] is not None
    assert seen[0].request == "Сколько будет 2 плюс 2?"


def test_recording_never_fails_a_turn(isolated_agent, monkeypatch):
    def explode(*a, **kw):
        raise RuntimeError("ledger is on fire")

    monkeypatch.setattr("mana.cognition.observed.record", explode)
    result = isolated_agent.solve_task("Сколько будет 2 плюс 2?")
    assert result["answer"]


def test_the_ledger_path_answers_to_config(isolated_config):
    # The journal's old bug: a state path that ignores Config cannot be
    # redirected by a test, and the suite wrote 46 real episodes into the
    # repository root before that was found.
    assert "findings_path" in type(isolated_config).STATE_PATH_FIELDS
    assert Path(isolated_config.findings_path).is_absolute()


def test_a_repeated_answer_is_recorded_too_using_the_remembered_prefix(tmp_path):
    # The other mechanical invariant, and the one the user hit: a
    # greeting answered with the previous topic. It needs the session
    # prefix, which is why the journal now keeps one in memory.
    journal = Journal(tmp_path / "e.jsonl")
    ledger = Ledger(tmp_path / "f.jsonl")
    said = ("Чтобы запустить конфигуратор 1С, откройте меню «Пуск», найдите "
            "1С:Предприятие и выберите режим Конфигуратор в списке баз.")

    journal.open("как запустить конфигуратор 1С?", session="s")
    first = journal.finish(said)

    journal.open("Мана, привет!", session="s")
    second = journal.finish(said)

    earlier = journal.session_recent("s", exclude=second.episode_id)
    assert [e.episode_id for e in earlier] == [first.episode_id]

    written = observed.record(second, earlier, ledger=ledger, targets=[])
    assert [w["invariant"] for w in written] == ["repeats_earlier_answer"]
