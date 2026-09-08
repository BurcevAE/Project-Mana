"""What the journal must record, and what it must never record.

The load-bearing tests here are the negative ones. A journal that misses
an episode costs one data point; a journal that writes a password into a
plain file on disk, or that fails a user's turn because a log write
failed, is a defect worse than not having one.
"""
from __future__ import annotations

import json
import threading

import pytest

from mana.journal import Journal, Episode, ToolCall, scrub, REDACTED
from mana.tools import ToolRegistry, FunctionTool, ToolResult


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------

def test_scrub_removes_a_password_typed_the_way_it_is_asked_for():
    # This exact shape is one of the three 1C scenarios the project was
    # asked to support, so it is ordinary usage, not a corner case.
    text = 'Мана запусти 1С базу UT11-ER логин Иванов пароль Qwerty123!'
    out = scrub(text)
    assert "Qwerty123!" not in out
    assert REDACTED in out
    assert "UT11-ER" in out           # the base name is not a secret
    assert "Иванов" in out


@pytest.mark.parametrize("text,secret", [
    ("password: hunter2", "hunter2"),
    ("PWD=s3cret", "s3cret"),
    ('1cv8.exe /S srv /N user /Ptopsecret', "topsecret"),
    ("api_key sk-ABCDEF0123456789xyz", "sk-ABCDEF0123456789xyz"),
    ("вот ключ gsk_0123456789abcdefghij", "gsk_0123456789abcdefghij"),
])
def test_scrub_covers_the_labelled_forms(text, secret):
    assert secret not in scrub(text)


def test_scrub_leaves_ordinary_text_alone():
    text = "Открой отчёт по продажам за март и посчитай итог"
    assert scrub(text) == text


def test_a_password_in_the_request_never_reaches_the_file(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    journal.open("запусти базу UT11-ER пароль Qwerty123!", session="s")
    journal.finish("Запущена база «UT11-ER».", "app_intent")

    raw = (tmp_path / "e.jsonl").read_text(encoding="utf-8")
    assert "Qwerty123!" not in raw
    assert "UT11-ER" in raw


def test_a_password_in_a_tool_error_never_reaches_the_file(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    journal.open("запусти", session="s")
    journal.note_call("onec_launch", False, 0.1,
                      'сбой команды: 1cv8.exe /N user /PQwerty123!')
    journal.finish("Не получилось.", "app_intent")

    raw = (tmp_path / "e.jsonl").read_text(encoding="utf-8")
    assert "Qwerty123!" not in raw


def test_tool_arguments_are_not_recorded_at_all(tmp_path):
    # The observer signature is name/ok/latency/error by design: arguments
    # are the field most likely to carry a credential with the least to
    # show for the risk, so there is nowhere for them to be written.
    journal = Journal(tmp_path / "e.jsonl")
    recorder = journal.open("q", session="s")
    journal.note_call("onec_launch", True, 0.0)
    episode = journal.finish("ok", "app_intent")
    assert episode is not None
    assert episode.calls[0].as_dict().keys() == {"tool", "ok", "latency", "error"}


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

def test_an_episode_records_the_calls_the_turn_made(tmp_path):
    journal = Journal(tmp_path / "e.jsonl", version="9.9")
    journal.open("открой Notepad++", session="sess-1")
    journal.note_call("open_in_editor", True, 0.25)
    episode = journal.finish("Notepad++ запущен (pid 42).", "app_intent")

    assert episode is not None
    assert episode.route == "app_intent"
    assert episode.tools_used() == ["open_in_editor"]
    assert episode.acted() is True
    assert episode.version == "9.9"
    assert episode.session == "sess-1"


def test_the_failure_this_exists_for_is_visible_in_the_record(tmp_path):
    # "Открой Notepad++" -> "Открываю Notepad++." with no tool call. The
    # journal does not judge this; it records the two facts that make it
    # judgeable at all.
    journal = Journal(tmp_path / "e.jsonl")
    journal.open("Открой Notepad++", session="s")
    episode = journal.finish("Открываю Notepad++.", "pipeline")

    assert episode is not None
    assert episode.calls == []
    assert episode.acted() is False
    assert "Открываю" in episode.answer


def test_an_unknown_route_is_stored_as_pipeline(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    journal.open("q", session="s")
    episode = journal.finish("a", "whatever-this-is")
    assert episode is not None and episode.route == "pipeline"


def test_long_text_is_clipped_not_dropped(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    journal.open("x" * 50_000, session="s")
    episode = journal.finish("y" * 50_000)
    assert episode is not None
    assert 0 < len(episode.request) < 3000
    assert episode.request.startswith("xxx")


# --------------------------------------------------------------------------
# a call outside a turn belongs to no turn
# --------------------------------------------------------------------------

def test_calls_with_no_episode_open_are_dropped(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    journal.note_call("llm_generate", True, 1.0)     # no open episode
    assert journal.episodes() == []


def test_a_background_thread_does_not_contaminate_a_users_turn(tmp_path):
    """A research cycle calling tools while somebody types must not have
    its work attributed to their turn. That is why the open episode is
    per thread rather than per journal."""
    journal = Journal(tmp_path / "e.jsonl")
    journal.open("вопрос пользователя", session="s")

    done = threading.Event()

    def background():
        for _ in range(5):
            journal.note_call("llm_generate", True, 0.1)
        done.set()

    worker = threading.Thread(target=background)
    worker.start()
    done.wait(timeout=5)
    worker.join()

    journal.note_call("web_search", True, 0.2)
    episode = journal.finish("ответ")
    assert episode is not None
    assert episode.tools_used() == ["web_search"]


def test_abandon_leaves_nothing_behind(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    journal.open("q", session="s")
    journal.note_call("llm_generate", True, 0.1)
    journal.abandon()
    assert journal.finish("a") is None
    assert journal.episodes() == []


# --------------------------------------------------------------------------
# the journal must never break the thing it observes
# --------------------------------------------------------------------------

def test_an_unwritable_path_does_not_raise(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    journal = Journal(blocker / "sub" / "e.jsonl")
    journal.open("q", session="s")
    assert journal.finish("a") is not None      # reported, not raised
    assert journal.episodes() == []


def test_a_malformed_line_does_not_hide_the_rest(tmp_path):
    path = tmp_path / "e.jsonl"
    journal = Journal(path)
    journal.open("first", session="s")
    journal.finish("a")
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{ this is not json\n")
    journal.open("second", session="s")
    journal.finish("b")

    requests = [e.request for e in journal.episodes()]
    assert requests == ["first", "second"]


# --------------------------------------------------------------------------
# reading back
# --------------------------------------------------------------------------

def test_episodes_come_back_oldest_first_and_limit_takes_the_newest(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    for i in range(5):
        journal.open(f"q{i}", session="s")
        journal.finish(f"a{i}")
    assert [e.request for e in journal.episodes()] == [f"q{i}" for i in range(5)]
    assert [e.request for e in journal.episodes(limit=2)] == ["q3", "q4"]


def test_rotation_is_bounded_and_keeps_the_previous_file_readable(
        tmp_path, monkeypatch):
    """Two files, never more. The oldest history is dropped on purpose.

    At the real MAX_BYTES that ceiling is tens of thousands of turns, so
    what is lost is history from years back; what must not happen is a
    log that grows without bound on a machine that runs for months.
    """
    import mana.journal as journal_module
    monkeypatch.setattr(journal_module, "MAX_BYTES", 400)
    journal = Journal(tmp_path / "e.jsonl")
    for i in range(30):
        journal.open(f"вопрос номер {i}", session="s")
        journal.finish(f"ответ номер {i}")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["e.jsonl", "e.jsonl.1"]
    read_back = [e.request for e in journal.episodes()]
    assert "вопрос номер 29" in read_back          # the newest survives
    assert "вопрос номер 0" not in read_back       # the oldest is gone
    assert read_back == sorted(read_back, key=lambda r: int(r.split()[-1]))


def test_stats_describe_without_judging(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    journal.open("открой notepad", session="s1")
    journal.note_call("open_in_editor", True, 0.1)
    journal.finish("запущен", "app_intent")
    journal.open("что такое 1С", session="s2")
    journal.finish("это учётная система", "pipeline")

    stats = journal.stats()
    assert stats["episodes"] == 2
    assert stats["sessions"] == 2
    assert stats["with_no_calls"] == 1
    assert stats["by_route"] == {"app_intent": 1, "pipeline": 1}
    assert stats["by_tool"] == {"open_in_editor": 1}
    # No rates, no verdicts: this counts, it does not grade.
    assert not any("rate" in k or "score" in k for k in stats)


def test_round_trip_through_json(tmp_path):
    episode = Episode(episode_id="x", session="s", started=1.0,
                      request="q", answer="a", route="app_intent",
                      calls=[ToolCall("t", True, 0.5, "")], latency=2.0)
    restored = Episode.from_dict(json.loads(json.dumps(episode.as_dict())))
    assert restored == episode


# --------------------------------------------------------------------------
# the registry seam
# --------------------------------------------------------------------------

def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(FunctionTool(
        "works", "", lambda **kw: ToolResult(ok=True, output="fine")))
    registry.register(FunctionTool(
        "fails", "", lambda **kw: ToolResult(ok=False, error="нет такого файла")))
    return registry


def test_every_dispatch_is_observed(tmp_path):
    journal = Journal(tmp_path / "e.jsonl")
    registry = _registry()
    registry.observe(journal.note_call)

    journal.open("сделай", session="s")
    registry.call("works")
    registry.call("fails")
    episode = journal.finish("готово", "app_intent")

    assert episode is not None
    assert [(c.tool, c.ok) for c in episode.calls] == [("works", True), ("fails", False)]
    assert "нет такого файла" in episode.calls[1].error


def test_an_unknown_tool_is_recorded_as_a_failed_call(tmp_path):
    # "the turn asked for something that does not exist" is exactly the
    # kind of fact this record is kept for.
    journal = Journal(tmp_path / "e.jsonl")
    registry = _registry()
    registry.observe(journal.note_call)
    journal.open("сделай", session="s")
    registry.call("no_such_tool")
    episode = journal.finish("готово")
    assert episode is not None
    assert episode.calls[0].tool == "no_such_tool"
    assert episode.calls[0].ok is False


def test_a_broken_observer_never_breaks_a_tool_call():
    registry = _registry()

    def explode(*args, **kwargs):
        raise RuntimeError("observer is broken")

    registry.observe(explode)
    result = registry.call("works")
    assert result.ok is True and result.output == "fine"


def test_observing_can_be_turned_off():
    registry = _registry()
    seen = []
    registry.observe(lambda *a: seen.append(a))
    registry.call("works")
    registry.observe(None)
    registry.call("works")
    assert len(seen) == 1


# --------------------------------------------------------------------------
# the wiring
#
# This project's recurring failure is machinery that is built and connected
# to nothing -- it has happened at least five times, most recently with ten
# application tools registered and none reachable from a question. These
# tests exist so a journal that quietly stops being written fails here
# rather than being discovered months later as an empty file.
# --------------------------------------------------------------------------

def test_a_real_agent_turn_is_recorded(isolated_agent, tmp_path):
    isolated_agent.journal = Journal(tmp_path / "e.jsonl",
                                     version=isolated_agent.VERSION)
    isolated_agent.tools.observe(isolated_agent.journal.note_call)

    isolated_agent.solve_task("Сколько будет 2 плюс 2?")

    episodes = isolated_agent.journal.episodes()
    assert len(episodes) == 1
    assert episodes[0].request == "Сколько будет 2 плюс 2?"
    assert episodes[0].session == isolated_agent.session_id
    assert episodes[0].version == isolated_agent.VERSION
    # The point of the whole module: the turn's tool calls are on record.
    assert episodes[0].calls, "a turn that called no tool at all is suspicious here"


def test_the_agent_connects_the_registry_to_the_journal(isolated_agent):
    """The seam itself, not a stand-in for it."""
    assert isinstance(isolated_agent.journal, Journal)
    assert isolated_agent.tools._observer is not None


def test_a_failing_turn_does_not_leave_an_episode_open(isolated_agent, tmp_path):
    """An episode left open would collect the NEXT turn's calls and
    attribute them to this one."""
    isolated_agent.journal = Journal(tmp_path / "e.jsonl")

    def explode(_task):
        raise RuntimeError("the turn failed")

    isolated_agent._solve_task = explode
    with pytest.raises(RuntimeError):
        isolated_agent.solve_task("что-нибудь")

    assert isolated_agent.journal.current() is None
    assert isolated_agent.journal.episodes() == []


def test_the_route_is_read_off_the_result_not_guessed():
    from mana.agent_parts.core import CoreMixin
    route = CoreMixin._route_of
    assert route({"trace": {"clarification_requested": True}}) == "clarification"
    assert route({"memory_action": "STORE", "trace": {}}) == "memory_write"
    assert route({"trace": {"app_intent": {"action": "launch_onec"}}}) == "app_intent"
    assert route({"trace": {}}) == "pipeline"
    assert route({}) == "pipeline"


def test_the_installed_program_can_show_the_journal():
    """`MANA.exe --journal`. A diagnostic that needs the repository is one
    nobody on a user's machine can run -- which is how `--exchange` came
    to be unreachable from every installation that was not a checkout."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "--journal" in literals


def test_the_journal_path_is_isolated_like_every_other_state_path(isolated_config):
    """Caught by the suite writing 46 real episodes into the repository.

    The journal first took its location from `data_root()` directly, so
    `isolated_config` -- which redirects every other file MANA writes --
    had nothing to redirect, and running the tests polluted the checkout.
    A path MANA writes to belongs to Config; that is what makes it
    isolable.
    """
    from mana.config import Config

    assert "journal_path" in Config.STATE_PATH_FIELDS
    assert "mana_state" in isolated_config.journal_path


def test_the_agent_writes_where_the_config_says(isolated_agent):
    from pathlib import Path

    assert (Path(isolated_agent.journal.path)
            == Path(isolated_agent.config.journal_path))
