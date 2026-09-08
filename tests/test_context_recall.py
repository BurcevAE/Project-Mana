"""What reaches the model as "the conversation".

Measured on a real session before this existed. Asked "Мана, привет!",
the model was handed its own routing log dressed as speech:

    USER:
    MANA: Не нашлось информации о погоде в Воронеже... 气象网站
    MANA: task= | route= | verified=none
    USER: Привет, Мана
    MANA: task=Привет, Мана | route= | verified=none

A quarter of the recalled window taught it that MANA says
"task= | route= | verified=none". The second render site printed the raw
kind name instead, so the transcript read "USER_MESSAGE:" and
"DECISION:".
"""
from __future__ import annotations

import pytest


def records():
    """A window shaped like the real one: log interleaved with speech."""
    return [
        {"kind": "USER_MESSAGE", "content": ""},
        {"kind": "MANA_RESPONSE", "content": "Не нашлось информации о погоде."},
        {"kind": "DECISION", "content": "task= | route= | verified=none"},
        {"kind": "USER_MESSAGE", "content": "Привет, Мана"},
        {"kind": "MANA_RESPONSE", "content": "Здравствуйте. Чем помочь?"},
        {"kind": "DECISION", "content": "task=Привет, Мана | route= | verified=none"},
    ]


def turns(isolated_agent, limit=12):
    return isolated_agent._spoken_turns(records(), limit)


# --------------------------------------------------------------------------
# the log is not speech
# --------------------------------------------------------------------------

def test_the_routing_log_never_reaches_the_transcript(isolated_agent):
    rendered = "\n".join(turns(isolated_agent))
    assert "route=" not in rendered
    assert "verified=none" not in rendered
    assert "DECISION" not in rendered


def test_a_raw_kind_name_never_reaches_the_transcript(isolated_agent):
    """The other render site printed the kind itself, so the model read
    "USER_MESSAGE:" as the speaker."""
    rendered = "\n".join(turns(isolated_agent))
    assert "USER_MESSAGE" not in rendered
    assert "MANA_RESPONSE" not in rendered


def test_speech_is_labelled_by_who_said_it(isolated_agent):
    rendered = turns(isolated_agent)
    assert "USER: Привет, Мана" in rendered
    assert "MANA: Здравствуйте. Чем помочь?" in rendered


def test_an_unknown_kind_is_dropped_rather_than_guessed(isolated_agent):
    """An allowlist: a kind added later goes missing rather than showing
    up as MANA talking to itself."""
    extra = records() + [{"kind": "SOMETHING_NEW", "content": "внутреннее"}]
    rendered = "\n".join(isolated_agent._spoken_turns(extra, 12))
    assert "внутреннее" not in rendered


def test_an_empty_turn_costs_no_line(isolated_agent):
    """It says nothing and still spends a line of the window."""
    for line in turns(isolated_agent):
        assert line not in ("USER: ", "MANA: ")


# --------------------------------------------------------------------------
# the limit counts turns
# --------------------------------------------------------------------------

def test_the_limit_is_applied_after_filtering(isolated_agent):
    """Slicing first spent a quarter of the window on records that are
    not speech."""
    assert len(isolated_agent._spoken_turns(records(), 3)) == 3
    kept = isolated_agent._spoken_turns(records(), 3)
    assert all("route=" not in line for line in kept)


def test_the_newest_turns_are_the_ones_kept(isolated_agent):
    kept = isolated_agent._spoken_turns(records(), 2)
    assert kept == ["USER: Привет, Мана", "MANA: Здравствуйте. Чем помочь?"]


def test_a_zero_limit_recalls_nothing(isolated_agent):
    """The echo retry sets it to zero to remove what is being copied."""
    assert isolated_agent._spoken_turns(records(), 0) == []


def test_no_records_is_no_transcript(isolated_agent):
    assert isolated_agent._spoken_turns([], 12) == []
    assert isolated_agent._spoken_turns(None, 12) == []


# --------------------------------------------------------------------------
# both render sites go through it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["_conversation_recall_context",
                                    "_render_evidence_context"])
def test_both_render_sites_use_the_allowlist(method):
    """There were two, wrong in different ways. A third would be a third
    chance to show the log to the model."""
    import inspect

    from mana.agent_parts import context

    source = inspect.getsource(getattr(context.ContextMixin, method))
    assert "_spoken_turns" in source
    assert 'kind")=="USER_MESSAGE"' not in source
