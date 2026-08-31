"""
mana.intent — detects ONE thing: whether a turn is about the conversation
itself rather than about the world.

Why only one thing. Two rounds of measurement showed that no similarity
threshold can separate "asking about X" from "complaining that you gave
me X". Measured with all-MiniLM-L6-v2 against a stored news entry:

    "Разве я просил новости?"            embedding 0.655   lexical 0.500
    "что там было про ИИ в новостях"     embedding 0.563   lexical 0.354

The complaint scores HIGHER than the weakest genuine request, on both
scoring paths independently. Raising any floor to exclude the complaint
rejects real queries first.

But the deeper point is that this was never a memory-relevance problem.
"Разве я просил новости?" refers to the PREVIOUS ASSISTANT TURN, not to
anything in long-term storage. Filtering storage harder cannot fix a
question that is not about storage. So the fix is to notice the reference
and answer from the recent exchange instead of searching memory at all.

Scope discipline, stated deliberately: this module does NOT implement a
general intent taxonomy. A nine-category classifier was considered and
rejected -- we have measured examples for exactly one distinction, and
inventing eight more categories before measuring them is the same mistake
that produced two wrong relevance thresholds (0.25, then 0.45). Add a
category when there is data demanding it.

Honest limitations:
  * This is lexical pattern matching, not language understanding. It will
    miss paraphrases it has no marker for, and it can fire on a sentence
    that merely resembles one.
  * It requires a previous assistant turn to exist. At the start of a
    session nothing is a reference to a previous turn, so detection is
    suppressed -- that is a correctness rule, not an optimisation.
  * Accuracy is measured in tests/test_intent.py against a labelled set.
    When you change the patterns, re-run it; the number is the point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


#: Markers that a turn is commenting on the exchange rather than asking
#: about the world. Grouped by the kind of reference so failures are
#: diagnosable: when a case is missed, the group tells you what is absent.
_REJECTION_MARKERS = (
    r"\bразве\s+я\s+(?:просил|спрашивал|хотел)",
    r"\bя\s+не\s+(?:просил|спрашивал|это\s+спрашивал)",
    r"\bя\s+(?:же\s+)?не\s+про\s+это\b",
    r"\bхватит\s+(?:про|об|о)\b",
    r"\bпри\s+чём\s+(?:тут|здесь)\b",
    r"\bэто\s+не\s+то\b",
    r"\bне\s+то,?\s+что\s+я\b",
)

_ASSISTANT_BEHAVIOUR_MARKERS = (
    # "зачем ты" alone was too broad -- it fired on "зачем ты нужен вообще?",
    # an honest question about the agent, not a correction. Require a
    # past-tense verb naming something it actually did, so the marker is
    # about a specific prior action rather than about the agent in general.
    r"\bзачем\s+ты\s+(?:дал|дала|выдал|сказал|сказала|написал|написала|"
    r"вспомнил|вспомнила|принёс|принесла|полез|полезла)\b",
    r"\bпочему\s+ты\s+(?:опять|снова|всё\s+время|вечно)\b",
    r"\bты\s+(?:опять|снова)\s+(?:про|об|о|дал|начал|вспомнил)\b",
    r"\bс\s+чего\s+ты\s+(?:взял|решил)\b",
    r"\bкто\s+тебя\s+просил\b",
    r"\bты\s+меня\s+не\s+понял\b",
    r"\bты\s+ошибся\b",
)

_CLARIFICATION_MARKERS = (
    r"\bя\s+имел\s+в\s+виду\b",
    r"\bя\s+спрашивал\s+(?:про|о|об)\b",
    r"\bя\s+про\s+другое\b",
    r"\bпереспрошу\b",
)

_MARKER_GROUPS = {
    "rejection": _REJECTION_MARKERS,
    "assistant_behaviour": _ASSISTANT_BEHAVIOUR_MARKERS,
    "clarification": _CLARIFICATION_MARKERS,
}

_COMPILED = {name: [re.compile(p, re.I) for p in pats] for name, pats in _MARKER_GROUPS.items()}


@dataclass
class ConversationReference:
    """Result of the check. `is_reference` is the decision; the rest is
    there so a wrong decision can be diagnosed rather than guessed at."""
    is_reference: bool
    kind: str = ""            # which marker group fired
    matched: str = ""         # the pattern that fired
    reason: str = ""          # why it did or did not fire

    def __bool__(self) -> bool:
        return self.is_reference


def refers_to_previous_turn(text: str, has_previous_assistant_turn: bool = True) -> ConversationReference:
    """Is this turn about what was just said, rather than about the world?

    `has_previous_assistant_turn` is required, not optional politeness: at
    the start of a session there is nothing to refer back to, so a
    sentence that looks like a complaint cannot be one. Suppressing
    detection there prevents the failure mode where an opening message is
    mistaken for a correction and memory is skipped for no reason.
    """
    query = (text or "").strip()
    if not query:
        return ConversationReference(False, reason="empty input")
    if not has_previous_assistant_turn:
        return ConversationReference(False, reason="no previous assistant turn to refer to")
    for kind, patterns in _COMPILED.items():
        for pattern in patterns:
            if pattern.search(query):
                return ConversationReference(True, kind=kind, matched=pattern.pattern,
                                              reason=f"matched {kind} marker")
    return ConversationReference(False, reason="no conversation-reference marker found")
