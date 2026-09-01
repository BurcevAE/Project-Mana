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
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1.1"


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
class AmbiguousReference:
    """Result of the ambiguity check.

    `is_ambiguous` is the decision; `candidates` are the competing topics
    it would ask about. Both are needed: the question MANA asks is built
    from the candidates by code, not left to the model to invent.
    """
    is_ambiguous: bool
    candidates: List[str] = field(default_factory=list)
    reason: str = ""

    def __bool__(self) -> bool:
        return self.is_ambiguous


#: A follow-up that names no subject of its own. These are the phrasings
#: that inherit their topic from the conversation -- and therefore the only
#: ones that can be ambiguous. A question that names its subject is never
#: ambiguous no matter how many topics preceded it.
#: NOTE: these are PREFIXES, so there is no trailing \b. The first version
#: had one, and `новост\b` therefore failed to match "новости" -- the exact
#: phrase from the reported scenario ("Какие последние новости?") was not
#: detected at all. A word boundary after a stem is a contradiction.
_TOPICLESS_FOLLOWUP = re.compile(
    r"\b(последн|свеж|новост|подробн|детал|ещё|еще|дальше|продолж|результат|"
    r"а\s+что|а\s+как|что\s+там|как\s+там)", re.I)


#: A turn that inherits its topic is SHORT. Found in live use: a 17-word
#: instruction -- "Я не говорил что матч был сегодня, узнай когда был
#: последний матч и с каким счётом закончился" -- was asked about, because
#: it names its subject with ordinary nouns ("матч", "счёт") rather than a
#: proper noun, and extract_entities only sees names and acronyms. So
#: "extract_entities returned nothing" does NOT mean "no subject named".
#: Length is the signal that separates them: a genuine topic-inheriting
#: follow-up is a few words ("а что там?", "какие последние новости?"),
#: while a sentence long enough to explain itself carries its own subject.
MAX_FOLLOWUP_WORDS = 6


def is_ambiguous_followup(text: str, recent_topics: Sequence[Sequence[str]],
                           min_candidates: int = 2,
                           max_words: int = MAX_FOLLOWUP_WORDS) -> AmbiguousReference:
    """Would answering this require guessing WHICH earlier topic is meant?

    Asks only when both conditions hold:

      1. the question names no subject itself -- extract_entities finds
         nothing, so it must inherit one from the conversation;
      2. the conversation offers >= `min_candidates` distinct topics, so
         there is a real choice rather than an obvious one.

    Condition 2 matters as much as condition 1. With a single topic in
    play there is nothing to guess and asking would be pure friction --
    and friction is the real cost here: a clarifying question is a refusal
    to answer, so over-asking is worse than the occasional wrong guess it
    prevents. That is why this is deliberately narrow, and why the
    false-ask rate is measured (tests/test_intent.py) rather than assumed.

    `recent_topics` is a sequence of per-turn entity lists, most recent
    first, as produced by graph_memory.extract_entities.
    """
    query = (text or "").strip()
    if not query:
        return AmbiguousReference(False, reason="empty input")
    from .graph_memory import extract_entities
    if extract_entities(query):
        return AmbiguousReference(False, reason="the question names its own subject")
    words = len(query.split())
    if words > max_words:
        return AmbiguousReference(
            False, reason=f"{words} words -- long enough to carry its own subject")
    if not _TOPICLESS_FOLLOWUP.search(query):
        return AmbiguousReference(False, reason="not a topic-inheriting follow-up")

    # One candidate per earlier TURN, not per entity. Counting entities
    # made "Какой результат матча Спартак — ЦСКА?" look like two competing
    # topics and produced "про ЦСКА или Спартак?" -- offering a choice
    # between two teams in a single question the user asked once. A
    # candidate is an earlier subject the user raised, and one question
    # raises one subject however many names it contains.
    candidates: List[str] = []
    for topics in recent_topics:
        if not topics:
            continue
        label = topics[0]
        if label not in candidates:
            candidates.append(label)
    if len(candidates) < min_candidates:
        return AmbiguousReference(False, candidates=candidates,
                                   reason=f"only {len(candidates)} topic(s) in play -- nothing to guess between")
    return AmbiguousReference(True, candidates=candidates,
                               reason=f"{len(candidates)} competing topics and no subject named")


def format_clarifying_question(candidates: Sequence[str], limit: int = 3) -> str:
    """Build the question from the detected topics, in code.

    Deliberately not delegated to the model: the whole point is to stop
    guessing, and a generated question could invent an option that was
    never discussed.
    """
    # Topics are stored lowercased (they are graph keys), but "про ии,
    # цска" reads badly. Restore a plausible surface form: short all-letter
    # tokens are acronyms, everything else is a proper noun. A heuristic,
    # and occasionally it will capitalise something odd -- preferable to
    # showing the user raw index keys.
    def _surface(token: str) -> str:
        return token.upper() if len(token) <= 4 else token.capitalize()

    shown = [_surface(c) for c in candidates[:limit]]
    if len(shown) < 2:
        return "Уточни, пожалуйста, о чём именно речь?"
    listed = ", ".join(shown[:-1]) + f" или {shown[-1]}"
    return f"Уточни, пожалуйста: про {listed}?"


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
