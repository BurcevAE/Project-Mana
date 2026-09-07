"""
mana.echo_guard — catching an answer that merely repeats the last one.

The failure, reproduced on a clean state directory
---------------------------------------------------
    Какая погода в Воронеже?  -> погода в Воронеже, +16°, ветер 4,9 м/с
    Привет, Мана              -> погода в Воронеже, +16°, ветер 4,9 м/с

Not a coincidence and not concurrency. `[RECENT CONVERSATION]` puts prior
turns into the prompt verbatim, as `MANA: <answer>`, and a small local
model reading six thousand characters of preamble, tools and recalled
history finds that the most answer-shaped text in front of it is its own
previous answer. When the new question offers little to work with -- a
greeting, an acknowledgement -- copying wins.

Why a check and not a better prompt
------------------------------------
Instructing the model to treat history as history is a hope. This is a
measurement: the answer either repeats a previous one or it does not, and
that is decidable after the fact. The project already refuses to ship
hopes where a check is available -- the same reason the allowlist in
`onec_dataset` is not a list of forbidden words.

Repetition is not always wrong
-------------------------------
Asked the same question twice, the same answer is correct. So the
comparison is against the previous answer AND the previous question: a
repeat only counts when the questions differ. Getting this backwards
would make MANA refuse to be consistent, which is worse than the bug.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: How alike two answers must be to count as the same one.
#:
#: Set from the asymmetry, not from taste. A false positive costs one
#: extra call and nothing else -- if the retry repeats too, the original
#: answer is returned unchanged. A false negative costs a wrong answer
#: shown to a person. So the bar leans permissive.
#:
#: Measured on the real failure: a verbatim repeat scored 0.98, and the
#: milder form -- "Привет! " followed by the same weather report --
#: scored 0.856 and slipped under a 0.90 bar.
SAME_ANSWER = 0.80

#: How different two questions must be before repetition is suspicious.
#: "Какая погода в Воронеже?" and "а в Москве?" are different questions
#: with legitimately similar answers, so the bar sits low.
DIFFERENT_QUESTION = 0.75

#: Below this an answer is too short for similarity to mean anything --
#: "Не нашлось." repeated is not evidence of anything.
MIN_LENGTH = 60


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def similarity(left: str, right: str) -> float:
    left, right = _normalise(left), _normalise(right)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def previous_exchange(memory: Any, session_id: str,
                      look_back: int = 6) -> Tuple[str, str]:
    """The last question and the answer it got, or two empty strings.

    Read from the store rather than from anything held in the process,
    because the process may have restarted between the two turns and the
    echo survives that -- it lives in the prompt, which is rebuilt from
    the store every time.
    """
    try:
        recent = list(memory.recent(session_id, look_back))
    except Exception:
        return "", ""

    answer = question = ""
    for record in reversed(recent):
        kind = record.get("kind")
        if not answer and kind == "MANA_RESPONSE":
            answer = str(record.get("content") or "")
            continue
        if answer and kind == "USER_MESSAGE":
            question = str(record.get("content") or "")
            break
    return question, answer


def previous_exchanges(memory: Any, session_id: str,
                       limit: int = 1, look_back: int = 40) -> list:
    """The last `limit` question/answer pairs, newest first.

    `previous_exchange` returns one, which is what the live guard has
    always compared against -- and why a repeat older than a single turn
    is invisible to it. Measured: an answer identical to one three turns
    back scored 1.00 against that turn and 0.018 against the turn the
    guard looked at, and passed.

    How many are actually compared is `policy.echo_lookback`, whose
    default is 1: today's behaviour exactly, until a candidate carrying a
    larger value is accepted.
    """
    try:
        recent = list(memory.recent(session_id, look_back))
    except Exception:
        return []

    pairs, answer = [], ""
    for record in reversed(recent):
        kind = record.get("kind")
        if not answer and kind == "MANA_RESPONSE":
            answer = str(record.get("content") or "")
            continue
        if answer and kind == "USER_MESSAGE":
            pairs.append((str(record.get("content") or ""), answer))
            answer = ""
            if len(pairs) >= max(1, int(limit)):
                break
    return pairs


def repeats_any_previous(task: str, answer: str,
                         pairs: Sequence[Tuple[str, str]]) -> Optional[Dict[str, Any]]:
    """The first earlier exchange this answer repeats, if any.

    Newest first, so the report names the nearest repeat rather than an
    arbitrary one.
    """
    for question, previous in pairs:
        found = repeats_previous(task, answer, question, previous)
        if found is not None:
            return found
    return None


def repeats_previous(task: str, answer: str, previous_question: str,
                     previous_answer: str) -> Optional[Dict[str, Any]]:
    """Details of the repetition, or None if this answer is its own.

    Returns the numbers rather than a bare boolean so a caller can record
    WHY it decided that, and so a threshold that turns out to be wrong
    can be argued with from evidence instead of taste.
    """
    if len(_normalise(answer)) < MIN_LENGTH:
        return None
    if not previous_answer:
        return None

    # Thresholds come from the policy in force, whose defaults are the
    # constants above -- so nothing changes until a candidate is accepted.
    from . import policy as policy_mod
    same_answer = float(policy_mod.get("echo_same_answer"))
    different_question = float(policy_mod.get("echo_different_question"))

    answer_likeness = similarity(answer, previous_answer)
    if answer_likeness < same_answer:
        return None

    question_likeness = similarity(task, previous_question)
    if question_likeness >= different_question:
        # Same question, same answer. That is consistency, not an echo.
        return None

    return {
        "answer_similarity": round(answer_likeness, 3),
        "question_similarity": round(question_likeness, 3),
        "previous_question": previous_question[:120],
    }
