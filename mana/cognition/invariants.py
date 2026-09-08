"""
mana.cognition.invariants — finding MANA's own failures in the record.

Why this exists
---------------
Every defect this project has fixed was found by a person using MANA and
telling us. That does not scale, and it is why the same class of failure
keeps coming back in a new shape: a narration where an action should have
been, an answer repeating an older answer, a request understood by nobody.
Patching each instance by hand is years of work that never ends.

This is the other approach. An invariant is a predicate over a recorded
episode that, when violated, names a failure **without anybody being
asked**. It reads `mana/journal.py` -- what the turn requested, which
tools it actually dispatched, what it answered -- and reports
contradictions between those three.

Shadow, deliberately
--------------------
Nothing here can change an answer, block a turn, or feed the self-model.
It reads a record that has already been written and reports on it. That
is not timidity: a detector wired into the live path would have to be
right, and a detector that only reports may be wide, cheap and wrong
sometimes, which is what makes the next property possible.

The asymmetry that breaks the circularity
------------------------------------------
The obvious objection: "MANA cannot tell that it should have launched 1C
-- that is exactly the matcher that failed." True for the *actor*, false
for the *detector*, and the difference is what this module runs on.

  * `mana/apps/intent.py` must be narrow. A false positive there launches
    a program at somebody who asked a question.
  * A detector may be wide. A false positive here costs one line in a
    report that a person reads.

So the detector is a deliberately over-broad version of the actor, and
**the gap between them is the measurement**: every episode where the wide
detector fires and the narrow actor did nothing is a candidate failure.
That generalises past this one case -- for any narrow mechanism, a wider
shadow twin measures what the mechanism is missing.

Two kinds of verdict, never mixed
----------------------------------
`MECHANICAL` violations are decided by comparing recorded fields: which
tools ran, whether an answer repeats an earlier one above a stated
threshold. No opinion about meaning enters.

`PATTERN` violations depend on matching free text against a word list.
They are useful and they are guesses. Reporting them as the same kind of
fact would be exactly the failure this project keeps having -- a claim
resting on nothing -- so the kind travels with every violation, and
anything downstream that turns findings into training material may use
`MECHANICAL` alone without a person confirming.

Measured on the first seven real episodes
------------------------------------------
    "я хочу поработать с 1С запусти пожалуйста конфигуратор..."
        -> instructions for the human, onec_launch not called
    "хорошо, спасибо"
        -> byte-identical to the answer given three turns earlier
           (similarity 1.00 against that one, 0.018 against the
           previous turn, which is all the live guard compares)

Both are found here without anybody reporting them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..echo_guard import similarity, SAME_ANSWER, DIFFERENT_QUESTION, MIN_LENGTH
from ..journal import Episode
from ..outcome import CONTRADICTED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Decided by comparing recorded fields. Reproducible, no opinion about
#: meaning. Downstream may use these without a person confirming.
MECHANICAL = "mechanical"

#: Depends on matching free text against a word list. Useful, and a guess.
PATTERN = "pattern"

#: Tools that change something outside MANA. A turn that asked for an
#: action and dispatched none of these did not act, whatever it said.
#:
#: An allowlist, not a denylist -- the project's recurring principle. A
#: new tool is not an action until somebody says so here, which fails
#: towards reporting a violation rather than towards missing one.
ACTION_TOOLS = frozenset({
    "open_in_editor", "onec_launch", "onec_create_base",
    "onec_confirm_write", "write_document", "run_code",
})

#: First person, asserting the action happened or is happening right now.
#: Infinitives are excluded on purpose: "вы хотели запустить конфигуратор"
#: describes the request, "запускаю конфигуратор" claims the deed.
_CLAIMED = re.compile(
    r"\b(?:"
    r"откры(?:ваю|л|ла)|"
    r"запуска(?:ю|ем)|запусти(?:л|ла)|"
    r"созда(?:ю|ём|л|ла)|"
    r"сохрани(?:л|ла)|записа(?:л|ла)|"
    r"выполни(?:л|ла)|сдела(?:л|ла)"
    r")\b", re.IGNORECASE)

#: Verbs that ask for something to happen on this machine. Anywhere in
#: the message, unlike `apps/intent.py`, which requires the start of it --
#: that narrowness is precisely what this is here to measure.
_ASKED_TO_ACT = re.compile(
    r"\b(?:запусти|запустить|запустишь|открой|открыть|откроешь|открыла|"
    r"стартуй|включи|создай|создать)\b", re.IGNORECASE)

#: An interrogative opening the message: it asks *about* an action rather
#: than *for* one. "как запустить 1С на новом компьютере?" is advice, and
#: a detector that flagged it would fill the report with questions until
#: nobody read it -- measured, it was the first false positive found.
#:
#: Only at the start, and only these words. "Можешь запустить 1С?" is a
#: request wearing a question mark, so the question mark alone is not the
#: test; excluding on it would lose real requests, which for a detector
#: is the expensive mistake.
_ASKS_ABOUT = re.compile(
    r"^\s*(?:как|каким образом|почему|зачем|что такое|что за|"
    r"где|когда|можно ли|стоит ли|возможно ли|в чём|в чем)\b",
    re.IGNORECASE)

#: Things on this machine an action could target, beyond the base names
#: read from the machine itself.
_TARGETS = re.compile(
    r"(?:notepad\+\+|notepad|блокнот|конфигуратор|1\s*[сcС]\b|"
    r"word|excel|документ|таблиц)", re.IGNORECASE)

_HOMOGLYPHS = str.maketrans("АВЕКМНОРСТУХаветкмнорсух",
                            "ABEKMHOPCTUXabetkmhopcux")


def _fold(text: str) -> str:
    return (text or "").lower().translate(_HOMOGLYPHS)


@dataclass(frozen=True)
class Violation:
    """One failure found in the record, with what decided it."""
    invariant: str
    kind: str                      # MECHANICAL | PATTERN
    episode_id: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"invariant": self.invariant, "kind": self.kind,
                "episode_id": self.episode_id, "reason": self.reason,
                "evidence": dict(self.evidence)}


# --------------------------------------------------------------------------
# the invariants
#
# Each takes one episode plus the episodes before it in the same session,
# and returns a Violation or None. Nothing else: an invariant that needed
# to call a model or ask a person would not be an invariant.
# --------------------------------------------------------------------------

def repeats_earlier_answer(episode: Episode,
                           earlier: Sequence[Episode]) -> Optional[Violation]:
    """The answer repeats one already given, to a different question.

    MECHANICAL: two recorded strings and a stated threshold.

    Against the WHOLE session, which is the difference from the live
    guard. `echo_guard.previous_exchange` stops at the first previous
    answer, so a repeat of anything older than one turn is invisible to
    it -- measured: an answer identical to one three turns back scored
    1.00 against that turn and 0.018 against the turn the guard actually
    compared, and passed.

    Repetition is not always wrong: asked the same thing twice, the same
    answer is correct. So the questions must differ too.
    """
    answer = (episode.answer or "").strip()
    if len(answer) < MIN_LENGTH:
        return None

    for previous in reversed(earlier):
        if previous.session != episode.session:
            continue
        same = similarity(answer, previous.answer)
        if same < SAME_ANSWER:
            continue
        asked = similarity(episode.request, previous.request)
        if asked >= DIFFERENT_QUESTION:
            continue                      # the same question, fairly
        return Violation(
            "repeats_earlier_answer", MECHANICAL, episode.episode_id,
            f"ответ повторяет ответ, данный ранее в этой сессии "
            f"(похожесть {same:.2f} при пороге {SAME_ANSWER}), "
            f"хотя вопрос был другой (похожесть {asked:.2f})",
            {"answer_similarity": round(same, 3),
             "question_similarity": round(asked, 3),
             "repeated_from": previous.episode_id,
             "turns_back": len(earlier) - list(earlier).index(previous),
             "earlier_request": previous.request[:120]})
    return None


def claimed_action_without_acting(episode: Episode,
                                  earlier: Sequence[Episode]) -> Optional[Violation]:
    """The answer says it did something; no action tool succeeded.

    PATTERN on the claim, MECHANICAL on the tools -- reported as PATTERN,
    because the weaker half decides.

    This is the failure the whole journal was built for: "Открой
    Notepad++" answered "Открываю Notepad++." with zero tool calls.
    """
    claim = _CLAIMED.search(episode.answer or "")
    if not claim:
        return None
    acted = [c.tool for c in episode.calls if c.ok and c.tool in ACTION_TOOLS]
    if acted:
        return None
    return Violation(
        "claimed_action_without_acting", PATTERN, episode.episode_id,
        f"ответ утверждает действие («{claim.group(0)}»), "
        f"но ни один инструмент действия не отработал",
        {"claim": claim.group(0),
         "tools_called": episode.tools_used(),
         "action_tools_ok": acted})


def actionable_request_not_acted_on(
        episode: Episode, earlier: Sequence[Episode],
        known_targets: Sequence[str] = ()) -> Optional[Violation]:
    """A request to act on this machine, and nothing acted.

    PATTERN. The wide twin of `apps/intent.py`: the verb may sit anywhere
    in the message, and a base name matches by word stems rather than as
    a literal substring, so "конфигуратор информационной базы" reaches
    "Информационная база" -- which the narrow actor does not.

    The gap between the two is the point. Where this fires and the actor
    did nothing, either the actor is too narrow or this is too wide, and
    a person reading the report can tell which in one glance.
    """
    text = episode.request or ""
    if not _ASKED_TO_ACT.search(text):
        return None
    if _ASKS_ABOUT.match(text):
        return None

    target = ""
    found = _TARGETS.search(text)
    if found:
        target = found.group(0)
    else:
        target = _stemmed_target(text, known_targets)
    if not target:
        return None

    acted = [c.tool for c in episode.calls if c.ok and c.tool in ACTION_TOOLS]
    if acted:
        return None
    return Violation(
        "actionable_request_not_acted_on", PATTERN, episode.episode_id,
        f"просьба сделать что-то на этой машине («{target}»), "
        f"но ни один инструмент действия не отработал",
        {"target": target, "route": episode.route,
         "tools_called": episode.tools_used()})


def _stemmed_target(text: str, known_targets: Sequence[str]) -> str:
    """A known name the message mentions, tolerating Russian inflection.

    Every word stem of the name must appear. Requiring all of them is
    what keeps "база данных" from matching "Информационная база": the
    generic word is shared, the distinctive one is not.
    """
    folded = _fold(text)
    for name in sorted(known_targets, key=len, reverse=True):
        words = [w for w in re.split(r"[\s\-_]+", _fold(name)) if len(w) > 2]
        if not words:
            continue
        # `max(3, ...)` and not `max(4, ...)`: at four the floor equalled
        # the whole word for "база", so no stem was taken at all and
        # "базы" -- the form anybody actually writes -- missed. The short
        # generic word can afford a short stem because every word of the
        # name must match, and the distinctive one carries the
        # discrimination: "база данных" still fails on "информацион".
        if all(w[:max(3, len(w) - 2)] in folded for w in words):
            return name
    return ""


def acted_but_state_disagrees(episode: Episode,
                              earlier: Sequence[Episode]) -> Optional[Violation]:
    """A tool ran, looked at the machine, and found the wrong state.

    The only invariant here that reads no text at all. The other three
    compare an answer against a request or against an earlier answer,
    which is a judgement however carefully it is made; this one reads a
    verdict the tool itself reached by comparing what it was asked for
    against what it found (see mana.outcome). There is nothing left to
    interpret, and no word list to be wrong.

    `unobserved` is not a violation. A tool that could not check is not a
    tool that got it wrong, and folding the two together would rebuild
    the assumption the outcome layer exists to remove.
    """
    disagreed = [call.tool for call in episode.calls
                 if call.verified == CONTRADICTED]
    if not disagreed:
        return None
    return Violation(
        invariant="acted_but_state_disagrees",
        kind=MECHANICAL,
        episode_id=episode.episode_id,
        reason=("инструмент сам сообщил, что состояние машины не совпало с "
                "запрошенным: " + ", ".join(disagreed)),
        evidence={"tools": disagreed, "answer": episode.answer[:200]})


#: Every invariant, in the order a report lists them. Explicit rather than
#: discovered, so one added without a line here fails a test instead of
#: quietly never running -- this project's recurring failure is machinery
#: that is built and connected to nothing.
INVARIANTS: Sequence[Callable[..., Optional[Violation]]] = (
    repeats_earlier_answer,
    claimed_action_without_acting,
    actionable_request_not_acted_on,
    acted_but_state_disagrees,
)


def known_targets() -> List[str]:
    """Names this machine really has. Empty on any trouble."""
    try:
        from ..apps.onec_launch import bases
        return [str(b.get("name") or "") for b in bases() if b.get("name")]
    except Exception:
        return []


def scan(episodes: Sequence[Episode],
         targets: Optional[Sequence[str]] = None) -> List[Violation]:
    """Every violation in a run of episodes, oldest first.

    One episode may violate several invariants; each is reported, because
    "narrated an action" and "was asked to act and did not" are different
    failures with different fixes, and collapsing them would hide one.
    """
    if targets is None:
        targets = known_targets()
    found: List[Violation] = []
    for index, episode in enumerate(episodes):
        found.extend(check(episode, episodes[:index], targets))
    return found


def check(episode: Episode, earlier: Sequence[Episode] = (),
          targets: Optional[Sequence[str]] = None) -> List[Violation]:
    """Every invariant one episode violates, in report order.

    Split out of `scan` because the end of a turn has one episode and its
    own prefix, not a run to walk. `scan` calls this, so which invariants
    apply and how each is called is written down once -- two copies of
    that would drift the first time an invariant took a new argument.
    """
    if targets is None:
        targets = known_targets()
    found: List[Violation] = []
    for invariant in INVARIANTS:
        try:
            if invariant is actionable_request_not_acted_on:
                violation = invariant(episode, earlier, targets)
            else:
                violation = invariant(episode, earlier)
        except Exception:
            # A broken invariant must not stop the scan; it is the least
            # important thing running.
            continue
        if violation is not None:
            found.append(violation)
    return found


def summarise(episodes: Sequence[Episode],
              violations: Sequence[Violation]) -> Dict[str, Any]:
    """The shape of what was found, with the counts stated plainly.

    Deliberately reports the denominator. "12 нарушений" means nothing
    without how many turns were looked at, and a rate quoted off seven
    episodes is not a rate -- so both numbers travel together and the
    caller can see the sample is small.
    """
    by_invariant: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    for violation in violations:
        by_invariant[violation.invariant] = by_invariant.get(violation.invariant, 0) + 1
        by_kind[violation.kind] = by_kind.get(violation.kind, 0) + 1
    affected = {v.episode_id for v in violations}
    return {
        "episodes": len(episodes),
        "episodes_with_a_violation": len(affected),
        "violations": len(violations),
        "by_invariant": by_invariant,
        "by_kind": by_kind,
        "invariants_run": [f.__name__ for f in INVARIANTS],
    }
