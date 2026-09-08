"""
mana.cognition.hypothesis — from a diagnosed conflict to a change nobody wrote.

The line this is on the far side of
-------------------------------------
A knob is a change a person thought of, wrote down and gave options to.
Everything MANA has adopted until now was a choice among those. This
module is the other thing: it reads *why* a decision failed, states a
claim about the failure that is not a parameter, and builds the concrete
change that claim implies.

The easy version of this would be a language model producing a hundred
variants until one wins. That is not a hypothesis, it is a search with a
holdout for a scoreboard, and enough variants will beat any holdout. So
there is no model here and the space is deliberately narrow: one claim per
diagnosed shape, and at most `MAX_CANDIDATES` concrete forms of it, mined
under conditions strict enough that most groups yield none.

The two shapes, and what each one licenses
--------------------------------------------
`compiler.diagnose` separates them, and they are different claims:

  AMBIGUOUS -- a feature of another class fired and decided, with nothing
    of the right class to compete. The claim: where features co-occur,
    the more specific one must decide, not the one that happens to be
    consulted first. What it licenses: a discriminator for the right
    class, seated before the built-in list.

  ABSENT -- nothing fired at all and the answer came from the fallback.
    The claim: the vocabulary has no way to name this class, so no
    ordering can fix it and a feature has to exist. What it licenses: a
    discriminator, and this time its novelty is the whole point.

Both end in a marker with a seat, which is what makes them testable by the
same protocol. The claims differ, and the difference is what stops this
being a search: an ABSENT diagnosis cannot be answered by reordering, and
an AMBIGUOUS one cannot be answered by a marker that also appears
elsewhere.

Mined on discovery, and nowhere else
--------------------------------------
Candidates are mined from the discovery seeds and never from a hidden
split. The thresholds are stated here rather than tuned per run, and they
are strict on purpose: a marker must be in nearly every item of its class
and almost never outside it. A marker that leaks is exactly how a rule
fixes one group by breaking another, which is the failure the last
adoption made and the no-regression check caught.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .compiler import ABSENT, AMBIGUOUS, diagnose
from .rules import Rule

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: A marker must appear in at least this share of its class's items.
#: High: a discriminator that covers half a class is a second failure
#: mode waiting, not a rule.
MIN_SUPPORT = 0.90

#: And in at most this share of everything else. Near zero because a
#: marker that leaks is how a rule fixes one group by breaking another --
#: measured once already, when a text check took `code` from 100% to 37%.
MAX_LEAK = 0.02

#: How many concrete forms of one claim may be carried forward. Small on
#: purpose: the number of candidates is the size of the search, and a
#: search large enough will beat a holdout by luck.
MAX_CANDIDATES = 3

#: The longest a marker may be. Length is the tiebreak between equally
#: scoring markers, and on a templated corpus the longest substring is
#: the template: the first run of this mined a marker ending in a
#: line break and the word after it, which covers every logic task,
#: leaks nowhere, and is a fact about how the prompt is laid out
#: rather than about what the task is. A marker crossing a line
#: break is rejected for the same reason.
MAX_MARKER = 40

#: Words too common to be anybody's discriminator, and punctuation that
#: would make a marker match everything.
_TOKEN = re.compile(r"[\w«»:]+", re.UNICODE)
_STOP = frozenset({
    "и", "в", "во", "на", "с", "со", "по", "за", "из", "к", "у", "о", "об",
    "the", "a", "of", "to", "in", "что", "как", "не", "для", "это", "если",
})


@dataclass(frozen=True)
class Hypothesis:
    """A claim about why a decision fails, and what it licenses.

    `claim` is prose because it is addressed to a person, and `shape` is
    the machine-readable half. The two are kept together so a reader can
    disagree with the sentence and a test can check the branch.
    """
    where: str
    group: str
    wanted: str
    shape: str
    claim: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"[{self.shape}] {self.group}: {self.claim}"

    def as_dict(self) -> Dict[str, Any]:
        return {"where": self.where, "group": self.group,
                "wanted": self.wanted, "shape": self.shape,
                "claim": self.claim, "evidence": dict(self.evidence)}


def _phrases(text: str) -> set:
    """Candidate markers that really occur in this text.

    Adjacent words as they stand, not as they stand after filtering. The
    first version built bigrams from a token list with short words
    dropped, so "позиции считая" was mined from "позиции 2, считая с
    начала" -- a phrase with 100% measured support that could never match
    anything, because the classifier looks for it as a substring and it is
    not one. Measuring with one thing and matching with another is the
    error this project keeps finding; here it is again, in the module
    that builds the thing that gets matched.
    """
    lowered = (text or "").lower()
    words = [(m.group(0), m.start(), m.end())
             for m in _TOKEN.finditer(lowered)]
    out = set()
    for index, (word, start, end) in enumerate(words):
        if len(word) > 2 and word not in _STOP:
            out.add(word)
        if index + 1 < len(words):
            # The span between them included, so what is mined is a
            # substring of the text and not a reconstruction of one.
            out.add(lowered[start:words[index + 1][2]])
    return out


def observe(where: str, samples: Callable[[str], Iterable[str]],
            groups: Sequence[str], truth: Callable[[str], str],
            decide: Callable[[str], str]) -> List[Hypothesis]:
    """Watch a decision fail, and say what kind of failure it is.

    One hypothesis per group that fails, not one per failing item: a
    claim about a class is what can be tested, and a claim about a
    sentence is an anecdote.
    """
    out: List[Hypothesis] = []
    for group in groups:
        wanted = truth(group)
        shapes: Counter = Counter()
        competing: Counter = Counter()
        wrong = total = 0
        for text in samples(group):
            total += 1
            got = decide(text)
            if got == wanted:
                continue
            wrong += 1
            found = diagnose(text, wanted, got)
            shapes[found["shape"]] += 1
            for kind, marker in found.get("competing", ())[:1]:
                competing[(kind, marker)] += 1
        if not wrong or not total:
            continue
        shape, _ = shapes.most_common(1)[0]
        evidence = {"wrong": wrong, "of": total,
                    "shapes": dict(shapes),
                    "competing": {f"{k}:{m}": n for (k, m), n in competing.items()}}
        if shape == AMBIGUOUS:
            (kind, marker), _ = competing.most_common(1)[0]
            claim = (f"признак «{marker}» класса «{kind}» решает там, где "
                     f"задача принадлежит «{wanted}»; при совместном "
                     f"присутствии признаков решать должен более "
                     f"специфичный, а не первый по списку")
        else:
            claim = (f"у класса «{wanted}» нет ни одного признака в "
                     f"словаре: порядком это не чинится, признак должен "
                     f"появиться")
        out.append(Hypothesis(where=where, group=group, wanted=wanted,
                              shape=shape, claim=claim, evidence=evidence))
    return out


def forms(hypothesis: Hypothesis,
          samples: Callable[[str], Iterable[str]],
          groups: Sequence[str]) -> List[Rule]:
    """The concrete changes a claim licenses, mined from discovery text.

    A marker that is in nearly every item of its class and almost never
    outside it. Both halves matter: the first makes it a rule rather than
    a special case, and the second is what stops it fixing one group by
    breaking another.

    Returns nothing when the text has no such marker, which is a real
    answer -- a claim that cannot be turned into a change is a claim that
    has to wait for a wider region than this module is allowed to touch.
    """
    inside_texts: List[str] = []
    outside_texts: List[str] = []
    for group in groups:
        for text in samples(group):
            lowered = (text or "").lower()
            (inside_texts if group == hypothesis.group else outside_texts
             ).append(lowered)
    if not inside_texts or not outside_texts:
        return []

    seen: Counter = Counter()
    for lowered in inside_texts:
        seen.update(_phrases(lowered))

    scored: List[Tuple[float, float, str]] = []
    for phrase, rough in seen.items():
        if rough / len(inside_texts) < MIN_SUPPORT:
            continue
        # Checked with the matcher the classifier will use, on the raw
        # text. A candidate scored one way and applied another is a
        # candidate whose measurement is about something else.
        if len(phrase) > MAX_MARKER or "\n" in phrase:
            # Layout, not content. See MAX_MARKER.
            continue
        support = sum(1 for t in inside_texts if phrase in t) / len(inside_texts)
        if support < MIN_SUPPORT:
            continue
        leak = sum(1 for t in outside_texts if phrase in t) / len(outside_texts)
        if leak > MAX_LEAK:
            continue
        scored.append((support, -leak, phrase))
    # Support first, then how little it leaks, then LENGTH -- longer wins.
    # A short marker can pass on a narrow corpus and mean nothing outside
    # it: "чем" covers every logic task here and every second sentence a
    # person writes. Among markers that score the same, the more specific
    # one is the one that will still be a discriminator somewhere else.
    scored.sort(key=lambda row: (row[0], row[1], len(row[2])), reverse=True)

    return [Rule(where=hypothesis.where, marker=phrase,
                 decides=hypothesis.wanted, before_builtin=True,
                 lift=-negative_leak, support=support)
            for support, negative_leak, phrase in scored[:MAX_CANDIDATES]]
