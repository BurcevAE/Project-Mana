"""
mana.episode_affinity — how strongly does a turn belong to an episode?

NOT WIRED INTO THE AGENT. This module is scored offline by
scripts/benchmark_episode_affinity.py first. Integrating it into
_build_context before the numbers are known is how you lose half a
conversation and then go looking for why.

Design decision, and the reason this is `affinity` rather than
`boundary_detector`: five previous heuristics in this codebase each
looked sound and each was refuted by the first real dialogue that hit it
(relevance threshold, entity extraction, acronym extraction, ambiguity
length, topic candidates). A hard boundary makes such an error
catastrophic -- a turn is either in or out, and if it is wrongly out, the
context is gone. A graded score makes the same error survivable: the turn
ranks lower, other evidence can outweigh it, and a close call can be
escalated to a clarifying question instead of guessed.

So this module answers "how related?" and deliberately does NOT answer
"which episode is this?". That decision belongs higher up, where the cost
of being wrong is visible.

Measured constraint that shaped the implementation: word-level overlap
does not work in Russian. On real turns, "что там с новостями" scored
0.000 against "Какие есть последние новости про ИИ?" -- the correct
episode -- while scoring 0.111 against an unrelated turn about dates,
because "новости" and "новостями" are different tokens. Any scorer here
must survive inflection; the three below exist to be compared, not
assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: Russian inflectional endings, longest first so that "ами" is stripped
#: before "ми". Deliberately crude: this is suffix stripping, not
#: morphology. It exists to make "новости"/"новостями" collide, and it
#: will over-strip some words -- which the benchmark measures rather than
#: hides.
_SUFFIXES = (
    "иями", "ями", "ами", "иях", "ях", "ах", "ов", "ев", "ий", "ый", "ой",
    "ая", "яя", "ое", "ее", "ые", "ие", "ыми", "ими", "ого", "его", "ому",
    "ему", "ем", "ом", "ах", "ям", "ам", "ей", "ой", "ую", "юю", "их", "ых",
    "и", "ы", "а", "я", "о", "е", "у", "ю", "ь", "й",
)

_STOP = {
    "и", "в", "на", "с", "по", "к", "о", "об", "от", "до", "за", "из", "у",
    "не", "что", "это", "как", "а", "но", "же", "ли", "бы", "то", "так",
    "был", "была", "было", "были", "есть", "для", "про", "или", "если",
    "мне", "меня", "тебя", "ты", "я", "он", "она", "они", "мы", "вы",
}


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP]


def stem(word: str) -> str:
    """Strip one inflectional ending. Crude by design -- see _SUFFIXES."""
    for suffix in _SUFFIXES:
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _char_ngrams(text: str, n: int = 4) -> set:
    out = set()
    for token in _tokens(text):
        padded = token
        if len(padded) <= n:
            out.add(padded)
            continue
        for i in range(len(padded) - n + 1):
            out.add(padded[i:i + n])
    return out


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- the three scorers under comparison ---------------------------------

def score_words(query: str, text: str) -> float:
    """Plain word overlap -- the baseline that failed on inflection."""
    return _jaccard(set(_tokens(query)), set(_tokens(text)))


def score_stems(query: str, text: str) -> float:
    """Word overlap after suffix stripping."""
    return _jaccard({stem(w) for w in _tokens(query)},
                    {stem(w) for w in _tokens(text)})


def score_ngrams(query: str, text: str) -> float:
    """Character n-gram overlap -- no morphology model needed."""
    return _jaccard(_char_ngrams(query), _char_ngrams(text))


SCORERS = {"words": score_words, "stems": score_stems, "ngrams": score_ngrams}


@dataclass
class Episode:
    """An episode is a SET OF TURNS WITH SCORES, not a partition.

    A turn may relate to several episodes at once, which is what removes
    the need for a boundary oracle: nothing has to decide where one
    episode ends. Returning to an earlier subject is then just that
    episode scoring highest again, not a segmentation event.
    """
    episode_id: str
    turns: List[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.turns.append(text)


#: Default follows the benchmark, not intuition: `stems` scored R@1 1.00
#: with zero contamination and zero loss on real dialogues, while
#: `ngrams` ranked the AI-news episode above football for a football
#: query (R@1 0.80). Re-run scripts/benchmark_episode_affinity.py before
#: changing this.
DEFAULT_SCORER = "stems"


def episode_affinity(query: str, episode: Episode, scorer: str = DEFAULT_SCORER,
                      top_k: int = 3) -> float:
    """Affinity of a turn to an episode.

    Scored against the episode's BEST-MATCHING turns rather than its mean:
    an episode accumulates turns, and averaging lets a long episode dilute
    a strong match into nothing. Taking the top few keeps a single clearly
    relevant turn decisive.
    """
    fn = SCORERS[scorer]
    scores = sorted((fn(query, turn) for turn in episode.turns), reverse=True)
    if not scores:
        return 0.0
    best = scores[:top_k]
    return sum(best) / len(best)


def rank_episodes(query: str, episodes: Sequence[Episode],
                   scorer: str = DEFAULT_SCORER) -> List[tuple]:
    """Episodes ordered by affinity, strongest first. No decision is made
    here -- the caller decides whether the top score is clear enough to
    act on, or close enough to a rival to be worth asking about."""
    ranked = [(ep.episode_id, episode_affinity(query, ep, scorer)) for ep in episodes]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
