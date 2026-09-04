"""
mana.cognition.representations — Level 3: discovering that the current
vocabulary is not enough.

The problem with "evolve representations"
-----------------------------------------
It is easy to say and easy to fake. Adding a field called `invariant` to a
schema changes nothing unless something reads it, and a system that
renames its data structures can report an expanding representation space
forever without becoming able to do anything new.

So this module needs two things the phrase does not supply on its own: a
computable definition of *insufficient*, and a route by which a new field
changes behaviour rather than paperwork.

When is a representation insufficient? -- a computable answer
--------------------------------------------------------------
A representation describes tasks. It is insufficient exactly when it maps
situations that behave differently onto the same description:

    two tasks, identical under the current fields, one solved and one not

Those are **collisions**, and they are countable. A representation with no
collisions explains every difference in outcome it has seen; one where
half the pairs collide explains almost nothing. That gives insufficiency a
number rather than an opinion, and it makes "we need a richer vocabulary"
a claim with evidence behind it.

Where a new field comes from
----------------------------
Not from an LLM's imagination. From the collisions themselves: among the
pairs the current fields cannot separate, which candidate field separates
them best? That is ordinary information gain, computed over features that
are cheap, deterministic and already extractable from the task text -- so
proposing a representation costs nothing and can be done over the whole
observation history for free.

The candidate field library is deliberately small and structural. It
contains things like "how deeply nested is the expression" and "how many
distinct numbers appear", not "is this task about physics": semantic
features would need a model to extract, which would put a model's judgement
inside the definition of MANA's own representation space.

What stops this from being renaming
-----------------------------------
A proposed field has to survive the same gate as everything else. It is
adopted only when a program using the richer description measurably beats
the same program without it -- so a field that nothing can act on is
rejected however well it separates the collisions. Separation makes a
field worth *testing*; only the experiment makes it worth keeping.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .genome import Representation
from .self_model import Observation, band_of

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: A field extractor: task text in, a hashable bucket out. Bucketed rather
#: than continuous because a description is only useful if different tasks
#: can share one -- a field returning a unique float per task would give a
#: perfect zero collision rate and explain nothing.
Extractor = Callable[[str], Any]

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _band(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def _length_band(text: str) -> str:
    return _band(len(_WORD_RE.findall(text)), (8, 20, 45), ("tiny", "short", "medium", "long"))


def _number_count(text: str) -> str:
    return _band(len(_NUM_RE.findall(text)), (1, 3, 7), ("none", "few", "several", "many"))


def _magnitude(text: str) -> str:
    numbers = [abs(float(n.replace(",", "."))) for n in _NUM_RE.findall(text)]
    if not numbers:
        return "none"
    return _band(max(numbers), (10, 1000, 100000), ("unit", "small", "medium", "large"))


def _operator_count(text: str) -> str:
    return _band(len(re.findall(r"[+\-*/^]|\*\*", text)), (1, 3), ("none", "few", "many"))


def _nesting_depth(text: str) -> str:
    depth = current = 0
    for ch in text:
        if ch == "(":
            current += 1
            depth = max(depth, current)
        elif ch == ")":
            current = max(0, current - 1)
    return _band(depth, (1, 2), ("flat", "one", "deep"))


def _question_count(text: str) -> str:
    return _band(text.count("?"), (1, 2), ("none", "one", "several"))


def _has_enumeration(text: str) -> bool:
    return bool(re.search(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+", text))


def _distinct_ratio(text: str) -> str:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return "empty"
    return _band(len(set(words)) / len(words), (0.6, 0.85), ("repetitive", "mixed", "varied"))


def _leading_verb(text: str) -> str:
    """The instruction word the task opens with.

    A structural feature, not a semantic one: it is the literal first
    token, lowercased. Which instruction was given often matters more to
    how a task should be approached than what it is about.
    """
    words = _WORD_RE.findall(text.strip())
    return words[0].lower() if words else ""


def _has_code_markers(text: str) -> bool:
    return bool(re.search(r"```|def |функци|python|return", text, re.I))


def _list_answer_expected(text: str) -> bool:
    return bool(re.search(r"через запятую|списком|перечисли|выпиши", text, re.I))


#: Everything a representation may be built from. Structural and cheap on
#: purpose -- see the module docstring on why nothing here needs a model.
FIELD_LIBRARY: Dict[str, Extractor] = {
    "length_band": _length_band,
    "number_count": _number_count,
    "magnitude": _magnitude,
    "operator_count": _operator_count,
    "nesting_depth": _nesting_depth,
    "question_count": _question_count,
    "has_enumeration": _has_enumeration,
    "distinct_ratio": _distinct_ratio,
    "leading_verb": _leading_verb,
    "has_code_markers": _has_code_markers,
    "list_answer_expected": _list_answer_expected,
}

#: Fields the baseline `task_view` already carries, which come from
#: elsewhere in the system rather than from the text.
_EXTERNAL_FIELDS = {"task", "category", "difficulty"}


def describe(task_text: str, fields: Sequence[str], domain: str = "",
             difficulty: float = 0.5) -> Tuple[Tuple[str, Any], ...]:
    """Describe one task under a given set of fields.

    Returns a sorted tuple so two descriptions are equal exactly when the
    tasks are indistinguishable under this representation -- which is the
    comparison the whole module is built on.
    """
    described: List[Tuple[str, Any]] = []
    for name in sorted(fields):
        if name == "task":
            continue                       # the text itself makes everything unique
        if name == "category":
            described.append((name, domain))
        elif name == "difficulty":
            described.append((name, band_of(difficulty)))
        else:
            extractor = FIELD_LIBRARY.get(name)
            if extractor is not None:
                described.append((name, extractor(task_text)))
    return tuple(described)


@dataclass
class Collision:
    """Two tasks the representation cannot tell apart that behaved
    differently."""
    description: Tuple[Tuple[str, Any], ...]
    solved: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.solved) + len(self.failed)

    @property
    def pairs(self) -> int:
        """Contradicting pairs, which is what actually measures the damage:
        one success against nine failures is far less confusing than five
        against five."""
        return len(self.solved) * len(self.failed)


@dataclass
class Insufficiency:
    """How much of the observed variation the representation fails to
    explain."""
    representation: str
    fields: Tuple[str, ...]
    observations: int
    distinct_descriptions: int
    colliding_pairs: int
    total_pairs: int
    collisions: List[Collision] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """Share of contradicting pairs the representation cannot separate.

        0.0 means every difference in outcome is explained by some
        difference in description; 1.0 means the vocabulary explains
        nothing at all.
        """
        return self.colliding_pairs / self.total_pairs if self.total_pairs else 0.0

    def describe(self) -> str:
        if not self.total_pairs:
            return f"{self.representation}: недостаточно наблюдений для оценки"
        return (f"{self.representation}: {self.rate:.0%} противоречащих пар неразличимы "
                f"({self.colliding_pairs}/{self.total_pairs}), "
                f"{self.distinct_descriptions} различных описаний "
                f"на {self.observations} наблюдений")

    def as_dict(self) -> Dict[str, Any]:
        return {"representation": self.representation, "fields": list(self.fields),
                "observations": self.observations,
                "distinct_descriptions": self.distinct_descriptions,
                "colliding_pairs": self.colliding_pairs, "total_pairs": self.total_pairs,
                "rate": round(self.rate, 4)}


def measure_insufficiency(representation: Representation,
                          observations: Sequence[Observation],
                          task_texts: Dict[str, str]) -> Insufficiency:
    """Count what the representation cannot explain.

    Needs the task texts because an `Observation` records the outcome, not
    the question. Observations whose text is unavailable are skipped rather
    than described as empty strings -- an unknown task collides with every
    other unknown task and would manufacture insufficiency out of missing
    data.
    """
    groups: Dict[Tuple[Tuple[str, Any], ...], Collision] = {}
    counted = 0
    for o in observations:
        text = task_texts.get(o.task_id)
        if text is None:
            continue
        counted += 1
        key = describe(text, representation.fields, o.domain, o.difficulty)
        collision = groups.setdefault(key, Collision(description=key))
        (collision.solved if o.correct else collision.failed).append(o.task_id)

    colliding = sum(c.pairs for c in groups.values())
    solved_total = sum(len(c.solved) for c in groups.values())
    failed_total = sum(len(c.failed) for c in groups.values())
    return Insufficiency(
        representation=representation.name, fields=tuple(representation.fields),
        observations=counted, distinct_descriptions=len(groups),
        colliding_pairs=colliding, total_pairs=solved_total * failed_total,
        collisions=sorted((c for c in groups.values() if c.pairs > 0),
                          key=lambda c: -c.pairs))


@dataclass
class FieldProposal:
    """A candidate field, and how much of the confusion it would remove."""
    field_name: str
    separates_pairs: int
    remaining_pairs: int
    reduction: float
    values_seen: List[Any] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"field": self.field_name, "separates_pairs": self.separates_pairs,
                "remaining_pairs": self.remaining_pairs,
                "reduction": round(self.reduction, 4),
                "values_seen": [str(v) for v in self.values_seen[:6]]}


def propose_fields(insufficiency: Insufficiency, observations: Sequence[Observation],
                   task_texts: Dict[str, str], limit: int = 3) -> List[FieldProposal]:
    """Which unused field best separates what the current one cannot?

    Ordinary information gain over the colliding groups, and nothing more
    clever: a field that splits a confused group into a solved half and a
    failed half has explained something, and one that splits it evenly has
    not. Ranked by how many contradicting pairs it removes, because that is
    the quantity insufficiency is measured in.
    """
    by_id = {o.task_id: o for o in observations}
    unused = [name for name in FIELD_LIBRARY
              if name not in insufficiency.fields and name not in _EXTERNAL_FIELDS]
    proposals: List[FieldProposal] = []

    for name in unused:
        extractor = FIELD_LIBRARY[name]
        remaining = 0
        values: List[Any] = []
        for collision in insufficiency.collisions:
            buckets: Dict[Any, List[bool]] = {}
            for task_id in collision.solved + collision.failed:
                text = task_texts.get(task_id)
                observation = by_id.get(task_id)
                if text is None or observation is None:
                    continue
                value = extractor(text)
                buckets.setdefault(value, []).append(observation.correct)
                if value not in values:
                    values.append(value)
            for outcomes in buckets.values():
                solved = sum(1 for x in outcomes if x)
                remaining += solved * (len(outcomes) - solved)
        separated = insufficiency.colliding_pairs - remaining
        if separated <= 0:
            continue
        proposals.append(FieldProposal(
            field_name=name, separates_pairs=separated, remaining_pairs=remaining,
            reduction=separated / insufficiency.colliding_pairs
            if insufficiency.colliding_pairs else 0.0,
            values_seen=values))

    proposals.sort(key=lambda p: -p.separates_pairs)
    return proposals[:limit]


def insufficiency_gap(insufficiency: Insufficiency, threshold: float = 0.3,
                      min_observations: int = 20) -> Optional[Dict[str, Any]]:
    """Is the vocabulary itself worth working on right now?

    Returns None below the threshold rather than always reporting a gap.
    Every representation collides somewhat, and treating a 5% rate as a
    problem would send the system rewriting its vocabulary in response to
    ordinary noise -- the Level-3 equivalent of chasing a lucky streak.
    """
    if insufficiency.observations < min_observations:
        return None
    if insufficiency.rate < threshold:
        return None
    return {
        "kind": "representation",
        "representation": insufficiency.representation,
        "rate": round(insufficiency.rate, 4),
        "description": (f"представление «{insufficiency.representation}» не различает "
                        f"{insufficiency.rate:.0%} противоречащих пар — словарь описания "
                        f"задачи может быть недостаточен"),
        "worst_collision": (insufficiency.collisions[0].description
                            if insufficiency.collisions else ()),
    }


def enriched(representation: Representation, proposal: FieldProposal) -> Representation:
    """The same representation with one field added.

    One field at a time, matching the one-mutation-one-effect discipline
    the rest of the system follows: adding three at once produces a
    representation that works without saying which addition did it.
    """
    return Representation(
        name=representation.name,
        fields=tuple(representation.fields) + (proposal.field_name,),
        description=(f"{representation.description} + {proposal.field_name} "
                     f"(разделяет {proposal.separates_pairs} пар)").strip(),
        derived_from=representation.name)
