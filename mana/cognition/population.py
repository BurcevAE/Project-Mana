"""
mana.cognition.population — a cognitive ecology, not a champion.

What single-champion search loses
---------------------------------
`evolve_pipeline` keeps one best genome and discards everything else. That
is correct for tuning and wrong for open-endedness, and the failure is
specific: **a specialist is thrown away for being average.** A program
that is superb on hard arithmetic and mediocre elsewhere loses to one that
is uniformly middling, so the strategy that would have been the seed of a
real capability never survives its first comparison.

The structure here is a grid of niches, one cell per (domain, difficulty
band), each holding its own best performer -- the MAP-Elites idea, which
exists precisely because "best overall" and "best at something" are
different questions and only the second compounds.

Three populations in one, and they are not interchangeable
----------------------------------------------------------
  * **Elites** -- the best in each niche. What the compiler should
    actually reach for.
  * **Novel** -- behaviourally unlike anything else, whatever they score.
    Kept because the productive direction usually looks bad at first: a
    mechanism that will eventually win often starts by failing
    differently rather than by failing less.
  * **Retired** -- evicted, but remembered by signature, so the search
    does not re-derive something it already discarded.

Coverage is the measurable part of "expanded space"
---------------------------------------------------
"MANA can do more than before" needs a number or it is a slogan.
`coverage()` reports how many niches have any occupant and how many have a
competent one. That is the honest proxy: a system filling twelve niches at
0.8 can do more than one filling three at 0.9, and a single global score
cannot express the difference.

Bounded, deliberately
---------------------
An archive that grows without limit turns novelty pressure into a memory
leak and makes every distance computation slower than the last. Eviction
removes the least novel non-elite, so what is lost is the candidate most
similar to something already kept.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .novelty import Behaviour, NoveltyArchive, distance
from .self_model import BANDS, band_of

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: A candidate must reach this in a niche before the niche counts as
#: covered *competently*. Occupancy and competence are reported
#: separately: filling a cell with something that fails most of the time
#: is not the same as covering it.
COMPETENT = 0.6

#: Room for novel-but-not-elite candidates. Small on purpose -- this is a
#: search budget, not a museum, and every extra member makes each novelty
#: assessment slower.
DEFAULT_NOVELTY_SLOTS = 24


@dataclass
class Candidate:
    """One program, what it scored where, and where it came from."""
    candidate_id: str
    steps: Tuple[str, ...]
    behaviour: Optional[Behaviour] = None
    scores: Dict[str, float] = field(default_factory=dict)     # niche -> accuracy
    trials: Dict[str, int] = field(default_factory=dict)       # niche -> observations
    cost: float = 0.0
    novelty: float = 0.0
    parent_id: Optional[str] = None
    origin: str = ""                                            # what produced it
    born: float = field(default_factory=time.time)

    def score_in(self, niche: str) -> float:
        return self.scores.get(niche, 0.0)

    def best_niche(self) -> Optional[str]:
        return max(self.scores, key=self.scores.get) if self.scores else None

    def breadth(self) -> int:
        """How many niches this candidate has been measured competent in.

        Reported alongside its best score because a specialist and a
        generalist with the same peak are different animals, and the
        distinction is invisible in a single number.
        """
        return sum(1 for v in self.scores.values() if v >= COMPETENT)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = list(self.steps)
        payload["behaviour"] = self.behaviour.as_dict() if self.behaviour else None
        payload["breadth"] = self.breadth()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        behaviour = data.get("behaviour")
        return cls(candidate_id=data["candidate_id"], steps=tuple(data.get("steps") or ()),
                   behaviour=Behaviour.from_dict(behaviour) if behaviour else None,
                   scores={k: float(v) for k, v in (data.get("scores") or {}).items()},
                   trials={k: int(v) for k, v in (data.get("trials") or {}).items()},
                   cost=float(data.get("cost") or 0.0),
                   novelty=float(data.get("novelty") or 0.0),
                   parent_id=data.get("parent_id"), origin=data.get("origin", ""),
                   born=float(data.get("born") or time.time()))


def niche_of(domain: str, difficulty: float) -> str:
    return f"{domain}/{band_of(difficulty)}"


@dataclass
class Admission:
    """Why a candidate was kept or dropped. Returned rather than logged so
    a caller can act on the reason -- "displaced an elite" and "kept for
    novelty" mean different things for what to try next."""
    candidate_id: str
    admitted: bool
    as_elite: List[str] = field(default_factory=list)
    as_novel: bool = False
    displaced: List[str] = field(default_factory=list)
    novelty: float = 0.0
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Population:
    """The ecology: elites per niche, a novelty pool, and a memory of what
    has been retired."""

    def __init__(self, novelty_slots: int = DEFAULT_NOVELTY_SLOTS,
                 novelty_threshold: float = 0.25) -> None:
        self.novelty_slots = novelty_slots
        self.novelty_threshold = novelty_threshold
        self._elites: Dict[str, Candidate] = {}
        self._novel: List[Candidate] = []
        self._retired: Dict[str, str] = {}      # candidate_id -> reason

    # ---------- membership ----------

    def elites(self) -> Dict[str, Candidate]:
        return dict(self._elites)

    def novel(self) -> List[Candidate]:
        return list(self._novel)

    def members(self) -> List[Candidate]:
        seen: Dict[str, Candidate] = {}
        for c in list(self._elites.values()) + self._novel:
            seen[c.candidate_id] = c
        return list(seen.values())

    def __len__(self) -> int:
        return len(self.members())

    def _archive(self) -> NoveltyArchive:
        return NoveltyArchive([c.behaviour for c in self.members() if c.behaviour])

    # ---------- admission ----------

    def admit(self, candidate: Candidate) -> Admission:
        """Try to place a candidate. Elite anywhere, or novel, or neither.

        Both routes are checked, and elite status is checked first: a
        candidate that wins a niche belongs in the grid regardless of how
        familiar it looks, because being the best at something is a
        stronger reason to keep it than being unusual.
        """
        result = Admission(candidate_id=candidate.candidate_id, admitted=False)

        if candidate.behaviour is not None:
            verdict = self._archive().assess(candidate.behaviour, self.novelty_threshold)
            candidate.novelty = verdict.novelty
            result.novelty = verdict.novelty

        displaced: Dict[str, Candidate] = {}
        for niche, score in candidate.scores.items():
            incumbent = self._elites.get(niche)
            if incumbent is None:
                self._elites[niche] = candidate
                result.as_elite.append(niche)
            elif score > incumbent.score_in(niche):
                self._elites[niche] = candidate
                result.as_elite.append(niche)
                # The object, not just the id. Looking it up afterwards
                # fails: by then it is no longer an elite and not yet in
                # the novelty pool, so it is findable nowhere and vanishes
                # without being kept OR retired -- worse than either.
                displaced[incumbent.candidate_id] = incumbent
                result.displaced.append(incumbent.candidate_id)

        if result.as_elite:
            result.admitted = True
            result.reason = f"лучший в нишах: {', '.join(sorted(result.as_elite))}"
            self._prune_displaced(displaced)
            return result

        # Not elite anywhere. Novelty is the second route in, because the
        # productive direction usually looks bad before it looks good.
        if candidate.behaviour is not None and candidate.novelty >= self.novelty_threshold:
            self._novel.append(candidate)
            result.admitted = True
            result.as_novel = True
            result.reason = f"поведенчески новый (новизна {candidate.novelty:.2f})"
            self._evict_if_needed()
            return result

        self._retired[candidate.candidate_id] = "ни в одной нише не лучший и не новый"
        result.reason = self._retired[candidate.candidate_id]
        return result

    def _prune_displaced(self, displaced: Dict[str, "Candidate"]) -> None:
        """A displaced elite is not deleted -- it may still be novel.

        Deleting it immediately is the single-champion mistake in
        miniature: the candidate that just lost a niche may be the only
        member exploring a direction, and its score losing by 0.02 says
        nothing about that.
        """
        still_elite = {c.candidate_id for c in self._elites.values()}
        in_pool = {c.candidate_id for c in self._novel}
        for candidate_id, demoted in displaced.items():
            if candidate_id in still_elite or candidate_id in in_pool:
                continue          # holds another niche, or is already kept
            if demoted.novelty >= self.novelty_threshold:
                self._novel.append(demoted)
            else:
                self._retired[candidate_id] = "вытеснен и недостаточно нов"
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop the least novel non-elite when the pool overflows.

        Least novel rather than lowest scoring: the novelty pool is not
        ranked by performance, and evicting by score would slowly turn it
        into a second, worse elite grid.
        """
        while len(self._novel) > self.novelty_slots:
            victim = min(self._novel, key=lambda c: c.novelty)
            self._novel.remove(victim)
            self._retired[victim.candidate_id] = "вытеснен из пула новизны как наименее новый"

    def was_retired(self, candidate_id: str) -> Optional[str]:
        """So the search does not re-derive what it already discarded."""
        return self._retired.get(candidate_id)

    # ---------- what the population is for ----------

    def best_for(self, domain: str, difficulty: float) -> Optional[Candidate]:
        """The elite of this niche, or the nearest band that has one.

        Falling back to a neighbouring band rather than to a global best:
        a program that wins on easy arithmetic is a better guess for
        medium arithmetic than one that wins on hard logic, and "best
        overall" would return the latter.
        """
        niche = niche_of(domain, difficulty)
        if niche in self._elites:
            return self._elites[niche]
        for name, _lo, _hi in BANDS:
            neighbour = f"{domain}/{name}"
            if neighbour in self._elites:
                return self._elites[neighbour]
        return None

    def coverage(self, domains: Sequence[str] = ()) -> Dict[str, Any]:
        """How much of the space is occupied, and how much competently.

        The measurable proxy for "expanded cognitive space". Occupancy and
        competence are separate numbers because filling a cell with
        something that usually fails is not covering it.
        """
        bands = [b[0] for b in BANDS]
        all_domains = list(domains) or sorted({n.split("/")[0] for n in self._elites})
        total = len(all_domains) * len(bands)
        occupied = [n for n in self._elites if n.split("/")[0] in all_domains]
        competent = [n for n in occupied if self._elites[n].score_in(n) >= COMPETENT]
        return {
            "niches_total": total,
            "occupied": len(occupied),
            "competent": len(competent),
            "occupancy": len(occupied) / total if total else 0.0,
            "competence": len(competent) / total if total else 0.0,
            "empty": sorted(f"{d}/{b}" for d in all_domains for b in bands
                            if f"{d}/{b}" not in self._elites),
        }

    def diversity(self) -> Dict[str, Any]:
        """How different the members actually are from each other.

        A population of near-identical members has the shape of an ecology
        and none of the value, and coverage alone cannot see it: one
        program can occupy every niche.
        """
        members = [c for c in self.members() if c.behaviour]
        if len(members) < 2:
            return {"members": len(members), "mean_distance": None, "distinct": len(members)}
        distances = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                distances.append(distance(a.behaviour, b.behaviour)["distance"])
        mean = sum(distances) / len(distances)
        distinct = len({c.behaviour.outcomes for c in members})
        return {"members": len(members), "mean_distance": round(mean, 4),
                "distinct_behaviours": distinct,
                "min_distance": round(min(distances), 4)}

    def report(self, domains: Sequence[str] = ()) -> Dict[str, Any]:
        return {
            "size": len(self),
            "elites": {n: c.candidate_id for n, c in sorted(self._elites.items())},
            "novel_pool": len(self._novel),
            "retired": len(self._retired),
            "coverage": self.coverage(domains),
            "diversity": self.diversity(),
            "specialists": [
                {"candidate": c.candidate_id, "niche": c.best_niche(),
                 "score": round(c.score_in(c.best_niche() or ""), 3), "breadth": c.breadth()}
                for c in sorted(self._elites.values(), key=lambda c: -c.breadth())
                if c.breadth() <= 1][:5],
        }

    # ---------- persistence ----------

    def to_dict(self) -> Dict[str, Any]:
        return {"novelty_slots": self.novelty_slots,
                "novelty_threshold": self.novelty_threshold,
                "elites": {n: c.candidate_id for n, c in self._elites.items()},
                "members": [c.as_dict() for c in self.members()],
                "retired": dict(self._retired)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Population":
        population = cls(int(data.get("novelty_slots") or DEFAULT_NOVELTY_SLOTS),
                         float(data.get("novelty_threshold") or 0.25))
        by_id = {row["candidate_id"]: Candidate.from_dict(row)
                 for row in (data.get("members") or [])}
        for niche, candidate_id in (data.get("elites") or {}).items():
            if candidate_id in by_id:
                population._elites[niche] = by_id[candidate_id]
        elite_ids = {c.candidate_id for c in population._elites.values()}
        population._novel = [c for cid, c in by_id.items() if cid not in elite_ids]
        population._retired = dict(data.get("retired") or {})
        return population

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Population":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return cls()
