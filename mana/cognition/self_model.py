"""
mana.cognition.self_model — what MANA can do, derived only from what was
measured.

The rule that shapes everything here
------------------------------------
No model may set its own score. There is no `set_capability()`, no way to
assert competence, and `record()` accepts only observations that carry a
verdict from `core.oracle` -- a grader that computed the answer rather
than judging it. A self-model an LLM can write to is a self-report, and a
self-report is exactly the evidence this project is not allowed to accept.

Confidence is an interval, not a number
---------------------------------------
Three attempts and three successes is not "100% capable". The obvious
encoding -- score = successes/attempts -- makes a capability measured
three times indistinguishable from one measured three hundred times, and
gap detection would then chase whichever capability happened to start
badly.

So a capability carries a Wilson score interval. It is the standard
instrument for a binomial proportion at small n, it never runs outside
[0,1] the way the normal approximation does, and it needs no scipy inside
a package the immutable core imports. `score` is the observed rate,
`confidence_interval` is what the data actually supports, and
`uncertainty` is the width of that interval -- which is what makes
"measure this again" a proposal with a computable value rather than a
preference.

Conditions, not just averages
-----------------------------
"MANA is 0.7 at arithmetic" is nearly useless. The interesting statements
are conditional: 0.9 below difficulty 0.3 and 0.3 above it, or 0.8 on one
brain and 0.4 on another. Capabilities are therefore sliced by domain and
by difficulty band, and `failure_modes` records *how* the failures looked
-- a capability that fails by ignoring the output format needs a different
fix from one that fails by computing the wrong number.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

#: Difficulty bands. Three, because two hides the interesting middle and
#: five splits the evidence too thin to say anything about any of them.
BANDS = (("easy", 0.0, 0.35), ("medium", 0.35, 0.65), ("hard", 0.65, 1.01))

#: Below this many observations a capability is reported as unmeasured
#: rather than as a low score. The distinction matters: "we do not know"
#: and "it is bad" lead to different actions.
MIN_OBSERVATIONS = 5

#: Standard normal quantile for a 95% interval. Written out rather than
#: imported so this module stays free of scipy.
_Z95 = 1.959963984540054


def band_of(difficulty: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= difficulty < hi:
            return name
    return "hard"


def wilson_interval(successes: int, trials: int, z: float = _Z95) -> Tuple[float, float]:
    """Confidence interval for a proportion, correct at small n.

    The normal approximation gives intervals that run below 0 and above 1
    on the sample sizes this system actually collects, which then makes
    uncertainty comparisons meaningless. Wilson does not.
    """
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = (z / denominator) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass(frozen=True)
class Observation:
    """One graded attempt. The only thing that may enter the model.

    Carries the grader's reason as well as the verdict: "wrong" and
    "ignored the output format" are different failures with different
    fixes, and a model that records only correct/incorrect cannot tell a
    capability gap from a formatting habit.
    """
    task_id: str
    domain: str
    difficulty: float
    correct: bool
    reason: str = ""              # from core.oracle.Grade
    program: str = ""             # program signature, if one was used
    brain: str = ""
    calls: int = 0
    latency: float = 0.0
    at: float = field(default_factory=time.time)

    @property
    def band(self) -> str:
        return band_of(self.difficulty)

    @property
    def gradable(self) -> bool:
        return self.reason != "ungradable"


@dataclass(frozen=True)
class Capability:
    """A conditional statement about competence, with its evidence."""
    capability_id: str
    domain: str
    band: str
    observations: int
    successes: int
    score: float
    confidence_interval: Tuple[float, float]
    uncertainty: float
    failure_modes: Dict[str, int] = field(default_factory=dict)
    by_brain: Dict[str, float] = field(default_factory=dict)
    by_program: Dict[str, float] = field(default_factory=dict)
    mean_calls: float = 0.0
    mean_latency: float = 0.0

    @property
    def measured(self) -> bool:
        """False when there is too little evidence to claim anything.

        Reported separately from a low score because "we do not know" and
        "it is bad" lead to different actions -- one wants measurement,
        the other wants a fix.
        """
        return self.observations >= MIN_OBSERVATIONS

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["confidence_interval"] = list(self.confidence_interval)
        payload["measured"] = self.measured
        return payload

    def describe(self) -> str:
        if not self.measured:
            return (f"{self.capability_id}: не измерено "
                    f"({self.observations}/{MIN_OBSERVATIONS} наблюдений)")
        lo, hi = self.confidence_interval
        return (f"{self.capability_id}: {self.score:.2f} "
                f"[{lo:.2f}–{hi:.2f}] по {self.observations} наблюдениям")


class SelfModel:
    """Everything MANA has been measured doing, and nothing it claims."""

    def __init__(self, observations: Optional[Iterable[Observation]] = None) -> None:
        self._observations: List[Observation] = list(observations or ())

    # ---------- recording ----------

    def record(self, observation: Observation) -> None:
        self._observations.append(observation)

    def record_grades(self, tasks: Sequence[Any], grades: Sequence[Any],
                      program: str = "", brains: Sequence[str] = (),
                      calls: Sequence[int] = ()) -> int:
        """Ingest a batch straight from `core.oracle`.

        Takes graded results rather than raw text on purpose: the only
        route into this model runs through a grader that computed the
        answer, so there is no path by which a model's opinion of itself
        becomes a capability score.
        """
        added = 0
        for i, (task, grade) in enumerate(zip(tasks, grades)):
            self.record(Observation(
                task_id=getattr(task, "task_id", f"t{i}"),
                domain=getattr(task, "domain", "unknown"),
                difficulty=float(getattr(task, "difficulty", 0.5)),
                correct=bool(getattr(grade, "correct", False)),
                reason=str(getattr(grade, "reason", "")),
                program=program,
                brain=brains[i] if i < len(brains) else "",
                calls=calls[i] if i < len(calls) else 0))
            added += 1
        return added

    @property
    def observations(self) -> List[Observation]:
        return list(self._observations)

    # ---------- deriving capabilities ----------

    def _capability(self, capability_id: str, domain: str, band: str,
                    rows: Sequence[Observation]) -> Capability:
        gradable = [o for o in rows if o.gradable]
        successes = sum(1 for o in gradable if o.correct)
        trials = len(gradable)
        score = successes / trials if trials else 0.0
        lo, hi = wilson_interval(successes, trials)
        failures: Dict[str, int] = {}
        for o in rows:
            if not o.correct:
                failures[o.reason or "wrong"] = failures.get(o.reason or "wrong", 0) + 1

        def rate_by(key: str) -> Dict[str, float]:
            groups: Dict[str, List[Observation]] = {}
            for o in gradable:
                value = getattr(o, key)
                if value:
                    groups.setdefault(value, []).append(o)
            # Only report a slice that has enough of its own evidence --
            # a per-brain rate from two attempts is noise wearing a
            # decimal point.
            return {k: sum(1 for o in v if o.correct) / len(v)
                    for k, v in groups.items() if len(v) >= 3}

        return Capability(
            capability_id=capability_id, domain=domain, band=band,
            observations=trials, successes=successes, score=score,
            confidence_interval=(lo, hi), uncertainty=hi - lo,
            failure_modes=failures, by_brain=rate_by("brain"), by_program=rate_by("program"),
            mean_calls=statistics.fmean([o.calls for o in rows]) if rows else 0.0,
            mean_latency=statistics.fmean([o.latency for o in rows]) if rows else 0.0)

    def capabilities(self) -> Dict[str, Capability]:
        """One capability per (domain, difficulty band), plus a whole-domain
        roll-up.

        Both levels are kept because they answer different questions. The
        roll-up says whether a domain is worth attention at all; the bands
        say where inside it the system actually breaks, which is the thing
        a learning goal can be written against.
        """
        by_domain: Dict[str, List[Observation]] = {}
        by_slice: Dict[Tuple[str, str], List[Observation]] = {}
        for o in self._observations:
            by_domain.setdefault(o.domain, []).append(o)
            by_slice.setdefault((o.domain, o.band), []).append(o)

        out: Dict[str, Capability] = {}
        for domain, rows in by_domain.items():
            out[domain] = self._capability(domain, domain, "all", rows)
        for (domain, band), rows in by_slice.items():
            key = f"{domain}/{band}"
            out[key] = self._capability(key, domain, band, rows)
        return out

    def transfer_profile(self) -> Dict[str, Any]:
        """Where competence holds across domains and where it does not.

        Spread across domains is the interesting number: a system that is
        uniformly 0.6 has a general capability, one that is 0.9 and 0.2 has
        two different ones sharing a name. `transfers` names the pairs
        where the gap is small enough that a mechanism found in one is
        worth trying in the other.
        """
        caps = {k: c for k, c in self.capabilities().items() if c.band == "all" and c.measured}
        if len(caps) < 2:
            return {"domains": list(caps), "spread": None, "transfers": [], "does_not_transfer": []}
        scores = {d: c.score for d, c in caps.items()}
        mean = statistics.fmean(scores.values())
        transfers, blocked = [], []
        for a in sorted(scores):
            for b in sorted(scores):
                if a >= b:
                    continue
                gap = abs(scores[a] - scores[b])
                (transfers if gap <= 0.15 else blocked).append(
                    {"from": a, "to": b, "gap": round(gap, 3)})
        return {"domains": sorted(scores), "mean": round(mean, 3),
                "spread": round(max(scores.values()) - min(scores.values()), 3),
                "transfers": transfers, "does_not_transfer": blocked}

    def summary(self) -> Dict[str, Any]:
        caps = self.capabilities()
        measured = {k: c for k, c in caps.items() if c.measured}
        return {
            "observations": len(self._observations),
            "capabilities": {k: c.as_dict() for k, c in sorted(caps.items())},
            "measured_count": len(measured),
            "unmeasured": sorted(k for k, c in caps.items() if not c.measured),
            "weakest": min((c.describe() for c in measured.values()),
                           key=lambda s: s, default=""),
            "transfer": self.transfer_profile(),
        }

    # ---------- persistence ----------

    def to_dict(self) -> Dict[str, Any]:
        return {"version": __version__,
                "observations": [asdict(o) for o in self._observations]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfModel":
        return cls(Observation(**row) for row in (data.get("observations") or []))

    def save(self, path: Path) -> None:
        """Observations, not conclusions.

        Capabilities are always recomputed from the raw record, so a
        change to how they are derived applies retroactively to everything
        ever measured -- and a stored conclusion can never drift away from
        the evidence that produced it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SelfModel":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            # A corrupt self-model must not stop MANA from running. An
            # empty one is honest: it says nothing has been measured,
            # which is exactly the situation.
            return cls()
