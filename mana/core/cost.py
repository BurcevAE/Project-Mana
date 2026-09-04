"""
mana.core.cost — what a computation cost, in units that mean something.

Everything in MANA counted "calls" until now, and a call is not a unit of
cost: one call to a 120-billion-parameter remote model and one call to a
7B model running locally were the same number. That made the whole point
of the next generation -- "cheap computation first, large model only when
necessary" -- unmeasurable, and any claim of saving anything was
rhetoric.

Why this lives in the immutable core
------------------------------------
It is not an acceptance rule and no gate reads it. It is here for a
narrower reason: if the evolvable layer could redefine CostVector, it
could redefine what "cheaper" means, and the entire argument for a
specialised brain is an argument about cost. A system that may not
change the meaning of its own evaluation may not change the meaning of
its own units either.

What it deliberately does NOT do
--------------------------------
It does not collapse into one number. Reducing time, tokens and memory
to a single scalar needs weights, and nobody has measured those weights.
An invented weighting would look like a measurement and be a preference.
`efficiency()` therefore returns a ratio per unit and leaves the
comparison to a reader who can see all of them.

Unmeasured is not zero
----------------------
A remote API that does not report token use has an unknown token cost,
not a zero one, and treating it as zero makes the most expensive
substrate look free. Every aggregate carries how many contributions
could not be measured, so a reader can tell "1200 tokens over 8 calls"
from "1200 tokens over 8 calls, 5 of which reported nothing".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

#: What kind of machinery produced the work. Not a closed set: a new
#: substrate adds a name here rather than a branch anywhere else.
UNKNOWN = "unknown"
ALGORITHMIC = "algorithmic"
CLASSICAL_ML = "classical_ml"
SMALL_NEURAL = "small_neural"
ADAPTER = "adapter"
LOCAL_LLM = "local_llm"
REMOTE_LLM = "remote_llm"


@dataclass(frozen=True)
class CostVector:
    """What one piece of work cost. Fields that cannot be measured for a
    given substrate stay unmeasured rather than defaulting to zero."""

    calls: int = 0
    wall_seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    #: Calls whose token use was not reported. Kept because "no tokens
    #: recorded" and "zero tokens used" are different facts and only one
    #: of them is ever true of a language model.
    unmeasured_token_calls: int = 0
    #: Peak resident memory attributable to this work, in bytes. Only
    #: meaningful for computation that happened in THIS process: a remote
    #: model's memory is on someone else's machine, and a local server's
    #: is in another process. None means "not attributable here", which
    #: is the honest answer for every network call.
    rss_bytes: Optional[int] = None
    #: Calls per substrate class, so an aggregate can say what mix of
    #: machinery produced it.
    by_substrate: Mapping[str, int] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def tokens_complete(self) -> bool:
        """True when every call in this aggregate reported its token use."""
        return self.calls > 0 and self.unmeasured_token_calls == 0

    def add(self, other: "CostVector") -> "CostVector":
        """Combine two costs.

        `rss_bytes` takes the maximum rather than the sum: peak memory of
        two things that ran one after the other is the larger peak, not
        their total, and summing it would report a machine that never
        existed.
        """
        merged: Dict[str, int] = dict(self.by_substrate)
        for name, count in other.by_substrate.items():
            merged[name] = merged.get(name, 0) + count
        rss = [v for v in (self.rss_bytes, other.rss_bytes) if v is not None]
        return CostVector(
            calls=self.calls + other.calls,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            unmeasured_token_calls=(self.unmeasured_token_calls +
                                    other.unmeasured_token_calls),
            rss_bytes=max(rss) if rss else None,
            by_substrate=merged)

    def __add__(self, other: "CostVector") -> "CostVector":
        return self.add(other)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "calls": self.calls,
            "wall_seconds": round(self.wall_seconds, 4),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "unmeasured_token_calls": self.unmeasured_token_calls,
            "by_substrate": dict(self.by_substrate),
        }
        if self.rss_bytes is not None:
            payload["rss_bytes"] = self.rss_bytes
        return payload

    def describe(self) -> str:
        parts = [f"{self.calls} вызовов", f"{self.wall_seconds:.1f} с"]
        if self.tokens:
            note = "" if self.tokens_complete else f" (+{self.unmeasured_token_calls} без учёта)"
            parts.append(f"{self.tokens} токенов{note}")
        elif self.unmeasured_token_calls:
            parts.append(f"токены не измерены ({self.unmeasured_token_calls} вызовов)")
        if self.rss_bytes is not None:
            parts.append(f"{self.rss_bytes / 1e6:.0f} МБ")
        if self.by_substrate:
            mix = ", ".join(f"{k} x {v}" for k, v in sorted(self.by_substrate.items()))
            parts.append(mix)
        return " · ".join(parts)


def one_call(substrate: str, wall_seconds: float, tokens_in: Optional[int] = None,
             tokens_out: Optional[int] = None,
             rss_bytes: Optional[int] = None) -> CostVector:
    """The cost of a single call, with unreported tokens recorded as
    unreported rather than as zero."""
    measured = tokens_in is not None or tokens_out is not None
    return CostVector(
        calls=1, wall_seconds=float(wall_seconds),
        tokens_in=int(tokens_in or 0), tokens_out=int(tokens_out or 0),
        unmeasured_token_calls=0 if measured else 1,
        rss_bytes=rss_bytes, by_substrate={substrate or UNKNOWN: 1})


def efficiency(gain: float, cost: CostVector) -> Dict[str, Optional[float]]:
    """Capability gained per unit spent -- one ratio per unit, on purpose.

    There is no `efficiency_score` here and there should not be. Turning
    seconds, tokens and megabytes into one number requires weights that
    nobody has measured, and a weighting invented to make the arithmetic
    work would read as a measurement while being a preference. A reader
    who can see all four ratios can compare two brains honestly; a reader
    given one number cannot tell which unit it was bought with.

    A ratio is None when its unit was not measured -- never 0.0, which
    would read as "infinitely inefficient" for exactly the substrates
    whose cost we failed to record.
    """
    per_call = gain / cost.calls if cost.calls else None
    per_second = gain / cost.wall_seconds if cost.wall_seconds > 0 else None
    per_ktoken = (gain / (cost.tokens / 1000.0)
                  if cost.tokens_complete and cost.tokens else None)
    per_mb = (gain / (cost.rss_bytes / 1e6)
              if cost.rss_bytes else None)
    return {"per_call": per_call, "per_second": per_second,
            "per_1k_tokens": per_ktoken, "per_mb": per_mb}


def describe_efficiency(gain: float, cost: CostVector) -> str:
    ratios = efficiency(gain, cost)
    shown = [f"{name}={value:.4f}" for name, value in ratios.items()
             if value is not None]
    missing = [name for name, value in ratios.items() if value is None]
    text = "прирост " + (", ".join(shown) or "не измерен")
    if missing:
        text += f" · не измерено: {', '.join(missing)}"
    return text
