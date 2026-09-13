"""
mana.core.sequential — how much experience is enough, decided by the evidence.

`gates.MIN_PAIRED_TRIALS` answers "when may anything be said at all". It
does not answer "when has enough been seen", and the chess duels of 2.90
used it as both: every experiment stopped at thirty decided games and read
a forty-point-wide interval as a refutation. At that size an improvement
to 60% was accepted 4-12% of the time (exact enumeration, 2026-09-12) --
the rule could reliably see only effects of 70% and more, and eight
experiments in a row came out "исход не изменился".

This module replaces "stop at N" with a stopping rule fixed before the
first observation: Wald's sequential probability ratio test on paired
binary outcomes. That is the shape both consumers actually have -- the
chess duel (a decided game: the changed player won or the unchanged one
did; draws say nothing about which is better) and the acceptance gate (a
discordant pair: only one of the two answered right).

    H0: the change wins NO_EFFECT = half of the decided outcomes.
    H1: it wins NO_EFFECT + min_effect of them -- the smallest
        improvement worth having.

Each outcome moves the log likelihood ratio by ln(p1/p0) for a win and
ln((1-p1)/(1-p0)) for a loss. At ln((1-beta)/alpha) the test accepts, at
ln(beta/(1-alpha)) it rejects, between the two it asks for more. A strong
effect and a harmful one are decided quickly, one near the boundary
slowly -- "dynamic" in the only sense that keeps the error rates.

Fixed here, not by the caller:

  * the error rates, per class of consequence. A gate whose threshold the
    claimant chooses is not a gate; the most a caller can do by
    misnaming the consequence is pick the laxer of two rates written here;
  * the verdict is the FIRST crossing and never moves after it. Looking
    again after a decision, or stopping when the numbers happen to look
    good, is how a fixed-sample test is fooled -- this one stays valid
    under continuous monitoring precisely because the stop is the first
    crossing. Outcomes arriving later are counted as late, never weighed;
  * a ceiling ends in NOT_EVALUATED, never REJECTED. Running out of budget
    is a fact about the budget, not about the claim, and writing it down
    as a refutation closes a question nobody answered.

Chosen by the caller: `min_effect` -- how big an improvement is worth
having -- and the ceiling, which is what observations cost. A larger
`min_effect` does not make a false acceptance easier: under no effect the
chance of accepting stays below Wald's bound alpha/(1-beta) whatever the
target, and in practice under alpha, because the ratio overshoots the
boundary. What a larger target does is reject small real improvements --
a statement about what is worth having, made before the data.

"REJECTED" here reads narrowly: the evidence favours "no effect" over "an
effect of min_effect". A smaller effect is not excluded by it.

Measured (scripts/run_sequential.py, 4000 simulated duels per point,
draw share 26% from the real record, min_effect 0.10, reversible,
ceiling 400 decided; "fixed" is the 2.90 duel -- 30 decided, q=2):

    true share   fixed: adopted / games    sequential: adopted / games
       40%            0.0%  /  41               0.0%  /  37
       50%            1.1%  /  41               4.7%  /  97
       55%            3.1%  /  41              35.2%  / 153
       60%            8.9%  /  41              81.9%  / 136
       65%           23.0%  /  41              96.6%  /  91
       70%           43.1%  /  41              99.8%  /  63

The price is stated, not hidden: under no effect the sequential rule
accepts 4.7% against 1.1% (within its alpha of 5%), and an experiment on
a lever that does nothing costs about 97 games instead of 41. A harmful
lever is found out sooner than before. Wald's forecast matched every
simulated mean to within 10%.

Nothing uses this yet. Step 2 wires it into the chess duel behind a
flag; the gates are unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from .gates import ACCEPTED, NOT_EVALUATED, REJECTED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The null: the change wins half of the decided outcomes.
NO_EFFECT = 0.5

#: Neither boundary crossed yet. A state, not a verdict: nothing may be
#: concluded from it and nothing is written down as one.
CONTINUE = "CONTINUE"

#: Classes of consequence. REVERSIBLE: the change is provisional and is
#: re-tested on fresh evidence before it becomes permanent -- a false
#: acceptance has to survive twice. IRREVERSIBLE: nothing checks it again
#: (a patch to MANA's own source), so one test carries all the risk.
REVERSIBLE = "reversible"
IRREVERSIBLE = "irreversible"

#: (alpha, beta) per class: the chance of accepting a change with no
#: effect, and of rejecting one with exactly `min_effect`.
ERROR_RATES = {REVERSIBLE: (0.05, 0.20), IRREVERSIBLE: (0.01, 0.10)}


@dataclass(frozen=True)
class Plan:
    """The stopping rule, fixed before the first outcome is seen."""
    min_effect: float
    consequence: str = REVERSIBLE
    #: Most outcomes this test may weigh; 0 means no ceiling.
    ceiling: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.min_effect) < NO_EFFECT:
            raise ValueError(f"min_effect must lie in (0, {NO_EFFECT}), "
                             f"got {self.min_effect}")
        if self.consequence not in ERROR_RATES:
            raise ValueError(f"unknown consequence {self.consequence!r}; "
                             f"one of {sorted(ERROR_RATES)}")
        if int(self.ceiling) < 0:
            raise ValueError("ceiling must be >= 0")

    @property
    def alpha(self) -> float:
        return ERROR_RATES[self.consequence][0]

    @property
    def beta(self) -> float:
        return ERROR_RATES[self.consequence][1]

    @property
    def target(self) -> float:
        return NO_EFFECT + float(self.min_effect)

    @property
    def win_step(self) -> float:
        return math.log(self.target / NO_EFFECT)

    @property
    def loss_step(self) -> float:
        return math.log((1.0 - self.target) / (1.0 - NO_EFFECT))

    @property
    def accept_at(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def reject_at(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    def llr(self, won: int, lost: int) -> float:
        return int(won) * self.win_step + int(lost) * self.loss_step

    def as_dict(self) -> Dict[str, Any]:
        return {"min_effect": float(self.min_effect),
                "consequence": self.consequence, "ceiling": int(self.ceiling)}


@dataclass
class SequentialTest:
    """One running test: counts only, the verdict derived from them.

    The status is not stored. Counting stops at the first crossing, so the
    final counts ARE the crossing point and the verdict follows from them
    -- which means a saved record cannot claim a verdict its own numbers
    do not support.
    """
    plan: Plan
    won: int = 0
    lost: int = 0
    #: Outcomes that arrived after the verdict: counted, never weighed.
    late: int = 0

    @property
    def trials(self) -> int:
        return int(self.won) + int(self.lost)

    @property
    def llr(self) -> float:
        return self.plan.llr(self.won, self.lost)

    @property
    def status(self) -> str:
        llr = self.llr
        if llr >= self.plan.accept_at:
            return ACCEPTED
        if llr <= self.plan.reject_at:
            return REJECTED
        if self.plan.ceiling and self.trials >= self.plan.ceiling:
            return NOT_EVALUATED
        return CONTINUE

    @property
    def decided(self) -> bool:
        return self.status != CONTINUE

    def observe(self, won: bool) -> str:
        """Weigh one decided outcome, unless the verdict is already in."""
        if self.decided:
            self.late += 1
            return self.status
        if won:
            self.won += 1
        else:
            self.lost += 1
        return self.status

    def observe_all(self, outcomes: Iterable[bool]) -> str:
        """In the order they happened: the verdict is the first crossing."""
        for won in outcomes:
            self.observe(bool(won))
        return self.status

    def forecast(self) -> Dict[str, float]:
        """Where this is heading, from the share seen so far.

        A Laplace estimate rather than the raw share, so three wins out of
        three do not forecast certainty. An approximation -- it ignores the
        overshoot and the ceiling -- and says so by being a forecast.
        """
        share = (self.won + 1.0) / (self.trials + 2.0)
        return forecast(self.plan, share, start=self.llr)

    def note(self) -> str:
        plan = self.plan
        seen = f"{self.won} из {self.trials} решённых за изменение"
        if self.status == ACCEPTED:
            return (f"изменение выигрывает чаще: {seen}; граница принятия "
                    f"{plan.accept_at:.2f} пройдена ({self.llr:.2f}) — эффект "
                    f"не меньше {plan.min_effect:.0%} при ошибке "
                    f"{plan.alpha:.0%}")
        if self.status == REJECTED:
            return (f"эффекта в {plan.min_effect:.0%} нет: {seen}; граница "
                    f"отказа {plan.reject_at:.2f} пройдена ({self.llr:.2f}). "
                    f"Меньший эффект этим опытом не исключён")
        if self.status == NOT_EVALUATED:
            return (f"бюджет {plan.ceiling} исчерпан, ни одна граница не "
                    f"пройдена ({self.llr:.2f} между {plan.reject_at:.2f} и "
                    f"{plan.accept_at:.2f}): {seen}. Не хватило опыта — "
                    f"это не отказ")
        ahead = self.forecast()
        return (f"{seen}; до решения примерно ещё {ahead['trials']:.0f}, "
                f"шанс принятия сейчас {ahead['accept']:.0%}")

    def as_dict(self) -> Dict[str, Any]:
        return {"plan": self.plan.as_dict(), "won": int(self.won),
                "lost": int(self.lost), "late": int(self.late),
                "status": self.status, "llr": round(self.llr, 6)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SequentialTest":
        test = cls(plan=Plan(**data["plan"]), won=int(data.get("won", 0)),
                   lost=int(data.get("lost", 0)), late=int(data.get("late", 0)))
        said = data.get("status")
        if said is not None and said != test.status:
            raise ValueError(f"запись говорит {said}, а её собственные числа — "
                             f"{test.status}")
        return test


def forecast(plan: Plan, share: float, start: float = 0.0) -> Dict[str, float]:
    """Wald's approximations: chance of accepting, and outcomes still needed.

    `share` is the true (or estimated) share of decided outcomes the change
    wins; `start` is the ratio already accumulated, so a running test can
    ask how far it still has to go. Ignores the overshoot and the ceiling.
    """
    a = plan.accept_at - start
    b = plan.reject_at - start
    if a <= 0.0:
        return {"accept": 1.0, "trials": 0.0}
    if b >= 0.0:
        return {"accept": 0.0, "trials": 0.0}
    p = min(max(float(share), 1e-9), 1.0 - 1e-9)
    w, l = plan.win_step, plan.loss_step
    drift = p * w + (1.0 - p) * l
    if abs(drift) < 1e-12:
        second = p * w * w + (1.0 - p) * l * l
        return {"accept": -b / (a - b), "trials": -a * b / second}
    h = _tilt(p, w, l, drift)
    top, bottom = math.exp(h * a), math.exp(h * b)
    accept = (1.0 - bottom) / (top - bottom)
    trials = (accept * a + (1.0 - accept) * b) / drift
    return {"accept": accept, "trials": max(0.0, trials)}


def _tilt(p: float, w: float, l: float, drift: float) -> float:
    """The non-zero h with p*e^(h*w) + (1-p)*e^(h*l) = 1.

    The function is convex, zero at h = 0 with slope `drift`, so the other
    root lies on the side opposite to the drift. Bisection: no scipy in
    the core.
    """
    def f(h: float) -> float:
        return p * math.exp(h * w) + (1.0 - p) * math.exp(h * l) - 1.0

    sign = -1.0 if drift > 0 else 1.0
    near, far = sign * 1e-9, sign * 1.0
    while f(far) < 0.0:
        far *= 2.0
    for _ in range(200):
        mid = (near + far) / 2.0
        if f(mid) < 0.0:
            near = mid
        else:
            far = mid
    return (near + far) / 2.0
