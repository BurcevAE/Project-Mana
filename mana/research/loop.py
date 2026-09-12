"""
mana.research.loop — tell the explanations apart, check the one that is
left, and close with how far the answer goes.

    tell apart  while some probe separates the explanations in play: the
                probe with the most expected bits per unit of cost
    check       once they are told apart: the probe that would bring the
                largest share of the task inside the answer's boundary,
                times what an error costs, net of what it costs
    close       when no allowed probe is worth what it costs. The answer
                carries its standing, decided in `core/standing.py`
    revise      a leader a check refuted may give way once to one that fits
                everything; a second refutation by a check is UNEXPLAINED,
                and so is nothing fitting after one revision around the
                explanation the world last refuted

Carried over from `cognition/inquiry.py` 1.14 in its boundary mode, line for
line where it decides anything -- the same order of probes, the same ties --
so that the two can be compared action by action. Only the paths its
experiments kept came along; its docstring records the rest.

Evidence is counted for the class (step 2b of docs/RESEARCH_CONTRACT.md).
An observation made while some explanation led counts, beyond itself, for
every explanation no available probe tells from that one: each of them
predicted it alike, so each was committed to it in advance. The lab loop
counted it for the leader alone, and the first audit through
`core/standing.py` found 13-18% of answers whose boundary depended on which
member of the class had won a tie of weights. `evidence=MEMBER` keeps the
lab's rule for the A/B and for reproducing the lab action by action.

Repeatability
-------------
That an outcome seen once in a situation is the outcome there is a claim
of the answer. A task that assumes it (`Stakes.assume_repeatable`, the
default) gets an answer that says so. A task that does not gets it tested:
a situation counts as known, and as backing anything beyond itself, only
once its outcome repeated; a repeat is worth what it would bring inside the
boundary, like any other probe; and one different outcome refutes every
explanation that predicted a single one.

Why a question closes on its own terms
---------------------------------------
What is still needed to tell the explanations apart is counted between
classes -- the explanations no available probe separates. Belief spread
inside a class is bits no probe can deliver, and owing them made the lab's
loop give up with three quarters of its budget unspent.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..core import standing
from .contract import (ALWAYS, CHECK, DISTINGUISH, NEVER, OTHER, READ_ONLY, REVERSIBLE,
                       Explanation, Probe, Question, Stakes, World)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.3"

# Whom an observation counts for, beyond itself.
CLASS = "class"         # every explanation no available probe tells from the leader then
MEMBER = "member"       # the leader then alone -- the lab's rule, kept for the A/B

#: The explanations are told apart when the leading class holds this much
#: of the belief.
CONFIDENCE = 0.95
#: A rival starts half as likely as the leader it challenges.
RIVAL_PRIOR = 0.5
#: A leader a check refuted may give way this many times.
REVISIONS_AFTER_REFUTATION = 1
#: Below this an explanation is refuted and skipped.
ALIVE = 1e-12
#: A set never grows past this; a leader with no room to be challenged is
#: accepted, and `capped` says so.
MAX_EXPLANATIONS = 600

# Why a question is open.
NOT_ENOUGH_DATA = "NOT_ENOUGH_DATA"
UNCERTAIN = "UNCERTAIN"
UNTESTED = "UNTESTED"
CONTRADICTION = "CONTRADICTION"

# How a question was settled.
ANSWERED = "ANSWERED"
UNEXPLAINED = standing.UNEXPLAINED
REFUTED = "проверки дважды опровергли объяснение"

# Why the loop stopped.
NO_QUESTIONS = "NO_QUESTIONS"
NOTHING_WORTH_ASKING = "NOTHING_WORTH_ASKING"
BUDGET = "BUDGET"


def _entropy(weights: Sequence[float]) -> float:
    return -sum(w * math.log2(w) for w in weights if w > 0.0)


class Explanations:
    """Competing explanations for one question, how much each is believed,
    and everything observed -- kept, because an explanation added later is
    weighed against the whole history, not only what comes after it."""

    def __init__(self, explanations: Sequence[Explanation]) -> None:
        self.members = list(explanations)
        total = sum(h.weight for h in self.members)
        if total <= 0:
            raise ValueError("a set of explanations needs positive weight")
        for h in self.members:
            h.prior = h.weight / total
            h.weight = h.prior
        self.history: List[Tuple[Dict[str, Any], Any]] = []
        #: Beside the history: (P(outcome) by the family then, was the
        #: family unanimous then, who led then).
        self.evidence: List[Optional[Tuple[float, bool, Optional[Explanation]]]] = []
        self._pending: Optional[Tuple[float, bool, Optional[Explanation]]] = None
        #: Whom an observation counts for beyond itself: CLASS or MEMBER.
        self.evidence_for = CLASS
        self.observations = 0
        self.contradicted = 0
        self.settled: Optional[Tuple[str, Tuple[str, ...]]] = None
        #: How the answer stands for the task, once settled.
        self.standing: Optional[standing.Answer] = None
        self.subject = ""
        self.stakes: Optional[Stakes] = None
        self.space: Optional[List[Dict[str, Any]]] = None
        self.challenged: set = set()
        self.capped = False
        self.check_phase = False
        self.claims_open = False
        self.refuted_by_check = 0
        self._last_refuted: Optional[Explanation] = None
        #: Can every situation of the world be listed? When not, the space
        #: is the situations known so far and grows; see contract.py.
        self.complete = True
        self._space_version = 0
        self._predicted: Dict[Tuple[int, Any], Dict[Any, float]] = {}
        self._signature: Dict[int, Tuple[int, tuple]] = {}
        self._relevant_cache: Dict[int, Tuple[int, Tuple[str, ...]]] = {}
        self._coverage_cache: Optional[Tuple[int, set]] = None
        self._repeat_cache: Optional[Tuple[int, set]] = None
        self._inside_cache: Optional[Tuple[Tuple[int, int, int], FrozenSet]] = None
        self._situations_cache: Optional[List[standing.Situation]] = None

    def set_space(self, situations: Sequence[Dict[str, Any]]) -> None:
        """The situations known for this question. They only grow; when they
        do, everything read off them is read again."""
        if self.space is not None and len(situations) == len(self.space):
            return
        self.space = list(situations)
        self._space_version += 1
        self._situations_cache = None
        self._inside_cache = None

    # ---------- predictions ----------

    def _dist(self, h: Explanation, params: Dict[str, Any]) -> Dict[Any, float]:
        key = params.get("key")
        if key is None:
            return h.predict(params)
        slot = (id(h), key)
        dist = self._predicted.get(slot)
        if dist is None:
            dist = self._predicted[slot] = h.predict(params)
        return dist

    def live(self) -> List[Explanation]:
        return [h for h in self.members if h.weight > ALIVE]

    def get(self, name: str) -> Optional[Explanation]:
        return next((h for h in self.members if h.name == name), None)

    def leader(self) -> Tuple[str, float]:
        best = max(self.members, key=lambda h: h.weight)
        return (best.name, best.weight)

    def _lead(self) -> Optional[Explanation]:
        named = [h for h in self.live() if h.name != OTHER]
        return max(named, key=lambda h: h.weight) if named else None

    # ---------- belief ----------

    def entropy(self) -> float:
        return _entropy([h.weight for h in self.live()])

    def expected_gain(self, params: Dict[str, Any]) -> float:
        """Bits this probe is expected to remove: H now minus H after."""
        live = self.live()
        dists = [self._dist(h, params) for h in live]
        predictive: Dict[Any, float] = {}
        for h, dist in zip(live, dists):
            for outcome, p in dist.items():
                predictive[outcome] = predictive.get(outcome, 0.0) + h.weight * p
        after = 0.0
        for outcome, p_outcome in predictive.items():
            if p_outcome <= 0.0:
                continue
            after += p_outcome * _entropy([h.weight * d.get(outcome, 0.0) / p_outcome
                                           for h, d in zip(live, dists)])
        return max(0.0, _entropy([h.weight for h in live]) - after)

    def discriminates(self, params: Dict[str, Any]) -> bool:
        """Do the explanations in play disagree about this probe? OTHER is
        left out: it is the absence of an explanation."""
        live = [h for h in self.live() if h.name != OTHER]
        return len({tuple(sorted(self._dist(h, params).items())) for h in live}) > 1

    def _family_true(self, params: Dict[str, Any]) -> Optional[float]:
        named = [h for h in self.live() if h.name != OTHER]
        mass = sum(h.weight for h in named)
        if mass <= 0.0:
            return None
        return sum(h.weight * self._dist(h, params).get(True, 0.0) for h in named) / mass

    def update(self, params: Dict[str, Any], outcome: Any) -> bool:
        """Fold one observation in. False when no explanation allowed it: the
        weights stay and the question is marked contradicted."""
        self.observations += 1
        live = self.live()
        raw = [h.weight * self._dist(h, params).get(outcome, 0.0) for h in live]
        total = sum(raw)
        if total <= 0.0:
            self.contradicted += 1
            return False
        for h, r in zip(live, raw):
            h.weight = r / total
        self.history.append((params, outcome))
        self.evidence.append(self._pending)
        self._pending = None
        return True

    def add(self, rivals: Sequence[Explanation], prior_each: float) -> int:
        """Admit new explanations and re-weigh everything from the history."""
        known = {h.name for h in self.members}
        fresh = [r for r in rivals if r.name not in known]
        room = MAX_EXPLANATIONS - len(self.members)
        if room < len(fresh):
            self.capped = True
            fresh = fresh[:max(0, room)]
        if not fresh:
            return 0
        for r in fresh:
            r.prior = prior_each
        self.members.extend(fresh)
        total = sum(h.prior for h in self.members)
        logs = []
        for h in self.members:
            h.prior /= total
            log_weight = math.log(h.prior) if h.prior > 0 else -math.inf
            for params, outcome in self.history:
                p = self._dist(h, params).get(outcome, 0.0)
                if p <= 0.0:
                    log_weight = -math.inf
                    break
                log_weight += math.log(p)
            logs.append(log_weight)
        top = max(logs)
        raw = [math.exp(l - top) if l > -math.inf else 0.0 for l in logs]
        norm = sum(raw)
        for h, r in zip(self.members, raw):
            h.weight = r / norm
        return len(fresh)

    # ---------- classes ----------

    def _classes(self, space: Sequence[Dict[str, Any]]) -> List[List[Explanation]]:
        """Live explanations grouped by what they predict on every probe in
        `space`. The space of a question is fixed, which lets it be cached."""
        groups: Dict[tuple, List[Explanation]] = {}
        for h in self.live():
            groups.setdefault(self._signature_of(h), []).append(h)
        return list(groups.values())

    def _signature_of(self, h: Explanation) -> tuple:
        """What h predicts on every situation of the question's space; two
        explanations with one signature are one class. Read again when the
        space grows -- a class can split when the world shows more."""
        cached = self._signature.get(id(h))
        if cached is not None and cached[0] == self._space_version:
            return cached[1]
        sig = tuple(tuple(sorted(self._dist(h, p).items())) for p in self.space or [])
        self._signature[id(h)] = (self._space_version, sig)
        return sig

    def leading_class(self, space: Sequence[Dict[str, Any]]
                      ) -> Tuple[Tuple[str, ...], float, Explanation]:
        """The explanations no probe in `space` tells apart, with the most
        belief between them, and the one of them with the most."""
        if not space:
            lead = max(self.live(), key=lambda h: h.weight)
            return ((lead.name,), lead.weight, lead)
        best = max(self._classes(space), key=lambda g: sum(h.weight for h in g))
        lead = max(best, key=lambda h: h.weight)
        return (tuple(sorted(h.name for h in best)), sum(h.weight for h in best), lead)

    def class_entropy(self, space: Sequence[Dict[str, Any]]) -> float:
        """Uncertainty some probe in `space` could still remove."""
        if not space:
            return self.entropy()
        return _entropy([sum(h.weight for h in g) for g in self._classes(space)])

    # ---------- what an explanation reads, and the boundary ----------

    def reads(self, h: Explanation) -> Tuple[str, ...]:
        """The conditions h's prediction turns on, read off the space: two
        situations that differ only in c and get different predictions make
        c one it reads. Nothing is assumed about h's form.

        That holds only where the space is every situation there is. In an
        incomplete world h says what it reads; if it does not, it is taken
        to read everything, which reaches no further than what was seen."""
        space = self.space or []
        if not self.complete:
            if h.reads is not None:
                return tuple(h.reads)
            return tuple(sorted({k for p in space for k in (p.get("conditions") or {})}))
        cached = self._relevant_cache.get(id(h))
        if cached is not None and cached[0] == self._space_version:
            return cached[1]
        names = sorted({k for p in space for k in (p.get("conditions") or {})})
        relevant = []
        for c in names:
            groups: Dict[tuple, set] = {}
            for p in space:
                cond = p["conditions"]
                key = tuple((k, bool(cond[k])) for k in names if k != c)
                groups.setdefault(key, set()).add(tuple(sorted(self._dist(h, p).items())))
            if any(len(g) > 1 for g in groups.values()):
                relevant.append(c)
        self._relevant_cache[id(h)] = (self._space_version, tuple(relevant))
        return tuple(relevant)

    @staticmethod
    def _config_key(params: Dict[str, Any]) -> Any:
        key = params.get("key")
        if key is not None:
            return key
        return tuple(sorted((k, bool(v)) for k, v in (params.get("conditions") or {}).items()))

    def observed_coverage(self) -> set:
        """COVERAGE: the situations observed, whoever each helped choose."""
        if self._coverage_cache is None or self._coverage_cache[0] != len(self.history):
            self._coverage_cache = (len(self.history),
                                    {self._config_key(p) for p, _ in self.history})
        return self._coverage_cache[1]

    def covered(self, params: Dict[str, Any]) -> bool:
        return self._config_key(params) in self.observed_coverage()

    def _repeated(self) -> set:
        """Situations whose outcome was seen more than once, the same every
        time -- a fact about the world, whoever led."""
        if self._repeat_cache is None or self._repeat_cache[0] != len(self.history):
            outcomes: Dict[Any, set] = {}
            counts: Dict[Any, int] = {}
            for params, outcome in self.history:
                key = self._config_key(params)
                outcomes.setdefault(key, set()).add(outcome)
                counts[key] = counts.get(key, 0) + 1
            self._repeat_cache = (len(self.history),
                                  {k for k, n in counts.items() if n > 1 and len(outcomes[k]) == 1})
        return self._repeat_cache[1]

    @property
    def _checking(self) -> bool:
        """Is repeatability a claim to test, rather than an assumption?"""
        return self.stakes is not None and not self.stakes.assume_repeatable

    def _backing(self, lead: Explanation) -> Tuple[set, List[Dict[str, Any]]]:
        """What `lead`'s boundary may rest on: the situations seen, and the
        independent observations -- when repeatability is checked, only
        where the outcome repeated."""
        observed = set(self.observed_coverage())
        independent = self._independent(lead)
        if self._checking:
            repeated = self._repeated()
            observed &= repeated
            independent = [p for p in independent if self._config_key(p) in repeated]
        return observed, [p.get("conditions") or {} for p in independent]

    def _vouches(self, record: Optional[Tuple[float, bool, Optional[Explanation]]],
                 params: Dict[str, Any], outcome: Any, lead: Explanation) -> bool:
        """EVIDENCE: may this observation count for `lead` beyond itself?
        Only if `lead` was committed to it before it happened: it led then --
        or, counting for the class, an explanation no available probe tells
        from it led then -- or the whole family agreed then and `lead` with
        it."""
        if record is None:
            return False
        right, unanimous, led_by = record[:3]
        if led_by is lead:
            return True
        if (self.evidence_for == CLASS and led_by is not None and self.space
                and self._signature_of(led_by) == self._signature_of(lead)):
            return True
        return unanimous and abs(self._dist(lead, params).get(outcome, 0.0) - right) < 1e-9

    def _independent(self, lead: Explanation) -> List[Dict[str, Any]]:
        return [params for (params, outcome), record in zip(self.history, self.evidence)
                if self._vouches(record, params, outcome, lead)]

    def _situations(self) -> List[standing.Situation]:
        if self._situations_cache is None:
            self._situations_cache = [(self._config_key(p), p.get("conditions") or {})
                                      for p in self.space or []]
        return self._situations_cache

    def _weight(self, params: Dict[str, Any]) -> float:
        return max(0.0, float(self.stakes.weight(params))) if self.stakes else 1.0

    def inside(self, lead: Explanation, extra: Optional[Dict[str, Any]] = None) -> FrozenSet:
        """`lead`'s boundary, as core draws it; `extra` -- as if one more
        independent observation were made there (and, when repeatability is
        checked, as if it repeated what was seen)."""
        key = (id(lead), len(self.history), self._space_version)
        if extra is None and self._inside_cache is not None and self._inside_cache[0] == key:
            return self._inside_cache[1]
        observed, independent = self._backing(lead)
        if extra is not None:
            observed.add(self._config_key(extra))
            independent.append(extra.get("conditions") or {})
        backed = standing.inside(self._situations(), self.reads(lead), observed,
                                 independent, self.stakes.rule)
        if extra is None:
            self._inside_cache = (key, backed)
        return backed

    def stake(self, params: Dict[str, Any]) -> float:
        """What a probe here would bring inside the leader's boundary -- a
        share of the task's weight -- times what a wrong answer costs. Valued
        as if it confirms; if it refutes, the answer changes, which is worth
        more. Nothing where the outcome is already known -- seen, or, when
        repeatability is checked, seen to repeat."""
        if self.stakes is None or not self.space:
            return 0.0
        if self.covered(params) and (not self._checking
                                     or self._config_key(params) in self._repeated()):
            return 0.0
        lead = self._lead()
        if lead is None:
            return 0.0
        space = self.space
        total = sum(self._weight(q) for q in space)
        if total <= 0.0:
            return 0.0
        now = self.inside(lead)
        after = self.inside(lead, params)
        gained = sum(self._weight(q) for q in space
                     if self._config_key(q) in after and self._config_key(q) not in now)
        return self.stakes.error_cost * gained / total

    def answer_for(self, lead: Explanation) -> standing.Answer:
        observed, independent = self._backing(lead)
        task = {self._config_key(q): self._weight(q) for q in self.space or []}
        return standing.standing(self.subject, lead.name, self._situations(), self.reads(lead),
                                 observed, independent, self.stakes.rule, task,
                                 repeatability=(standing.REPEATABLE_CHECKED if self._checking
                                                else standing.REPEATABLE_ASSUMED))

    def note(self, params: Dict[str, Any], outcome: Any, check: bool) -> None:
        """Before `update`: what the evidence keeps about this observation --
        who led, whether the family agreed -- and whether it refuted the
        leader; a refutation by a check is counted."""
        p = self._family_true(params)
        lead = self._lead()
        if lead is not None and self._dist(lead, params).get(outcome, 0.0) <= 0.5:
            self._last_refuted = lead
            if check:
                self.refuted_by_check += 1
        right = (p if outcome else 1.0 - p) if p is not None else 0.0
        self._pending = (right, not self.discriminates(params), lead)


def condition_explanations(conditions: Sequence[str], noise: float = 0.0,
                           other: float = 0.05) -> Explanations:
    """Does an action's outcome depend on one observable condition? For
    every condition "it works when this holds"; plus always, never, and
    OTHER. Nothing about any particular world is in it."""
    def bern(yes: bool) -> Dict[bool, float]:
        return ({True: 1.0 - noise, False: noise} if yes
                else {True: noise, False: 1.0 - noise})

    named = [Explanation(f"зависит от {c}",
                         lambda params, c=c: bern(bool(params["conditions"][c])), reads=(c,))
             for c in conditions]
    named.append(Explanation(ALWAYS, lambda params: bern(True), reads=()))
    named.append(Explanation(NEVER, lambda params: bern(False), reads=()))
    share = (1.0 - other) / len(named)
    for h in named:
        h.weight = share
    return Explanations(named + [Explanation(OTHER, lambda params: {True: 0.5, False: 0.5},
                                             weight=other, reads=())])


def pairwise_rivals(conditions: Sequence[str]) -> Callable[[Explanation], List[Explanation]]:
    """Explanations that agree with a leader except in one combination of two
    conditions -- the challenge a leader has to survive before it is taken.
    Outcomes are taken to be boolean."""
    pairs = list(itertools.combinations(conditions, 2))

    def challenge(leader: Explanation) -> List[Explanation]:
        out = []
        for d1, d2 in pairs:
            for x in (False, True):
                for y in (False, True):
                    def predict(params, base=leader.predict, d1=d1, d2=d2, x=x, y=y):
                        dist = base(params)
                        seen = params["conditions"]
                        if bool(seen[d1]) == x and bool(seen[d2]) == y:
                            return {True: dist.get(False, 0.0), False: dist.get(True, 0.0)}
                        return dist
                    reads = (tuple(sorted(set(leader.reads) | {d1, d2}))
                             if leader.reads is not None else None)
                    out.append(Explanation(
                        f"{leader.name}, кроме {d1}={int(x)} и {d2}={int(y)}", predict,
                        reads=reads))
        return out
    return challenge


def _unexplained(e: Explanations, names: Tuple[str, ...]) -> Tuple[str, Tuple[str, ...]]:
    e.settled = (UNEXPLAINED, names)
    e.standing = standing.unexplained(e.subject, observed=len(e.history))
    return e.settled


def settle(e: Explanations, space: Sequence[Dict[str, Any]],
           challenge: Optional[Callable[[Explanation], List[Explanation]]] = None,
           confidence: float = CONFIDENCE) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Is this question answered? The same test whichever policy gathered
    the observations."""
    if e.settled:
        return e.settled
    e.check_phase = False
    if e.refuted_by_check > REVISIONS_AFTER_REFUTATION:
        return _unexplained(e, (REFUTED,))
    names, mass, lead = e.leading_class(space)
    if mass < confidence:
        return None
    if names == (OTHER,):
        lost = e._last_refuted
        if challenge is not None and lost is not None and lost.name not in e.challenged:
            # Nothing fits any more: one revision around the explanation the
            # world last refuted before calling the question unexplained.
            e.challenged.add(lost.name)
            e.add(challenge(lost), prior_each=RIVAL_PRIOR * lost.prior)
            e._last_refuted = None
            return None
        return _unexplained(e, names)
    if challenge is not None and lead.name not in e.challenged:
        e.challenged.add(lead.name)
        e.add(challenge(lead), prior_each=RIVAL_PRIOR * lead.prior)
        names, mass, lead = e.leading_class(space)
        if mass < confidence:
            return None                     # the challenge reopened it
        if names == (OTHER,):
            return _unexplained(e, names)
        if lead.name not in e.challenged and not e.capped:
            return None                     # a rival leads now; it is challenged next
    e.check_phase = True                    # told apart; only checking is left
    if e.claims_open:
        return None                         # a probe the task rests on is worth its cost
    e.standing = e.answer_for(lead)
    e.settled = (ANSWERED, names)
    return e.settled


def unsettled(knowledge: Dict[str, Explanations]) -> List[Question]:
    """A detector: every question not yet settled, and why it is open."""
    out: List[Question] = []
    for subject, e in knowledge.items():
        if e.settled:
            continue
        name, _ = e.leader()
        why = (NOT_ENOUGH_DATA if e.observations == 0
               else CONTRADICTION if e.contradicted
               else UNTESTED if e.challenged
               else UNEXPLAINED if name == OTHER
               else UNCERTAIN)
        out.append(Question(subject=subject, why_open=why, explanations=e,
                            source={"action": subject}))
    return out


@dataclass
class _Priced:
    question: Question
    probe: Probe
    gain: float
    estimated_calls: int
    value: float
    motive: str


@dataclass
class Step:
    subject: str
    why_open: str
    action: str
    situation: Any
    cost: int
    gain: float
    value: float
    outcome: Any
    leader: str
    weight: float
    explained: bool
    motive: str


@dataclass
class Report:
    steps: List[Step]
    spent: int
    stopped: str
    answers: Dict[str, standing.Answer]
    open_questions: List[Tuple[str, str, str, float]]


def inquire(detect: Callable[[], List[Question]], world: World, budget: int, stakes: Stakes,
            challenge: Optional[Callable[[Explanation], List[Explanation]]] = None,
            allowed: Sequence[str] = (READ_ONLY, REVERSIBLE),
            confidence: float = CONFIDENCE, evidence: str = CLASS) -> Report:
    """Settle what can be settled, then act where a question gains most per
    unit of cost, until nothing open is worth what it would take.
    `evidence`: whom an observation counts for beyond itself (CLASS, or
    MEMBER to do what the lab loop did)."""
    if evidence not in (CLASS, MEMBER):
        raise ValueError(f"evidence counts for {CLASS!r} or {MEMBER!r}, not {evidence!r}")
    steps: List[Step] = []
    answers: Dict[str, standing.Answer] = {}
    spent = 0
    closing = _entropy([confidence, 1.0 - confidence])
    known = getattr(world, "situations", None)
    complete = bool(getattr(world, "complete", True))
    while True:
        candidates: List[_Priced] = []
        still_open = 0
        blocked_by_budget = False
        for question in detect():
            e = question.explanations
            if e.stakes is None:
                e.stakes = stakes
                e.subject = question.subject
                e.evidence_for = evidence
            offered = world.probes_for(question)
            e.complete = complete
            e.set_space(known(question) if known is not None else [p.params for p in offered])
            probes = [p for p in offered if p.safety in allowed]
            # Open while some allowed probe protects more of the task than
            # it costs.
            e.claims_open = any(e.stake(p.params) > max(1, int(p.cost)) for p in probes)
            # What the world could tell apart decides when a question is
            # answered; what MANA is allowed to do decides only what it does
            # next.
            space = e.space
            done = settle(e, space, challenge, confidence)
            if done:
                answers[question.subject] = e.standing
                continue
            still_open += 1
            priced: List[_Priced] = []
            for probe in probes:
                cost = max(1, int(probe.cost))
                if e.check_phase:
                    # Told apart: a probe is worth the error it could prevent,
                    # net of its cost -- in actions, not bits.
                    stake = e.stake(probe.params)
                    if stake > cost:
                        priced.append(_Priced(question, probe, stake, cost,
                                              (stake - cost) / cost, CHECK))
                    continue
                gain = e.expected_gain(probe.params)
                priced.append(_Priced(question, probe, gain, cost, gain / cost, DISTINGUISH))
            best_rate = max((p.value for p in priced if p.motive == DISTINGUISH), default=0.0)
            checks_here = any(p.motive == CHECK for p in priced)
            if best_rate <= 0.0 and not checks_here:
                continue                    # no allowed probe says anything here
            if best_rate > 0.0:
                needed = max(0.0, e.class_entropy(space) - closing)
                if needed / best_rate > budget - spent:
                    blocked_by_budget = True    # would not finish in what is left
                    continue
            affordable = [p for p in priced
                          if p.gain > 0.0 and p.estimated_calls <= budget - spent]
            if not affordable:
                blocked_by_budget = True
            candidates.extend(affordable)
        if still_open == 0:
            stopped = NO_QUESTIONS
            break
        if not candidates:
            stopped = BUDGET if blocked_by_budget else NOTHING_WORTH_ASKING
            break
        # Bits and actions are different units: tell explanations apart
        # first, check when nothing is left to tell.
        pool = [p for p in candidates if p.motive == DISTINGUISH] or candidates
        chosen = max(pool, key=lambda p: p.value)
        outcome = world.act(chosen.probe)
        spent += chosen.estimated_calls
        e = chosen.question.explanations
        e.note(chosen.probe.params, outcome, chosen.motive == CHECK)
        explained = e.update(chosen.probe.params, outcome)
        name, weight = e.leader()
        steps.append(Step(subject=chosen.question.subject, why_open=chosen.question.why_open,
                          action=chosen.probe.action, situation=chosen.probe.params.get("key"),
                          cost=chosen.estimated_calls, gain=round(chosen.gain, 4),
                          value=round(chosen.value, 4), outcome=outcome, leader=name,
                          weight=round(weight, 4), explained=explained, motive=chosen.motive))
    remaining = [(q.subject, q.why_open) + q.explanations.leader() for q in detect()]
    return Report(steps=steps, spent=spent, stopped=stopped, answers=answers,
                  open_questions=remaining)
