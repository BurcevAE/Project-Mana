"""
mana.cognition.curriculum — deciding what to practise next, without being
able to rig the exam.

    capabilities -> gaps -> learning goals -> lessons -> evidence -> repeat

What "learning" means here, stated plainly
------------------------------------------
MANA does not adjust model weights. A lesson produces *evidence*, and
evidence changes three things: what the self-model believes it can do,
which program the population offers for a niche, and which brain the pool
prefers for a kind of task. That is a real and measurable form of
learning, and it is not the same thing as training -- calling it training
would be the kind of overclaim this project exists to avoid.

The rigging problem
-------------------
A system that chooses its own curriculum can choose an easy one. The
defence is structural rather than a rule: the curriculum picks only
*which* generator to draw from and at what difficulty; the tasks
themselves, their answers and their grading all come from `mana.core`,
which it cannot reach. So it can decide to practise hard arithmetic and
cannot decide what "hard arithmetic" means or what counts as a correct
answer.

The second defence is against a subtler cheat: re-serving tasks already
seen. A curriculum that keeps drawing from the same seed measures
memorisation and reports it as progress. Every lesson therefore takes a
fresh seed, and used seeds are recorded so a resumed curriculum cannot
accidentally repeat one.

Progression is earned, not counted
----------------------------------
A stage advances when the evidence supports it -- the capability clears
the competence bar with a confidence interval tight enough to mean
something -- not after N attempts. Advancing on attempt count is how a
curriculum walks a system straight into material it cannot do and then
records a long run of failures as a capability profile.

Stopping
--------
Three conditions, the same shape as the experiment lab: budget, a lesson
cap, and diminishing returns measured on realised information rather than
on success. A curriculum that keeps practising something already known
consumes the budget an unexplored niche needed.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import tasks as core_tasks
from .gaps import COMPETENCE, KNOWLEDGE, Gap, detect
from .self_model import MIN_OBSERVATIONS, Observation, SelfModel, band_of

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Stages in order. Difficulty first, then the two that are not about
#: difficulty at all: adversarial material aimed at a known failure
#: pattern, and the same competence somewhere it has never been used.
EASY, MEDIUM, HARD, ADVERSARIAL, TRANSFER = "easy", "medium", "hard", "adversarial", "transfer"
STAGES = (EASY, MEDIUM, HARD, ADVERSARIAL, TRANSFER)

_STAGE_DIFFICULTY: Dict[str, Tuple[float, float]] = {
    EASY: (0.05, 0.30),
    MEDIUM: (0.30, 0.60),
    HARD: (0.60, 0.95),
    ADVERSARIAL: (0.60, 0.95),
    TRANSFER: (0.30, 0.70),
}

#: A stage is passed when the capability's whole interval clears this.
#: The interval, not the point estimate: passing on a lucky mean is how a
#: curriculum advances into material the system cannot do.
PASS_BAR = 0.70

#: Tasks per lesson. Enough that a Wilson interval says something, small
#: enough that a wrong choice of goal costs one lesson rather than a day.
LESSON_SIZE = 12

#: Consecutive lessons without measurable movement before stopping.
DIMINISHING_WINDOW = 3
MIN_LESSON_VALUE = 0.03


@dataclass
class LearningGoal:
    """Something worth practising, and why.

    Carries the gap it came from so a finished curriculum can be read
    backwards: which weakness produced which lesson, and did the evidence
    move.
    """
    goal_id: str
    domain: str
    stage: str
    reason: str
    source_gap: str = ""
    kind: str = COMPETENCE
    priority: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Lesson:
    """One batch of generated practice, and what came back."""
    lesson_id: str
    goal: LearningGoal
    seed: int
    size: int
    difficulty: Tuple[float, float]
    observations: int = 0
    correct: int = 0
    calls_used: int = 0
    elapsed: float = 0.0
    before: Optional[Tuple[float, float]] = None     # capability interval before
    after: Optional[Tuple[float, float]] = None      # and after
    passed: bool = False

    @property
    def accuracy(self) -> float:
        return self.correct / self.observations if self.observations else 0.0

    @property
    def information(self) -> float:
        """How much the interval actually narrowed.

        The realised value of the lesson, used for the diminishing-returns
        check. Measured on resolution rather than on accuracy: a lesson
        that confirms a capability is poor has taught the system
        something, and one that changes nothing has not, whatever the
        score was.
        """
        if not self.after:
            return 0.0
        # No prior interval means nothing was known, and "nothing known"
        # IS an interval: the whole unit range. Scoring the first lesson
        # at zero would rank the most informative measurement a system
        # ever takes as worthless, and the diminishing-returns check would
        # then stop a curriculum on its very first move.
        before = self.before if self.before else (0.0, 1.0)
        return max(0.0, (before[1] - before[0]) - (self.after[1] - self.after[0]))

    def summary(self) -> Dict[str, Any]:
        return {"lesson": self.lesson_id, "domain": self.goal.domain,
                "stage": self.goal.stage, "size": self.size,
                "accuracy": round(self.accuracy, 3),
                "information": round(self.information, 4),
                "passed": self.passed, "calls": self.calls_used,
                "elapsed": round(self.elapsed, 1)}


def goals_from_model(model: SelfModel, domains: Sequence[str] = (),
                     limit: int = 4) -> List[LearningGoal]:
    """Turn detected gaps into things to practise.

    A knowledge gap becomes a lesson at its own band -- it needs
    measurement, and measurement is exactly what a lesson produces. A
    competence gap becomes a lesson one stage BELOW where it failed:
    practising at the level something is already failing at produces more
    failures, while the band underneath is where the missing ability
    actually lives.
    """
    out: List[LearningGoal] = []
    for gap in detect(model):
        if domains and gap.domain not in domains:
            continue
        stage = gap.band if gap.band in STAGES else MEDIUM
        if gap.kind == COMPETENCE:
            index = STAGES.index(stage) if stage in STAGES else 1
            stage = STAGES[max(0, index - 1)]
        out.append(LearningGoal(
            goal_id=uuid.uuid4().hex[:10], domain=gap.domain, stage=stage,
            reason=gap.description, source_gap=gap.gap_id, kind=gap.kind,
            priority=gap.priority))
        if len(out) >= limit:
            break
    return out


def goals_from_coverage(coverage: Dict[str, Any], limit: int = 3) -> List[LearningGoal]:
    """Empty niches are goals too.

    A gap needs observations to be detected at all, so a niche nothing has
    ever attempted produces no gap and would otherwise never be visited --
    the blind spot a purely gap-driven curriculum cannot see.
    """
    out: List[LearningGoal] = []
    for niche in (coverage.get("empty") or [])[:limit]:
        domain, _, band = niche.partition("/")
        out.append(LearningGoal(
            goal_id=uuid.uuid4().hex[:10], domain=domain,
            stage=band if band in STAGES else MEDIUM,
            reason=f"ниша {niche} ни разу не пробовалась",
            kind=KNOWLEDGE, priority=0.5))
    return out


#: Answers one task, reporting correctness and what it cost. Injected so a
#: curriculum can be driven by a live agent, a program runtime, or a
#: simulation in tests.
LessonRunner = Callable[[Any], Tuple[bool, str, int]]


class Curriculum:
    """Chooses lessons, runs them, and knows when to stop."""

    def __init__(self, model: SelfModel, budget_calls: int = 1500,
                 max_lessons: int = 20, lesson_size: int = LESSON_SIZE) -> None:
        self.model = model
        self.budget_calls = budget_calls
        self.max_lessons = max_lessons
        self.lesson_size = lesson_size
        self.calls_used = 0
        self.lessons: List[Lesson] = []
        self.used_seeds: set = set()
        self._seed_counter = 0

    # ---------- stopping ----------

    def budget_left(self) -> int:
        return max(0, self.budget_calls - self.calls_used)

    def stop_reason(self) -> str:
        if self.budget_left() <= 0:
            return f"бюджет исчерпан ({self.calls_used}/{self.budget_calls} вызовов)"
        if len(self.lessons) >= self.max_lessons:
            return f"достигнут предел уроков ({self.max_lessons})"
        recent = [l.information for l in self.lessons[-DIMINISHING_WINDOW:]]
        if len(recent) >= DIMINISHING_WINDOW and max(recent) < MIN_LESSON_VALUE:
            return (f"убывающая отдача: последние {DIMINISHING_WINDOW} уроков "
                    f"ничего не уточнили")
        return ""

    # ---------- planning ----------

    def next_seed(self) -> int:
        """A seed nothing has drawn from before.

        Reusing one measures memorisation and reports it as progress,
        which is the quietest way a self-directed curriculum can lie about
        its own results.
        """
        while True:
            self._seed_counter += 1
            seed = 90_000 + self._seed_counter * 7919      # coprime stride
            if seed not in self.used_seeds:
                self.used_seeds.add(seed)
                return seed

    def plan_lesson(self, goal: LearningGoal) -> Optional[Lesson]:
        if goal.domain not in core_tasks.DOMAINS:
            return None
        difficulty = _STAGE_DIFFICULTY.get(goal.stage, _STAGE_DIFFICULTY[MEDIUM])
        return Lesson(lesson_id=uuid.uuid4().hex[:10], goal=goal, seed=self.next_seed(),
                      size=self.lesson_size, difficulty=difficulty)

    def _interval(self, domain: str, band: str) -> Optional[Tuple[float, float]]:
        cap = self.model.capabilities().get(f"{domain}/{band}")
        return cap.confidence_interval if cap else None

    # ---------- running ----------

    def run_lesson(self, lesson: Lesson, runner: LessonRunner,
                   verifier: Any = None) -> Lesson:
        """Generate the practice, run it, and fold the evidence in.

        Tasks come from `core.tasks` -- the curriculum chose the domain and
        the difficulty, and nothing else. It cannot decide what a hard
        arithmetic task looks like, nor what a correct answer is.
        """
        from ..core import oracle

        band = band_of(sum(lesson.difficulty) / 2)
        lesson.before = self._interval(lesson.goal.domain, band)
        generated = core_tasks.generate(lesson.goal.domain, lesson.size, lesson.seed,
                                        difficulty_range=lesson.difficulty)
        started = time.perf_counter()
        for task in generated:
            failed = False
            try:
                correct, brain, calls = runner(task)
            except Exception:
                # A failed attempt is a wrong answer, not a lost lesson --
                # dropping it would quietly bias the sample toward the
                # tasks that happened to run.
                correct, brain, calls, failed = False, "", 0, True
            # But it is not the SAME kind of wrong answer, and recording
            # it as one hid a completely unwired runner behind a
            # plausible result: 72 observations, zero calls, a capability
            # profile of solid zeros that read like a weak model. The
            # failure mode goes into the record, so reading it back
            # separates "answered incorrectly" from "never answered".
            grade_reason = "error" if failed else ("" if correct else "wrong")
            self.model.record(Observation(
                task_id=task.task_id, domain=task.domain, difficulty=task.difficulty,
                correct=bool(correct), reason=grade_reason, brain=brain, calls=int(calls)))
            lesson.observations += 1
            lesson.correct += int(bool(correct))
            lesson.calls_used += int(calls)
        lesson.elapsed = time.perf_counter() - started
        self.calls_used += lesson.calls_used
        lesson.after = self._interval(lesson.goal.domain, band)
        lesson.passed = self._passed(lesson, band)
        self.lessons.append(lesson)
        return lesson

    def _passed(self, lesson: Lesson, band: str) -> bool:
        """A stage is cleared when the whole interval clears the bar.

        The interval rather than the mean: passing on a lucky mean walks
        the curriculum into material the system cannot do, and then records
        a long run of failures as though it were a capability profile.
        """
        cap = self.model.capabilities().get(f"{lesson.goal.domain}/{band}")
        if cap is None or cap.observations < MIN_OBSERVATIONS:
            return False
        return cap.confidence_interval[0] >= PASS_BAR

    def next_stage(self, goal: LearningGoal) -> Optional[LearningGoal]:
        """Where to go after a stage is passed.

        Forward through the difficulty stages, then to the two that are
        not about difficulty. Returns None at the end rather than looping,
        so a mastered domain stops consuming budget.
        """
        try:
            index = STAGES.index(goal.stage)
        except ValueError:
            return None
        if index + 1 >= len(STAGES):
            return None
        nxt = STAGES[index + 1]
        return LearningGoal(goal_id=uuid.uuid4().hex[:10], domain=goal.domain, stage=nxt,
                            reason=f"стадия {goal.stage} пройдена, дальше {nxt}",
                            source_gap=goal.source_gap, kind=goal.kind,
                            priority=goal.priority)

    def step(self, goals: Sequence[LearningGoal], runner: LessonRunner) -> Optional[Lesson]:
        """One iteration: pick the highest-priority goal that fits and run it."""
        if self.stop_reason():
            return None
        for goal in sorted(goals, key=lambda g: -g.priority):
            lesson = self.plan_lesson(goal)
            if lesson is None:
                continue
            return self.run_lesson(lesson, runner)
        return None

    # ---------- reporting ----------

    def report(self) -> Dict[str, Any]:
        passed = [l for l in self.lessons if l.passed]
        return {
            "lessons": len(self.lessons),
            "passed": len(passed),
            "calls_used": self.calls_used,
            "budget_left": self.budget_left(),
            "stop_reason": self.stop_reason(),
            "total_information": round(sum(l.information for l in self.lessons), 4),
            "history": [l.summary() for l in self.lessons],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"calls_used": self.calls_used, "used_seeds": sorted(self.used_seeds),
                   "lessons": [l.summary() for l in self.lessons]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
