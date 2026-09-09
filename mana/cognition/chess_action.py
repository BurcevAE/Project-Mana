"""
mana.cognition.chess_action — from a finding to a change, and to its test.

The gap this closes
--------------------
Everything before this observes. A finding says "the side with more of X
lost", and there it stopped: MANA could describe its own play in detail
and could not act on the description. That makes an analyst, not an agent,
and the difference is the whole point of the project.

    аналитик:  «X связан с исходом»
    агент:     «X связан с исходом. Если связь причинная, изменение X
                изменит исход. Меняю X и смотрю.»

Nobody tells it what to change
--------------------------------
The change is derived from the finding by one general operator, not by a
person deciding what pawn moves mean:

    находка: у кого больше X — тот чаще проигрывает
    действие: среди ходов, которые поиск оценил одинаково,
              выбирать тот, у которого X меньше

That works for any property in the vocabulary and encodes no opinion about
any of them. The direction comes from the measurement's own sign; the
property comes from whichever finding was accepted.

Why the tie is the right place to intervene
--------------------------------------------
Measured, not assumed: 61 to 77 per cent of moves are chosen with no
margin at all -- the search rates several moves identically and a shuffle
picks one. That is the one place where behaviour can change without
overriding anything the search actually decided, and it is large enough
for a change there to matter. A tie-break replaces a coin toss, not a
judgement.

Reach comes before the experiment
----------------------------------
Executable is not the same as able to change anything. A tie-break moves a
move only where the property differs among the moves the search rated
equally, and for a search that evaluates by material, the material after
a tied move is almost always equal too. Measured on three hundred real
ties: `pieces` differed in none of them, `material` and `captures` in one
per cent, `pawn_moves` in fifty-nine.

Seventy games each were spent on the first three before anybody asked.
Their verdict was REJECTED, which reads as "the finding was tested and
failed" and meant "the lever never moved". Reach is measured first now,
from positions already recorded, costs no games, and travels with the
experiment: a lever that cannot move anything is NOT_EVALUATED with the
reason, never REJECTED.

Executability comes before the experiment
------------------------------------------
A property is only a candidate if it can be computed for a move *before
that move is played*, by the same code that will then use it. Several
cannot: how many legal moves a side will have depends on the opponent's
reply, and how many moves the search considered is a property of the
search rather than of a move. Those are refused with the reason, and a
refusal is a result -- it names a gap between what MANA can measure and
what it can act on.

The check is run, not reasoned about: every candidate scorer is executed
on a real board before anything is spent on it. This project has been
caught before by a candidate that looked executable and was not.

What the experiment is
-----------------------
The changed player plays the unchanged one. Colours alternate, so the
first move is not the thing being measured, and the outcome of each game
is one observation. That is A against B in the strictest available form:
the two differ in exactly one tie-break rule, and everything else --
depth, evaluation, opening, the shuffle's remaining role -- is identical.

A verdict here is a causal claim and is treated as one: ACCEPTED means the
change moved the result, not that the property correlates with it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.gates import ACCEPTED, MIN_PAIRED_TRIALS, NOT_EVALUATED, REJECTED
from . import chess_outcome as outcome
from . import findings as ledger_mod

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

QUESTION = "меняет ли исход выбор среди равных ходов по этому свойству"

ALPHA = 0.05
NO_EFFECT = 0.5

#: Prefer more of the property, or less of it.
MORE, LESS = 1, -1


def _material_after(board: Any, move: Any) -> float:
    from . import chess_features as feat

    mover = board.turn
    board.push(move)
    try:
        return float(feat._material(board, mover))
    finally:
        board.pop()


def _pieces_after(board: Any, move: Any) -> float:
    mover = board.turn
    board.push(move)
    try:
        return float(sum(1 for piece in board.piece_map().values()
                         if piece.color == mover))
    finally:
        board.pop()


#: How to score a candidate move for each property, before it is played.
#: A property absent from here is not actionable, and saying so is a
#: result rather than an omission: it marks a gap between what MANA can
#: measure about itself and what it can do something about.
SCORERS: Dict[str, Callable[[Any, Any], float]] = {
    "captures": lambda board, move: 1.0 if board.is_capture(move) else 0.0,
    "checks_given": lambda board, move: 1.0 if board.gives_check(move) else 0.0,
    "promotions": lambda board, move: 1.0 if move.promotion else 0.0,
    "pawn_moves": lambda board, move: (
        1.0 if (board.piece_at(move.from_square)
                and board.piece_at(move.from_square).piece_type == 1) else 0.0),
    "king_moves": lambda board, move: (
        1.0 if (board.piece_at(move.from_square)
                and board.piece_at(move.from_square).piece_type == 6) else 0.0),
    "material": _material_after,
    "pieces": _pieces_after,
}

#: Why the others cannot be acted on. Stated rather than left blank: a
#: capability gap that nobody wrote down is one nobody closes.
NOT_ACTIONABLE = {
    "legal_moves": "зависит от ответа соперника, до хода не вычисляется",
    "checks_taken": "свойство позиции, в которую меня поставили, а не моего хода",
    "material_gained_3": "измеряется через три хода, а выбирать надо сейчас",
    "considered": "свойство поиска, а не хода",
    "tied_at_top": "свойство поиска, а не хода",
    "margin": "свойство поиска, а не хода",
}


@dataclass
class Change:
    """One derived intervention: break ties by a property, in a direction.

    `from_finding` is the provenance. A change nobody can trace back to
    the measurement that suggested it is a change somebody made up, and
    this project has required the trace since rules were first generated
    from diagnoses.
    """
    property: str
    direction: int
    window: str = outcome.WHOLE
    from_finding: str = ""
    share_seen: float = 0.0
    #: Other findings that suggested the same action -- the two windows
    #: usually agree, and one action answered once is one answer.
    also_from: tuple = ()

    def describe(self) -> str:
        way = "больше" if self.direction == MORE else "меньше"
        return (f"среди равных ходов выбирать тот, у которого "
                f"«{self.property}» {way}")

    def as_dict(self) -> Dict[str, Any]:
        return {"property": self.property, "direction": self.direction,
                "from_finding": self.from_finding,
                "also_from": list(self.also_from), "what": self.describe()}

    def choose(self, board: Any, tied: Sequence[Any],
               rng: random.Random) -> Any:
        """Pick among moves the search rated identically.

        Ties inside the tie are still broken by the shuffle: the change
        is meant to add one criterion, not to install a deterministic
        order that a single unrelated property would then dictate.
        """
        score = SCORERS[self.property]
        best, chosen = None, []
        for move in tied:
            value = score(board, move) * self.direction
            if best is None or value > best:
                best, chosen = value, [move]
            elif value == best:
                chosen.append(move)
        return rng.choice(chosen) if chosen else rng.choice(list(tied))


def executable(name: str) -> Tuple[bool, str]:
    """Can this property be computed for a move before it is played?

    Run, not reasoned about: the scorer is executed on a real board. A
    candidate that looks executable and is not has cost this project a
    discovery budget before.
    """
    if name not in SCORERS:
        return (False, NOT_ACTIONABLE.get(name, "нет способа вычислить до хода"))
    try:
        import chess

        board = chess.Board()
        for move in list(board.legal_moves)[:3]:
            value = SCORERS[name](board, move)
            float(value)
        return (True, "вычисляется до хода")
    except Exception as exc:
        return (False, f"не выполнилось: {type(exc).__name__}: {exc}")


def propose(findings: Sequence[ledger_mod.Finding]
            ) -> Tuple[List[Change], List[Dict[str, str]]]:
    """Turn accepted findings into changes, and say why the rest cannot.

    The direction is the measurement's own: where the side with more of a
    property lost, the change prefers less of it. Nobody decides what any
    property means.
    """
    changes: List[Change] = []
    refused: List[Dict[str, str]] = []
    for finding in findings:
        if finding.verdict != ACCEPTED or finding.question != outcome.QUESTION:
            continue
        name = str(finding.approach.get("property", ""))
        share = float(finding.measurement.get("effect") or 0.0)
        can, why = executable(name)
        if not can:
            refused.append({"property": name, "why": why,
                            "finding": finding.finding_id})
            continue
        change = Change(
            property=name,
            direction=MORE if share > NO_EFFECT else LESS,
            window=str(finding.approach.get("window", outcome.WHOLE)),
            from_finding=finding.finding_id, share_seen=share)
        # The same action can be suggested twice -- once by the whole-game
        # window and once by the early one -- and running it twice would
        # spend a budget on the same question and count one answer as two.
        twin = next((c for c in changes if c.property == change.property
                     and c.direction == change.direction), None)
        if twin is None:
            changes.append(change)
        else:
            twin.also_from = getattr(twin, "also_from", ()) + (finding.finding_id,)
    return (changes, refused)


class Tuned:
    """The unchanged player with one tie-break added.

    A wrapper rather than a new player: A and B have to differ in exactly
    one thing, and the surest way to guarantee that is for B to be A with
    a single call intercepted.
    """

    def __init__(self, player: Any, change: Change) -> None:
        self.player = player
        self.change = change
        self.name = f"{getattr(player, 'name', 'player')}+{change.property}"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.player, name)

    def choose(self, board: Any, rng: random.Random) -> Any:
        move = self.player.choose(board, rng)
        scores = getattr(self.player, "_root_scores", None)
        if not scores:
            return move
        top = max(score for _, score in scores)
        tied_san = [san for san, score in scores if score == top]
        if len(tied_san) < 2:
            return move                     # the search decided; leave it
        tied = [candidate for candidate in board.legal_moves
                if board.san(candidate) in tied_san]
        return self.change.choose(board, tied, rng) if tied else move


@dataclass
class Duel:
    """A against B: the changed player against the unchanged one."""
    change: Change
    games: int = 0
    changed_won: int = 0
    unchanged_won: int = 0
    drawn: int = 0

    @property
    def decided(self) -> int:
        return self.changed_won + self.unchanged_won

    def describe(self) -> str:
        return (f"{self.change.describe()}: из {self.games} партий "
                f"{self.changed_won} за изменённого, {self.unchanged_won} "
                f"за прежнего, {self.drawn} ничьих")


#: How many recorded positions are examined when measuring reach. Enough
#: for a share to mean something, small enough that the check stays free
#: compared with playing even one game.
POSITIONS_FOR_REACH = 400


#: Games per experiment. Forty left 27 to 34 decided once draws were
#: dropped, which is under the threshold -- measured, then raised.
GAMES_PER_EXPERIMENT = 70


def ties_in(games: Sequence[Dict[str, Any]], player: Any,
            limit: int = POSITIONS_FOR_REACH,
            every: int = 7) -> List[Tuple[Any, List[str]]]:
    """Positions from the record where the search rated several moves equal.

    Taken from games already played rather than generated: the question is
    what a lever would do in the positions this player actually reaches,
    and a sampled-from-nowhere position answers a different one.
    """
    import chess

    out: List[Tuple[Any, List[str]]] = []
    for game in games:
        board = chess.Board()
        for index, uci in enumerate(game.get("moves", [])):
            if len(out) >= limit:
                return out
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            if index % every == 0 and not board.is_game_over():
                player.choose(board, random.Random(1))
                scores = getattr(player, "_root_scores", None) or []
                if scores:
                    top = max(score for _, score in scores)
                    tied = [san for san, score in scores if score == top]
                    if len(tied) > 1:
                        out.append((board.copy(), tied))
            board.push(move)
    return out


def reach(change: Change, ties: Sequence[Tuple[Any, List[str]]]
          ) -> Dict[str, Any]:
    """How often this lever could change the move at all.

    Zero means the experiment is decided before it starts: the changed
    player and the unchanged one would play identically, and the result
    would be a report about the shuffle. Costs no games.
    """
    if change.property not in SCORERS:
        return {"share": 0.0, "ties": len(ties), "varies": 0}
    score = SCORERS[change.property]
    varies = 0
    for board, tied_san in ties:
        moves = [move for move in board.legal_moves
                 if board.san(move) in tied_san]
        if len({score(board, move) for move in moves}) > 1:
            varies += 1
    return {"share": round(varies / len(ties), 3) if ties else 0.0,
            "ties": len(ties), "varies": varies}


def disagreement(first: Change, second: Change,
                 ties: Sequence[Tuple[Any, List[str]]]) -> float:
    """How often two levers would pick different moves.

    Two that agree everywhere are one experiment, not two. Computed
    without playing anything, which is what makes it worth asking before
    a budget is committed rather than after.
    """
    if first.property not in SCORERS or second.property not in SCORERS:
        return 0.0
    rng = random.Random(0)
    differed = 0
    for board, tied_san in ties:
        moves = [move for move in board.legal_moves
                 if board.san(move) in tied_san]
        if not moves:
            continue
        if first.choose(board, moves, random.Random(1)) != \
                second.choose(board, moves, random.Random(1)):
            differed += 1
    return round(differed / len(ties), 3) if ties else 0.0


def duel(change: Change, games: int = GAMES_PER_EXPERIMENT, depth: int = 2,
         seed: int = 0, on_game: Optional[Callable[[int, str], None]] = None,
         stop: Any = None,
         control: Optional[Callable[[], Any]] = None) -> Duel:
    """Play the changed player against the unchanged one.

    Colours alternate, so the first move is not what is being measured.
    The two differ in exactly one tie-break rule: same depth, same
    evaluation, same shuffle everywhere else.

    `control` builds the player the candidate is measured against. It
    defaults to the player as written, which is right for the first
    change and wrong for every one after it -- once something is in
    force, the question is whether the candidate beats *that*, not what
    came before it. Passed in rather than looked up, so this module still
    knows nothing about versions.
    """
    import chess

    from .chess_arena import SearchPlayer

    out = Duel(change=change)
    for number in range(games):
        if stop is not None and stop.is_set():
            break
        changed_is_white = number % 2 == 0
        make = control or (lambda: SearchPlayer(depth=depth, trace=True))
        base_white = make()
        base_black = make()
        white = Tuned(base_white, change) if changed_is_white else base_white
        black = base_black if changed_is_white else Tuned(base_black, change)
        rng = random.Random(seed + number)
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 200:
            player = white if board.turn else black
            board.push(player.choose(board, rng))
        out.games += 1
        result = board.result()
        if result == "1/2-1/2" or not board.is_game_over():
            out.drawn += 1
            said = "ничья"
        else:
            white_won = result == "1-0"
            if white_won == changed_is_white:
                out.changed_won += 1
                said = "изменённый"
            else:
                out.unchanged_won += 1
                said = "прежний"
        if on_game is not None:
            on_game(number + 1, said)
    return out


def measure(result: Duel, questions: int = 1) -> Dict[str, Any]:
    from statistics import NormalDist

    from .self_model import wilson_interval

    trials = result.decided
    alpha = ALPHA / max(1, questions)
    if trials:
        z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
        low, high = wilson_interval(result.changed_won, trials, z=z)
    else:
        low, high = 0.0, 1.0
    return ledger_mod.measurement_of(
        trials=trials, interval=(low, high), null=NO_EFFECT,
        effect=round(result.changed_won / trials, 3) if trials else None,
        games=result.games, drawn=result.drawn,
        changed_won=result.changed_won, unchanged_won=result.unchanged_won,
        alpha=round(alpha, 5), questions_asked=questions)


def verdict_for(measurement: Dict[str, Any]) -> str:
    if int(measurement["trials"]) < MIN_PAIRED_TRIALS:
        return NOT_EVALUATED
    low, high = measurement["interval"]
    return ACCEPTED if (low > NO_EFFECT or high < NO_EFFECT) else REJECTED


def record(change: Change, result: Duel, depth: int,
           questions: int = 1, ledger: Optional[Any] = None,
           reached: Optional[Dict[str, Any]] = None,
           control_name: str = "", candidate_name: str = ""
           ) -> ledger_mod.Finding:
    """Write the experiment down, with the finding it came from.

    A causal claim, and the record says so: the approach carries the
    change, the conditions carry the player it was tried on, and
    `from_finding` points back at the correlation that suggested it.
    """
    from ..version import PRODUCT_VERSION

    measurement = measure(result, questions)
    if reached is not None:
        measurement["reach"] = reached["share"]
        measurement["reach_ties"] = reached["ties"]
    verdict = verdict_for(measurement)
    # An inert lever is not a tested claim. Calling it REJECTED says the
    # finding failed, when what failed was the experiment's ability to
    # differ from doing nothing -- and "tested and false" against "never
    # tested" is the distinction the gates in this project exist for.
    if reached is not None and not reached["varies"]:
        verdict = NOT_EVALUATED
    finding = ledger_mod.Finding(
        question=QUESTION,
        approach=change.as_dict(),
        verdict=verdict,
        measurement=measurement,
        conditions={"games": result.games, "search_depth": depth,
                    "opponent": "тот же игрок без изменения",
                    "control": control_name or "v0-base",
                    "candidate": candidate_name or "v0-base+change",
                    "version": PRODUCT_VERSION, "questions_asked": questions},
        note=_note(change, measurement, reached),
        version=PRODUCT_VERSION)
    book = ledger if ledger is not None else ledger_mod.Ledger()
    book.record(finding)
    return finding


def _note(change: Change, measurement: Dict[str, Any],
          reached: Optional[Dict[str, Any]] = None) -> str:
    if reached is not None and not reached["varies"]:
        return (f"{change.describe()}: рычаг не двигает ничего — свойство "
                f"одинаково у всех равных ходов в {reached['ties']} ничьих. "
                f"Эксперимент не о находке, а о жребии.")
    trials = int(measurement["trials"])
    if trials < MIN_PAIRED_TRIALS:
        return (f"{change.describe()}: решённых партий {trials}, нужно "
                f"{MIN_PAIRED_TRIALS}. Не измерено — это не ноль.")
    share = measurement.get("effect") or 0.0
    low, high = measurement["interval"]
    if low > NO_EFFECT:
        said = "изменение выигрывает чаще"
    elif high < NO_EFFECT:
        said = "изменение проигрывает чаще"
    else:
        said = "исход не изменился"
    seen = ("" if reached is None
            else f", рычаг работал в {reached['share']:.0%} ничьих")
    return (f"{change.describe()}: {said} ({share:.0%} из {trials} решённых, "
            f"интервал {low:.0%}…{high:.0%}{seen})")
