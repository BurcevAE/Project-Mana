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

When the levers run out
------------------------
The vocabulary of observations is fixed: thirteen properties, seven of
them computable before a move. Every one of those seven was answered on
v0-base -- five refuted causally, two inert -- and `_pick` returned
nothing. A loop that can only ever ask seven questions stops being a
research loop at question eight.

The space of actions is bigger than the list of properties, because an
ordering can have more than one key. `A then B` prefers what A prefers,
and among the moves A rates equally it prefers what B prefers. It is
neither A nor B, and it is built out of parts MANA already derived from
its own findings -- nothing is invented, only combined.

Composition is second order on purpose: it runs only when the first order
is exhausted, and a composite must earn its games the same way a
primitive does -- it must reach (`reach > 0`) and it must leave a
different set of moves than either part (`narrows > 0`). The second
condition is what stops the generator from selling a part back under a
new name, which is the usual way a generator of this kind lies.

When the declared levers run out too
-------------------------------------
Composition still only reorders what a person wrote into SCORERS. In a
world nobody wrote scorers for, that list is empty and the loop is over
before it starts, however long it observes. `learn` is the one operator
that makes a scorer instead of choosing one: for every action, the share
of the times it was played by the side that went on to win. It needs
nothing but the record -- actions by name, and who won -- and lists no
strategy. It runs only after composition is exhausted, and it is held to
everything a declared lever is: reach first, then the duel, then the
ledger. Being executable earns it nothing.

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
__version__ = "1.1"

QUESTION = "меняет ли исход выбор среди равных ходов по этому свойству"

ALPHA = 0.05
NO_EFFECT = 0.5

#: Prefer more of the property, or less of it.
MORE, LESS = 1, -1

#: The one kind of scorer nobody declared. Its values are not written in
#: this file: they are estimated from what happened, per action, and
#: carried by the change itself.
LEARNED = "learned_outcome"

#: An action seen fewer times than this says nothing about itself. It
#: gets the neutral value, so it can neither win nor lose a tie.
MIN_SEEN = 5
NEUTRAL = 0.5


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
    #: Further keys, applied only where everything before them ties.
    #: Empty for a primitive lever, and everything about a primitive is
    #: byte-identical to what it was before composition existed.
    then: tuple = ()
    #: For a learned scorer: action -> value, estimated from the record.
    #: None for every declared property, which is every change before
    #: the action space could grow.
    table: Optional[Dict[str, float]] = None

    def value(self, name: str, board: Any, move: Any) -> float:
        """One key for one move, declared or learned."""
        if name == LEARNED:
            return float((self.table or {}).get(move.uci(), NEUTRAL))
        return SCORERS[name](board, move)

    def keys(self, board: Any, move: Any) -> tuple:
        """The ordering key for one move: the primary, then the rest.

        A tuple compared left to right, so a later key only speaks where
        every earlier one is equal. That is what makes a composite a new
        lever rather than a louder version of its first part.
        """
        out = [self.value(self.property, board, move) * self.direction]
        for name, way in self.then:
            out.append(self.value(name, board, move) * way)
        return tuple(out)

    def name(self) -> str:
        """A short stable handle, used for caches and player names."""
        body = f"{self.property}{self.direction:+d}"
        for prop, way in self.then:
            body += f"×{prop}{way:+d}"
        return body

    def describe(self) -> str:
        way = "больше" if self.direction == MORE else "меньше"
        said = (f"среди равных ходов выбирать тот, у которого "
                f"«{self.property}» {way}")
        if self.property == LEARNED:
            # Stable on purpose: this sentence is part of the identity,
            # and a refit on a longer record is the same question.
            said = ("среди равных ходов выбирать тот, который в записанных "
                    "партиях чаще играла побеждавшая сторона")
        for prop, direction in self.then:
            way = "больше" if direction == MORE else "меньше"
            said += f", при равенстве — у которого «{prop}» {way}"
        return said

    def as_dict(self) -> Dict[str, Any]:
        return {"property": self.property, "direction": self.direction,
                "from_finding": self.from_finding,
                "also_from": list(self.also_from), "what": self.describe(),
                "then": [[prop, way] for prop, way in self.then],
                "learned_entries": len(self.table or {})}

    def identity(self) -> Dict[str, Any]:
        """What this experiment asks, with nothing that drifts.

        The observational finding that suggested a change is not part of
        the question: its own id contains the number of games it was
        computed on, so including it renamed the experiment every cycle
        and no ledger could say it had been run. It is a condition of the
        answer, and it is recorded as one.
        """
        out = {"property": self.property, "direction": self.direction,
               "what": self.describe()}
        # Only a composite says so. A primitive's identity has to stay
        # byte-identical to the one already written in the ledger, or
        # every refutation earned so far stops matching and the levers
        # answered on v0-base come back as if nothing had been tried.
        if self.then:
            out["then"] = [[prop, way] for prop, way in self.then]
        return out

    def choose(self, board: Any, tied: Sequence[Any],
               rng: random.Random) -> Any:
        """Pick among moves the search rated identically.

        Ties inside the tie are still broken by the shuffle: the change
        is meant to add one criterion, not to install a deterministic
        order that a single unrelated property would then dictate.
        """
        best, chosen = None, []
        for move in tied:
            value = self.keys(board, move)
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
        self.name = f"{getattr(player, 'name', 'player')}+{change.name()}"

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


def _moves(board: Any, tied: Sequence[Any]) -> List[Any]:
    """The tied moves, whether given as SAN or already resolved.

    Resolving SAN costs a legal-move walk per position, and the second
    order asks about the same four hundred positions dozens of times.
    Accepting moves directly lets a caller pay for that once.
    """
    if tied and not isinstance(tied[0], str):
        return list(tied)
    return [move for move in board.legal_moves if board.san(move) in tied]


def resolve(ties: Sequence[Tuple[Any, List[str]]]
            ) -> List[Tuple[Any, List[Any]]]:
    """The same ties with their moves resolved once."""
    return [(board, _moves(board, tied)) for board, tied in ties]


def known(change: Change) -> bool:
    """Every key this change orders by can be computed before the move."""
    def computable(name: str) -> bool:
        return name in SCORERS or (name == LEARNED and bool(change.table))

    return computable(change.property) and all(
        computable(name) for name, _ in change.then)


def table_digest(table: Optional[Dict[str, float]]) -> str:
    import hashlib
    import json

    body = json.dumps(sorted((table or {}).items()), ensure_ascii=False)
    return hashlib.blake2b(body.encode("utf-8"), digest_size=6).hexdigest()


def learn(games: Sequence[Dict[str, Any]],
          min_seen: int = MIN_SEEN) -> Optional[Change]:
    """A new way of acting, made from outcomes rather than declared.

    Every lever before this one scores a move by a property somebody
    wrote into SCORERS. When those run out -- and in an unfamiliar world
    there may be none to begin with -- the loop has nothing left to ask,
    however much it keeps observing. The only thing every world gives is
    a record of actions and who won. So the operator is:

        для каждого действия: как часто его играла сторона,
        которая потом выиграла

    and the change prefers, among the moves the search rated equally, the
    one with the higher share. Nothing about chess is in it -- the key is
    the world's own name for the action -- and no strategy is listed: the
    values come from the record, and whether they are worth anything is
    for the duel to say, exactly as for every other change.

    Laplace-smoothed, and an action seen fewer than `min_seen` times is
    left out, so it stays neutral instead of inheriting one lucky game.
    Draws are skipped: they say nothing about which side's moves won.
    """
    won: Dict[str, int] = {}
    seen: Dict[str, int] = {}
    used = 0
    for game in games:
        winner = str(game.get("winner") or "")
        if winner not in ("white", "black"):
            continue
        fen = str(game.get("initial_fen") or "startpos")
        white_first = fen == "startpos" or fen.split()[1:2] == ["w"]
        used += 1
        for index, uci in enumerate(game.get("moves") or []):
            white_moved = (index % 2 == 0) == white_first
            key = str(uci)
            seen[key] = seen.get(key, 0) + 1
            if (winner == "white") == white_moved:
                won[key] = won.get(key, 0) + 1
    table = {key: round((won.get(key, 0) + 1) / (count + 2), 4)
             for key, count in seen.items() if count >= min_seen}
    if not table:
        return None
    return Change(property=LEARNED, direction=MORE, table=table,
                  from_finding=f"record:{used}:{table_digest(table)}")


def reach(change: Change, ties: Sequence[Tuple[Any, List[str]]]
          ) -> Dict[str, Any]:
    """How often this lever could change the move at all.

    Zero means the experiment is decided before it starts: the changed
    player and the unchanged one would play identically, and the result
    would be a report about the shuffle. Costs no games.
    """
    if not known(change):
        return {"share": 0.0, "ties": len(ties), "varies": 0}
    varies = 0
    for board, tied in ties:
        moves = _moves(board, tied)
        if len({change.keys(board, move) for move in moves}) > 1:
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
    if not known(first) or not known(second):
        return 0.0
    differed = 0
    for board, tied in ties:
        moves = _moves(board, tied)
        if not moves:
            continue
        if first.choose(board, moves, random.Random(1)) != \
                second.choose(board, moves, random.Random(1)):
            differed += 1
    return round(differed / len(ties), 3) if ties else 0.0


def best_set(change: Change, board: Any, moves: Sequence[Any]) -> frozenset:
    """The moves this lever leaves for the shuffle to pick from."""
    best, chosen = None, []
    for move in moves:
        value = change.keys(board, move)
        if best is None or value > best:
            best, chosen = value, [move]
        elif value == best:
            chosen.append(move)
    return frozenset(chosen)


def narrows(made: Change, part: Change,
            ties: Sequence[Tuple[Any, List[str]]]) -> float:
    """How often two levers leave a different set of moves to choose from.

    `disagreement` asks which move each one ends up playing, and that
    answer depends on the shuffle: two levers can leave different sets
    and still be handed the same move by the same seed, which reports a
    real difference as none. For "is this composite anything other than
    its part" the sets are the honest comparison -- they are what the
    lever decides, and the shuffle is what it deliberately leaves alone.
    """
    if not known(made) or not known(part):
        return 0.0
    differed = 0
    for board, tied in ties:
        moves = _moves(board, tied)
        if not moves:
            continue
        if best_set(made, board, moves) != best_set(part, board, moves):
            differed += 1
    return round(differed / len(ties), 3) if ties else 0.0


def compose(changes: Sequence[Change],
            ties: Sequence[Tuple[Any, List[str]]]
            ) -> List[Tuple[Change, Dict[str, Any]]]:
    """New levers built from the ones already derived, when those run out.

    The vocabulary of observations is fixed and small, and every lever it
    yields directly has now been answered on v0-base: five refuted, two
    inert. Either the space of actions ends there, or it is bigger than
    the list of properties -- and it is, because an ordering can have a
    second key. `A then B` prefers what A prefers, and among the moves A
    rates equally it prefers what B prefers.

    Nothing is invented: the parts, their directions and their provenance
    all come from findings MANA already accepted. What is new is only the
    composition.

    Two things are required before a composite counts as a candidate, and
    both are computed from recorded positions without playing anything:

        reach > 0            -- it can move a move at all
        narrows A and B      -- it leaves a different set of moves than
                                either part does, so it is not one of
                                them under a new name, which is the way
                                a generator of this kind usually lies

    An inert part is welcome as a second key: `promotions` never varies
    among tied moves on its own, but among the moves that *another* lever
    rates equally it sometimes does, and that is a different question
    from the one its reach of zero answered.
    """
    ready = resolve(ties)
    out: List[Tuple[Change, Dict[str, Any]]] = []
    parts = [change for change in changes if not change.then and known(change)]
    for first in parts:
        for second in parts:
            if second.property == first.property:
                continue
            made = Change(
                property=first.property, direction=first.direction,
                window=first.window, from_finding=first.from_finding,
                share_seen=first.share_seen,
                also_from=tuple(first.also_from) + (second.from_finding,),
                then=((second.property, second.direction),))
            reached = reach(made, ready)
            if not reached["varies"]:
                continue
            apart = min(narrows(made, first, ready),
                        narrows(made, second, ready))
            if apart <= 0.0:
                continue           # a part wearing a new name, not a lever
            reached = dict(reached, differs=apart)
            out.append((made, reached))
    out.sort(key=lambda row: (-row[1]["differs"], -row[1]["share"],
                              row[0].name()))
    return out


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
        approach=change.identity(),
        verdict=verdict,
        measurement=measurement,
        conditions={"games": result.games, "search_depth": depth,
                    "opponent": "тот же игрок без изменения",
                    "control": control_name or "v0-base",
                    "candidate": candidate_name or "v0-base+change",
                    "from_finding": change.from_finding,
                    "also_from": list(change.also_from),
                    **({"learned_table": table_digest(change.table),
                        "learned_entries": len(change.table)}
                       if change.table else {}),
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
