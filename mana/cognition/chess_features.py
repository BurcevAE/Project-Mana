"""
mana.cognition.chess_features — what the board said, with nothing added.

Why this exists
----------------
`chess_findings` measures a vocabulary of classes -- capture, check,
no_margin, endgame -- and every one of them was written by a person. That
is not MANA finding a regularity in its own play; it is MANA measuring a
person's guesses about where the regularity would be. The strongest of
them, `capture_moderate`, was mined by hand from three games and then
rejected on a hundred and sixty.

So this module records facts and stops. `legal_moves = 1` is a property of
a position, not a claim that a forced move is good or bad. `material = -3`
is a count. Nothing here decides what any of it means, and no threshold or
band appears anywhere in the file -- a band is already an interpretation,
and the point is to leave that to whatever searches these.

The one thing that must not creep back in
------------------------------------------
A feature named for its supposed consequence. `is_blunder_prone`,
`bad_endgame`, `overextended` -- each of those is a conclusion wearing a
feature's clothes, and measuring one proves nothing except that somebody
already believed it. Every name here says what was counted.

Derived, never stored
----------------------
Nothing is written into the game record. The record holds the moves, the
search's trace and the judge's verdicts, and everything below is computed
from those on read. Two consequences, both wanted: a new observable
applies to every game ever played the moment it is written, without
replaying anything; and a feature cannot drift out of agreement with the
game it describes, because it is recomputed from that game every time.

No engine, and the result is the consequence
---------------------------------------------
Nothing here asks a stronger player what a move was worth. Centipawn loss
is Stockfish's opinion, it exists only where somebody paid for it -- 0 of
23452 opponent moves had one -- and self-knowledge built on it is
knowledge of what Stockfish thinks rather than of what happened.

The result of the game is neither borrowed nor one-sided: a game one side
won is one the other lost, so both players carry it, on every move, over
every game already played.

    материал      счёт фигур, свой для того, кто ходит
    через N       каким стал этот счёт через N своих ходов
    эпизод        чем кончилась партия

The cost is real: a game is sixty moves and one bit, so four hundred games
are four hundred observations rather than twenty-three thousand, and a
feature has to be common and lopsided before anything can be said about
it. The compensation is also real, and only self-play has it: the opponent
is the same player, so anything appearing equally on both sides cannot
correlate with the result at all. A whole class of spurious features dies
without anybody deciding it was spurious.

What it does not do
--------------------
It does not look for anything. Searching these for regularities is a
separate step, and one that needs the split discipline in
`cognition/trials.py` -- a free search over features, thresholds and
horizons is a machine for producing coincidences, and this project has
said so about smaller searches than that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

#: Standard piece values, for counting material. Not a judgement about
#: how to play: it is how the record's own evaluation counts, and a count
#: has to use some scale.
PIECE_VALUE = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}

#: How far ahead the later material is read, in the mover's own moves.
#: Three numbers rather than one because a move that looks even now and
#: ruinous later is invisible to the first, and which distance matters is
#: not something to decide in advance.
HORIZONS = (1, 3, 10)

#: Outcome of the game from MANA's side.
WON, LOST, DRAWN, UNFINISHED = "won", "lost", "drawn", "unfinished"


@dataclass
class Move:
    """One of MANA's moves, and what was true when it made it.

    Every field is a count, a name or a score that the position or the
    record supplies. None of them says whether anything was good.

    Everything is from the point of view of whoever moved. That is what
    makes the two sides interchangeable rather than one being described
    in the other's terms, and a game is two-sided by definition: the
    position I face next is the product of both players' choices.
    """
    game: str
    ply: int
    source: str = ""
    level: int = 0
    #: Which side moved, and whether it was MANA's. A fact about the row,
    #: recorded so an analysis can condition on it -- not an instruction
    #: to treat the two differently.
    mover: str = "white"
    ours: bool = True
    #: Facts about the position before the move.
    legal_moves: int = 0
    material: int = 0
    our_pieces: int = 0
    their_pieces: int = 0
    in_check: bool = False
    #: Facts about the move itself.
    piece: str = ""
    is_capture: bool = False
    captured: str = ""
    is_promotion: bool = False
    gives_check: bool = False
    #: Facts about the search that produced it.
    considered: int = 0
    tied_at_top: int = 0
    margin: float = 0.0
    nodes: int = 0
    depth: int = 0
    #: How the mover's own material stood n of its own moves later, minus
    #: what it is now. A count, available for both sides without an
    #: engine. Absent where the game ended before that horizon -- an
    #: unmeasured horizon is not one at which nothing changed.
    later: Dict[int, Optional[float]] = field(default_factory=dict)
    #: The consequence: how the game ended, from the mover's side.
    outcome: str = UNFINISHED

    def as_dict(self) -> Dict[str, Any]:
        row = {name: getattr(self, name) for name in (
            "game", "ply", "source", "level", "mover", "ours",
            "legal_moves", "material",
            "our_pieces", "their_pieces", "in_check", "piece", "is_capture",
            "captured", "is_promotion", "gives_check", "considered",
            "tied_at_top", "margin", "nodes", "depth", "outcome")}
        row.update({f"material_after_{n}": self.later.get(n) for n in HORIZONS})
        return row


def _material(board: Any, us: bool) -> int:
    import chess

    total = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE.get(piece.piece_type, 0)
        total += value if piece.color == us else -value
    return total


def _counts(board: Any, us: bool) -> tuple:
    ours = sum(1 for piece in board.piece_map().values() if piece.color == us)
    theirs = sum(1 for piece in board.piece_map().values() if piece.color != us)
    return ours, theirs


def _outcome(game: Dict[str, Any]) -> str:
    status = str(game.get("status", ""))
    if status in ("", "started", "created"):
        return UNFINISHED
    winner = str(game.get("winner", "") or "")
    if not winner:
        return DRAWN
    return WON if winner == str(game.get("us", "")) else LOST


def _from(outcome: str, ours: bool) -> str:
    """The result as the mover saw it. A game MANA lost is one the
    opponent won, and describing both rows in MANA's terms would make the
    two sides incomparable in exactly the field an analysis cares most
    about."""
    if ours or outcome in (DRAWN, UNFINISHED):
        return outcome
    return LOST if outcome == WON else WON


def observe(game: Dict[str, Any]) -> List[Move]:
    """Replay one recorded game and report what was true at each of its moves.

    Replayed rather than read off stored fields: the moves are the record,
    and anything derived from them cannot disagree with them. Adding an
    observable therefore applies to every game already played, with no
    replaying of anything on a board that matters.
    """
    import chess

    us = str(game.get("us", "white")) == "white"
    start = str(game.get("initial_fen", "startpos"))
    board = chess.Board() if start in ("", "startpos") else chess.Board(start)
    traced = {int(row.get("ply", 0)): row for row in game.get("thoughts", [])}
    outcome = _outcome(game)

    out: List[Move] = []
    for index, uci in enumerate(game.get("moves", []), start=1):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        mover = board.turn
        piece = board.piece_at(move.from_square)
        taken = board.piece_at(move.to_square)
        ours, theirs = _counts(board, mover)
        trace = traced.get(index, {})
        scores = [score for _, score in trace.get("considered", [])]
        top = max(scores) if scores else 0.0
        row = Move(
            game=str(game.get("game", "")), ply=index,
            source=str(game.get("source", "")), level=int(game.get("level", 0) or 0),
            mover="white" if mover else "black", ours=bool(mover == us),
            legal_moves=board.legal_moves.count(),
            material=_material(board, mover), our_pieces=ours,
            their_pieces=theirs,
            in_check=board.is_check(),
            piece=piece.symbol().upper() if piece else "",
            is_capture=board.is_capture(move),
            captured=taken.symbol().upper() if taken else "",
            is_promotion=move.promotion is not None,
            considered=len(scores),
            tied_at_top=sum(1 for score in scores if score == top),
            margin=float(trace.get("margin", 0.0)),
            nodes=int(trace.get("nodes", 0)), depth=int(trace.get("depth", 0)),
            outcome=_from(outcome, mover == us))
        board.push(move)
        row.gives_check = board.is_check()
        out.append(row)

    _fill_horizons(out)
    return out


def _fill_horizons(moves: Sequence[Move]) -> None:
    """How the mover's own material stood n of its own moves later.

    A count rather than an engine's verdict: it exists for both sides,
    over every game already played, and needs nothing bought. Absent
    rather than zero where the game ended first -- an unmeasured horizon
    is not a horizon at which nothing changed.
    """
    # Per side. With both players in one list, "n moves later" has to mean
    # n moves by the same player, or a horizon would compare a position to
    # one the other side was looking at.
    for side in (True, False):
        mine = [row for row in moves if row.ours is side]
        counts = [row.material for row in mine]
        for index, row in enumerate(mine):
            for n in HORIZONS:
                ahead = index + n
                later = counts[ahead] if ahead < len(counts) else None
                row.later[n] = (None if later is None
                                else float(later - row.material))


def observe_all(games: Sequence[Dict[str, Any]]) -> List[Move]:
    out: List[Move] = []
    for game in games:
        try:
            out.extend(observe(game))
        except Exception:
            continue                    # a damaged game is skipped, not fatal
    return out


def describe(moves: Sequence[Move]) -> str:
    """What was recorded, with the denominator on every count.

    Deliberately not a ranking: sorting observables by anything is the
    first step of deciding which one matters, and that decision is not
    this module's to make.
    """
    if not moves:
        return "наблюдений нет"
    ours = [row for row in moves if row.ours]
    decided = [row for row in moves if row.outcome in (WON, LOST)]
    lines = [f"ходов всего: {len(moves)} (из них MANA {len(ours)}, "
             f"соперника {len(moves) - len(ours)})",
             f"  с известным исходом: {len(decided)}",
             f"партий: {len({row.game for row in moves})}"]
    for name in ("legal_moves", "material", "our_pieces", "considered",
                 "tied_at_top"):
        values = [getattr(row, name) for row in moves]
        lines.append(f"  {name:<13} мин {min(values)}, медиана "
                     f"{sorted(values)[len(values) // 2]}, макс {max(values)}")
    for n in HORIZONS:
        known = [row.later.get(n) for row in moves if row.later.get(n) is not None]
        lines.append(f"  материал через {n:>2} своих ходов: известно "
                     f"{len(known)} из {len(moves)}")
    kinds: Dict[str, int] = {}
    for row in moves:
        kinds[row.outcome] = kinds.get(row.outcome, 0) + 1
    lines.append(f"  исходы партий у ходов: {kinds}")
    return "\n".join(lines)
