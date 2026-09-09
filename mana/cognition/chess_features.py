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

Horizons
---------
Centipawn loss answers "was this move worse than the best one" and nothing
else. It cannot see a move that is fine now and ruinous in six plies, and
a two-ply search makes exactly that mistake by construction. The judge
already stored an engine score at every one of MANA's moves, so the later
scores are in the record already: `after(n)` is the evaluation n of MANA's
own moves later, and `outcome` is how the game ended. No new judging, no
new games.

    короткий      сколько стоил ход прямо сейчас
    средний       что стало через 2 и 6 своих ходов
    длинный       что стало через 20
    эпизод        чем кончилась партия

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
__version__ = "1.0"

#: Standard piece values, for counting material. Not a judgement about
#: how to play: it is how the record's own evaluation counts, and a count
#: has to use some scale.
PIECE_VALUE = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}

#: How far ahead the later evaluations are read, in MANA's own moves.
#: Three numbers rather than one because a move that is fine now and
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
    """
    game: str
    ply: int
    source: str = ""
    level: int = 0
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
    #: Facts the judge supplied. None where it did not judge.
    score_before: Optional[int] = None
    score_after: Optional[int] = None
    loss: Optional[float] = None
    judged_by: str = ""
    #: Later evaluations, in MANA's own moves: {1: delta, 3: delta, ...}.
    #: Absent where the game ended before that horizon.
    later: Dict[int, Optional[float]] = field(default_factory=dict)
    outcome: str = UNFINISHED

    def as_dict(self) -> Dict[str, Any]:
        row = {name: getattr(self, name) for name in (
            "game", "ply", "source", "level", "legal_moves", "material",
            "our_pieces", "their_pieces", "in_check", "piece", "is_capture",
            "captured", "is_promotion", "gives_check", "considered",
            "tied_at_top", "margin", "nodes", "depth", "score_before",
            "score_after", "loss", "judged_by", "outcome")}
        row.update({f"after_{n}": self.later.get(n) for n in HORIZONS})
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
    judged = {int(row.get("ply", 0)): row for row in game.get("judged", [])}
    traced = {int(row.get("ply", 0)): row for row in game.get("thoughts", [])}
    outcome = _outcome(game)

    out: List[Move] = []
    for index, uci in enumerate(game.get("moves", []), start=1):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if board.turn != us or move not in board.legal_moves:
            if move in board.legal_moves:
                board.push(move)
                continue
            break
        piece = board.piece_at(move.from_square)
        taken = board.piece_at(move.to_square)
        ours, theirs = _counts(board, us)
        verdict = judged.get(index, {})
        trace = traced.get(index, {})
        scores = [score for _, score in trace.get("considered", [])]
        top = max(scores) if scores else 0.0
        row = Move(
            game=str(game.get("game", "")), ply=index,
            source=str(game.get("source", "")), level=int(game.get("level", 0) or 0),
            legal_moves=board.legal_moves.count(),
            material=_material(board, us), our_pieces=ours, their_pieces=theirs,
            in_check=board.is_check(),
            piece=piece.symbol().upper() if piece else "",
            is_capture=board.is_capture(move),
            captured=taken.symbol().upper() if taken else "",
            is_promotion=move.promotion is not None,
            considered=len(scores),
            tied_at_top=sum(1 for score in scores if score == top),
            margin=float(trace.get("margin", 0.0)),
            nodes=int(trace.get("nodes", 0)), depth=int(trace.get("depth", 0)),
            score_before=verdict.get("score_before"),
            score_after=verdict.get("score_after"),
            loss=verdict.get("loss"),
            judged_by=str(verdict.get("judged_by", "")),
            outcome=outcome)
        board.push(move)
        row.gives_check = board.is_check()
        out.append(row)

    _fill_horizons(out)
    return out


def _fill_horizons(moves: Sequence[Move]) -> None:
    """How the evaluation stood n of MANA's own moves later.

    Read from scores the judge already stored, so no position is analysed
    twice and no game is replayed against an engine. Absent rather than
    zero where the game ended first: an unmeasured horizon is not a
    horizon at which nothing changed.
    """
    scores = [row.score_after for row in moves]
    for index, row in enumerate(moves):
        if row.score_after is None:
            row.later = {n: None for n in HORIZONS}
            continue
        for n in HORIZONS:
            ahead = index + n
            later = scores[ahead] if ahead < len(scores) else None
            row.later[n] = (None if later is None
                            else float(later - row.score_after))


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
    judged = [row for row in moves if row.loss is not None]
    lines = [f"ходов MANA: {len(moves)}, из них судимых: {len(judged)}",
             f"партий: {len({row.game for row in moves})}"]
    for name in ("legal_moves", "material", "our_pieces", "considered",
                 "tied_at_top"):
        values = [getattr(row, name) for row in moves]
        lines.append(f"  {name:<13} мин {min(values)}, медиана "
                     f"{sorted(values)[len(values) // 2]}, макс {max(values)}")
    for n in HORIZONS:
        known = [row.later.get(n) for row in moves if row.later.get(n) is not None]
        lines.append(f"  через {n:>2} своих ходов: известно {len(known)} "
                     f"из {len(moves)}")
    kinds: Dict[str, int] = {}
    for row in moves:
        kinds[row.outcome] = kinds.get(row.outcome, 0) + 1
    lines.append(f"  исходы партий у ходов: {kinds}")
    return "\n".join(lines)
