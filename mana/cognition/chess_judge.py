"""
mana.cognition.chess_judge — an outside judge, never the brain.

The distinction this module exists to hold
--------------------------------------------
Stockfish decides nothing here. It is asked, after a move has been made,
how much that move cost -- and that is all it is ever asked. The moment an
engine picks the move, the experiment stops being about MANA and becomes a
measurement of the engine, which is the trap the whole chess programme is
worth running only if it avoids.

    MANA          состояние → свой поиск → ход
    Stockfish     ход → насколько он хуже лучшего

So `SearchPlayer` in `chess_arena.py` stays the player: alpha-beta over an
evaluation MANA owns. This module supplies the signal that makes an error
visible, which the result of a game does not: a game lost in forty moves
says one thing once, and forty judged moves say forty things with a
position attached to each.

Why the loss and not the verdict
---------------------------------
"Won or lost" is the aggregate the user was right to refuse as a first
measure. It moves for reasons that have nothing to do with what MANA
learned -- a weaker opponent, a time control, a lucky swindle -- and it
cannot separate a game thrown away in one move from a game slowly
mishandled. Centipawn loss per move is per-decision, and a class of
failures is something you can only find per decision.

    loss = best_score_before − score_after_my_move

Both from the same engine, at the same depth, from the mover's point of
view. A positive loss means the move gave something up.

When the engine is absent
--------------------------
`available()` says so and every reader degrades to `material_loss`, which
needs nothing outside the project: how much material the side to move can
be shown to lose to the opponent's best reply, by the same shallow search
the player already uses. It catches hung pieces and simple tactics and
nothing else, and it says which of the two judged a position, because a
number whose source cannot be told apart is a number nobody can compare.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Where an acquired engine is kept: beside the user's other state, not in
#: the package. A binary in the package is lost on reinstall and shipped
#: to anyone who clones the repository.
ENGINE_DIRNAME = "tools/stockfish"

#: How deep the judge looks. Fixed rather than tuned: a judge whose depth
#: moves between measurements is two judges, and the comparison between
#: them means nothing.
JUDGE_DEPTH = 12

#: Above this, a move is worth reporting as a mistake at all. A handful of
#: centipawns is engine noise between depths, not a decision anybody made.
MISTAKE = 100

#: And above this it is the kind of error that decides games.
BLUNDER = 300

#: The most one move may be said to have cost. A forced mate converts to
#: ten thousand centipawns, and one of those in a hundred moves would set
#: the mean by itself -- so a mean over moves would be a report about
#: whether a game ended in mate rather than about how the moves were
#: played. The raw scores stay in the record; only the loss is capped,
#: and a thousand is "gave up a queen", which is already the top of the
#: scale anybody reads.
MAX_LOSS = 1000.0

#: Names of the two judges, so a measurement always says which one made it.
BY_ENGINE = "stockfish"
BY_MATERIAL = "material"


def engine_path() -> Optional[Path]:
    """The engine, if this installation has one.

    Looked for where `acquire` puts it, then on PATH. Returns None rather
    than raising: an absent engine is a normal state with a working
    fallback, not an error.
    """
    for root in searched():
        if not root.is_dir():
            continue
        for found in sorted(root.glob("stockfish*.exe")) + sorted(root.glob("stockfish*")):
            if found.is_file():
                return found
    found = shutil.which("stockfish")
    return Path(found) if found else None


def searched() -> List[Path]:
    """Every place the engine is looked for, in order.

    Three of them, because they differ exactly when it matters. In a
    checkout `data_root()` is the working directory, so a run started
    from the repository looked inside the repository and judged three
    real games by material while a working Stockfish sat in
    %LOCALAPPDATA%. `shared_data_root()` was the fix and it honours
    MANA_DATA_DIR, so setting that variable collapses both to one and
    brings the same failure back -- hence `platform_data_root()`, which
    ignores every override.

    Exposed rather than kept private because a refusal that names nothing
    is one nobody can act on, and "Stockfish не найден" was read three
    times today without telling anyone where to look.
    """
    from ..paths import platform_data_root, resolve_data_path, shared_data_root

    roots: List[Path] = []
    for root in (Path(resolve_data_path(ENGINE_DIRNAME)),
                 shared_data_root() / ENGINE_DIRNAME,
                 platform_data_root() / ENGINE_DIRNAME):
        if root not in roots:
            roots.append(root)
    return roots


def available() -> bool:
    return engine_path() is not None


@dataclass
class Judged:
    """One move, and how much it cost according to whoever judged it."""
    ply: int
    fen: str
    move: str
    #: Centipawns given up. Zero means it matched the judge's best.
    loss: float
    best: str = ""
    judged_by: str = BY_ENGINE
    score_before: Optional[int] = None
    score_after: Optional[int] = None

    @property
    def mistake(self) -> bool:
        return self.loss >= MISTAKE

    @property
    def blunder(self) -> bool:
        return self.loss >= BLUNDER

    def describe(self) -> str:
        kind = ("зевок" if self.blunder else "ошибка" if self.mistake else "ок")
        best = f", лучше {self.best}" if self.best and self.loss else ""
        return (f"ход {self.ply}: {self.move} — {kind}, потеря "
                f"{self.loss:.0f}{best} [{self.judged_by}]")

    def as_dict(self) -> Dict[str, Any]:
        return {"ply": self.ply, "fen": self.fen, "move": self.move,
                "loss": round(self.loss, 1), "best": self.best,
                "judged_by": self.judged_by, "mistake": self.mistake,
                "blunder": self.blunder,
                "score_before": self.score_before,
                "score_after": self.score_after}


class Judge:
    """Asks an engine what a move cost. Never asked what to play.

    Opened once and reused: starting a process per position costs more
    than the analysis and would make the judge the expensive part of an
    experiment about something else.
    """

    def __init__(self, depth: int = JUDGE_DEPTH,
                 path: Optional[Path] = None) -> None:
        self.depth = int(depth)
        self.path = path or engine_path()
        self._engine: Any = None

    def __enter__(self) -> "Judge":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def kind(self) -> str:
        return BY_ENGINE if self.path else BY_MATERIAL

    def _open(self) -> Any:
        if self._engine is None:
            import chess.engine

            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    # ---------- judging ----------

    def _score(self, board: Any, mover: bool) -> Tuple[int, str]:
        """The engine's score from the mover's side, and its best move."""
        import chess
        import chess.engine

        info = self._open().analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].pov(mover)
        best = ""
        line = info.get("pv") or []
        if line:
            best = board.san(line[0]) if board.is_legal(line[0]) else line[0].uci()
        # A forced mate is not a number of pawns. Clamped rather than
        # converted, so a mate cannot dominate an average of centipawns
        # and turn one lost game into a whole class of "errors".
        return (score.score(mate_score=10000), best)

    def judge_move(self, board: Any, move: Any, ply: int = 0) -> Judged:
        """How much this move cost, from the position it was made in.

        `board` is the position BEFORE the move. Both scores are taken
        from the mover's point of view at the same depth, so the
        difference is about the move and not about whose turn it is.
        """
        mover = board.turn
        san = board.san(move)
        fen = board.fen()

        if not self.path:
            return self._by_material(board, move, ply, san, fen)

        before, best = self._score(board, mover)
        board.push(move)
        try:
            after, _ = self._score(board, mover)
        finally:
            board.pop()
        return Judged(ply=ply, fen=fen, move=san,
                      loss=min(MAX_LOSS, max(0.0, float(before - after))),
                      best=best, judged_by=BY_ENGINE,
                      score_before=before, score_after=after)

    def _by_material(self, board: Any, move: Any, ply: int,
                     san: str, fen: str) -> Judged:
        """The fallback: what the opponent's best reply wins in material.

        Catches a hung piece and a one-move tactic. It does not catch a
        positional mistake at all, and the record says `material` so
        nobody compares the two scales.
        """
        from .chess_arena import material

        mover = board.turn
        sign = 1.0 if mover else -1.0
        before = sign * material(board)
        board.push(move)
        try:
            worst = None
            for reply in board.legal_moves:
                board.push(reply)
                try:
                    value = sign * material(board)
                finally:
                    board.pop()
                worst = value if worst is None else min(worst, value)
            after = worst if worst is not None else sign * material(board)
        finally:
            board.pop()
        return Judged(ply=ply, fen=fen, move=san,
                      loss=min(MAX_LOSS, max(0.0, float(before - after))),
                      best="", judged_by=BY_MATERIAL)

    def judge_game(self, moves: Sequence[Any], side: Optional[bool] = None,
                   start: Optional[Any] = None) -> List[Judged]:
        """Every move of one side in a played game, judged in order.

        `side` None judges both. Replayed from the start rather than
        judged as it was played, so a judgement can be redone later at a
        different depth without the game being played again.
        """
        import chess

        board = start.copy() if start is not None else chess.Board()
        out: List[Judged] = []
        for ply, move in enumerate(moves, start=1):
            if side is None or board.turn == side:
                out.append(self.judge_move(board, move, ply))
            board.push(move)
        return out


def summarise(judged: Sequence[Judged]) -> Dict[str, Any]:
    """What a run of judged moves says, with the denominator.

    The denominator travels because "нашли 23 ошибки" means nothing
    without how many moves were looked at -- the same rule
    `invariants.summarise` states for turns.
    """
    losses = [row.loss for row in judged]
    return {"moves": len(judged),
            "mistakes": sum(1 for row in judged if row.mistake),
            "blunders": sum(1 for row in judged if row.blunder),
            "mean_loss": round(sum(losses) / len(losses), 1) if losses else 0.0,
            "worst": round(max(losses), 1) if losses else 0.0,
            "judged_by": sorted({row.judged_by for row in judged})}
