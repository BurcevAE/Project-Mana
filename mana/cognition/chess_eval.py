"""
mana.cognition.chess_eval — the part that is learned, and how not to fool it.

What is learned and what is not
--------------------------------
`brain_factory.choose_mechanism` splits this task in two, and the split is
not a convenience:

    search      exactly computable given an evaluation   -> algorithmic
    evaluation  features known, needs examples           -> classical_ml

So the alpha-beta search in `chess_arena` stays written; the evaluation
function is fitted from games. A domain whose answer is computable does
not need a model that approximates it.

Two ways to fool yourself, and what is done about each
-------------------------------------------------------
**1. Cross-validation across positions from the same game.** Four
positions sampled from one game share an opening, a pawn structure and a
label. Plain k-fold puts some in the training half and some in the
validation half, and reports a score that mostly measures whether the
model recognised the game. Every split here is grouped by game --
`GroupKFold` on `game_id` -- so a game is wholly in one fold.

**2. More parameters than independent examples.** Piece-square tables are
384 numbers; on a thousand games that memorises. The feature set below is
deliberately about a dozen numbers, all differences between the sides, so
there are roughly a hundred independent games per parameter rather than
three.

Cheap on purpose
----------------
This function runs at every leaf of the search -- tens of thousands of
times per move. Features are bitboard population counts, and a fitted
model evaluates as plain Python arithmetic over stored coefficients, not
a call into sklearn per position. A feature that needed legal-move
generation would be worth more per position and far less per second, and
`brain_factory` is being asked which mechanism is cheapest, not which is
richest.

Scaled to centipawns
--------------------
The fitted output predicts the game result in [0, 1]. The search compares
it against mate scores of +-100000, so it is stretched to a
centipawn-like range: a raw difference of 0.01 must not be lost beside a
forced mate, and mate must still dominate everything.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

#: What the model sees. Every one is a difference (White minus Black), so
#: a position and its mirror score opposite by construction rather than
#: by the model happening to learn it.
BASE12: Tuple[str, ...] = (
    "pawns", "knights", "bishops", "rooks", "queens",
    "bishop_pair", "side_to_move", "castling",
    "doubled_pawns", "isolated_pawns", "passed_pawns",
    "central_pawns",
)

#: Five more, all antisymmetric like the first twelve. A symmetric feature
#: -- a game-phase scalar, say -- would break the mirror property that
#: stops the fit preferring a colour, so there is not one.
EXTENDED: Tuple[str, ...] = BASE12 + (
    "king_shield",       # own pawns beside the king
    "rooks_open_file",   # rooks on files with no pawns at all
    "central_pieces",    # knights and bishops on the sixteen central squares
    "advanced_pawns",    # pawns past the middle of the board
    "king_centrality",   # how central the king is, which flips sign by endgame
)

#: The sets a candidate may be fitted on. An allowlist, so "feature_set"
#: is a condition with a stated range rather than an open invitation.
FEATURE_SETS: Dict[str, Tuple[str, ...]] = {"base12": BASE12,
                                            "extended": EXTENDED}

DEFAULT_FEATURE_SET = "base12"

#: Kept as the module-level name the older callers and tests import. It is
#: the base set, unchanged: the point of parameterising was that the old
#: twelve keep behaving exactly as they did.
FEATURE_NAMES: Tuple[str, ...] = BASE12

#: Predicted result is in [0, 1]; the search works in centipawns beside
#: mate scores of +-100000. This stretches one to the other.
CENTIPAWN_SCALE = 2000.0

#: Folds for the grouped cross-validation. Five is enough to see a score
#: that is not noise without spending five times the fit on a corpus this
#: size.
FOLDS = 5

#: Where a fitted evaluation is stored.
MODEL_DIRNAME = "chess_corpus"
MODEL_FILENAME = "evaluation.json"


def _popcount(bits: int) -> int:
    return bin(bits).count("1")


def features(board: Any, chess: Any,
             feature_set: str = DEFAULT_FEATURE_SET) -> List[float]:
    """Differences, all bitboard counts, in the order the set names them.

    `chess` is passed in rather than imported: the module is only
    reachable through `acquire`, and importing it here would bypass the
    verification that everything downstream rests on.

    The base twelve are computed exactly as before. That is the whole
    point of the parameter: the old set has to keep behaving identically
    under the new module version, or the version and the feature set
    could not be varied one at a time.
    """
    white = board.occupied_co[True]
    black = board.occupied_co[False]

    def diff(mask: int) -> int:
        return _popcount(mask & white) - _popcount(mask & black)

    pawns = board.pawns
    white_pawns = pawns & white
    black_pawns = pawns & black

    doubled = isolated = 0
    passed = 0
    files = chess.BB_FILES
    white_files = [white_pawns & f for f in files]
    black_files = [black_pawns & f for f in files]
    for index in range(8):
        w = _popcount(white_files[index])
        b = _popcount(black_files[index])
        doubled += max(0, w - 1) - max(0, b - 1)

        left = index - 1
        right = index + 1
        w_neighbours = ((_popcount(white_files[left]) if left >= 0 else 0)
                        + (_popcount(white_files[right]) if right < 8 else 0))
        b_neighbours = ((_popcount(black_files[left]) if left >= 0 else 0)
                        + (_popcount(black_files[right]) if right < 8 else 0))
        isolated += (1 if w and not w_neighbours else 0)
        isolated -= (1 if b and not b_neighbours else 0)

        # A crude passed-pawn count: no enemy pawn on this file or the
        # ones beside it. It ignores rank, which a real engine would not
        # -- but this has to be cheap, and it is a feature the fit can
        # weigh or discard rather than a rule anybody is asserting.
        w_blockers = sum(_popcount(black_files[i])
                         for i in (left, index, right) if 0 <= i < 8)
        b_blockers = sum(_popcount(white_files[i])
                         for i in (left, index, right) if 0 <= i < 8)
        passed += (1 if w and not w_blockers else 0)
        passed -= (1 if b and not b_blockers else 0)

    centre = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5

    base = [
        float(diff(pawns)),
        float(diff(board.knights)),
        float(diff(board.bishops)),
        float(diff(board.rooks)),
        float(diff(board.queens)),
        float((1 if _popcount(board.bishops & white) >= 2 else 0)
              - (1 if _popcount(board.bishops & black) >= 2 else 0)),
        1.0 if board.turn else -1.0,
        float(_popcount(board.castling_rights & white)
              - _popcount(board.castling_rights & black)),
        float(doubled),
        float(isolated),
        float(passed),
        float(diff(pawns & centre)),
    ]
    if feature_set == "base12":
        return base
    if feature_set != "extended":
        raise ValueError(f"неизвестный набор признаков {feature_set!r}; "
                         f"объявлены {sorted(FEATURE_SETS)}")
    return base + _extended(board, chess, white, black, pawns, diff)


def _extended(board: Any, chess: Any, white: int, black: int, pawns: int,
              diff: Any) -> List[float]:
    """The five added ones, each a White-minus-Black difference."""
    white_pawns, black_pawns = pawns & white, pawns & black

    def shield(colour_pawns: int, colour: bool) -> int:
        square = board.king(colour)
        if square is None:
            return 0
        return _popcount(chess.BB_KING_ATTACKS[square] & colour_pawns)

    open_files = 0
    for file_bb in chess.BB_FILES:
        if not (pawns & file_bb):
            open_files |= file_bb

    central = 0
    for file_bb in chess.BB_FILES[2:6]:
        central |= file_bb
    central &= (chess.BB_RANK_3 | chess.BB_RANK_4
                | chess.BB_RANK_5 | chess.BB_RANK_6)
    minors = board.knights | board.bishops

    advanced_white = _popcount(white_pawns & (chess.BB_RANK_5 | chess.BB_RANK_6
                                              | chess.BB_RANK_7))
    advanced_black = _popcount(black_pawns & (chess.BB_RANK_4 | chess.BB_RANK_3
                                              | chess.BB_RANK_2))

    def centrality(colour: bool) -> int:
        square = board.king(colour)
        if square is None:
            return 0
        # Manhattan distance from the middle, negated so more central is
        # larger. Files and ranks are 0..7, the middle sits at 3.5.
        file_index, rank_index = chess.square_file(square), chess.square_rank(square)
        return -(abs(2 * file_index - 7) + abs(2 * rank_index - 7)) // 2

    return [
        float(shield(white_pawns, True) - shield(black_pawns, False)),
        float(diff(board.rooks & open_files)),
        float(diff(minors & central)),
        float(advanced_white - advanced_black),
        float(centrality(True) - centrality(False)),
    ]


@dataclass(frozen=True)
class LearnedEval:
    """A fitted evaluation that scores a position with plain arithmetic.

    Carries the standardisation with the coefficients: a model stored
    without the scaling it was fitted under is a model that silently
    produces nonsense on the next run.
    """
    coefficients: Tuple[float, ...]
    intercept: float
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    names: Tuple[str, ...] = FEATURE_NAMES
    #: Which set the vector came from. Stored with the coefficients for
    #: the same reason the scaling is: a model that does not know its own
    #: feature set scores the wrong vector on the next run, silently.
    feature_set: str = DEFAULT_FEATURE_SET
    trained_games: int = 0
    trained_positions: int = 0
    cv_r2: float = 0.0
    cv_folds: int = 0

    def score(self, board: Any, chess: Any) -> float:
        """Centipawn-like, from White's point of view."""
        raw = self.intercept
        values = features(board, chess, self.feature_set)
        for value, mean, scale, coefficient in zip(
                values, self.mean, self.scale, self.coefficients):
            raw += coefficient * ((value - mean) / scale)
        return (raw - 0.5) * CENTIPAWN_SCALE

    def weights(self) -> Dict[str, float]:
        """What it learned, per feature, comparable because standardised.

        Reported because a fitted evaluation that cannot be read is a
        result nobody can argue with -- and the first thing to check is
        whether a pawn ended up worth more than a queen.
        """
        return {name: round(c, 4)
                for name, c in zip(self.names, self.coefficients)}

    def as_dict(self) -> Dict[str, Any]:
        return {"coefficients": list(self.coefficients),
                "intercept": self.intercept, "mean": list(self.mean),
                "scale": list(self.scale), "names": list(self.names),
                "feature_set": self.feature_set,
                "trained_games": self.trained_games,
                "trained_positions": self.trained_positions,
                "cv_r2": self.cv_r2, "cv_folds": self.cv_folds}

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "LearnedEval":
        return cls(coefficients=tuple(float(x) for x in row["coefficients"]),
                   intercept=float(row["intercept"]),
                   mean=tuple(float(x) for x in row["mean"]),
                   scale=tuple(float(x) for x in row["scale"]),
                   names=tuple(row.get("names") or FEATURE_NAMES),
                   feature_set=str(row.get("feature_set")
                                   or DEFAULT_FEATURE_SET),
                   trained_games=int(row.get("trained_games") or 0),
                   trained_positions=int(row.get("trained_positions") or 0),
                   cv_r2=float(row.get("cv_r2") or 0.0),
                   cv_folds=int(row.get("cv_folds") or 0))


class NotEnoughGames(RuntimeError):
    """Fitting was attempted below the factory's own threshold.

    Raised rather than fitted anyway: `MIN_ML_EXAMPLES` counts examples,
    and satisfying it with correlated positions from a few hundred games
    meets the number while missing the reason for it.
    """


def design_matrix(games: Sequence[Any], chess: Any,
                  feature_set: str = DEFAULT_FEATURE_SET
                  ) -> Tuple[List[List[float]], List[float], List[int]]:
    """Features, labels and the game each row came from.

    The groups are returned, not optional. Every split downstream needs
    them, and a caller that has to remember to build them is a caller
    that will forget.
    """
    rows: List[List[float]] = []
    labels: List[float] = []
    groups: List[int] = []
    for index, game in enumerate(games):
        for fen in game.sampled:
            rows.append(features(chess.Board(fen), chess, feature_set))
            labels.append(float(game.score))
            groups.append(index)
    return rows, labels, groups


def fit(games: Sequence[Any], chess: Any,
        min_games: Optional[int] = None,
        feature_set: str = DEFAULT_FEATURE_SET) -> LearnedEval:
    """Fit an evaluation, cross-validated with games kept whole.

    `min_games` defaults to `brain_factory.MIN_ML_EXAMPLES`, counted in
    games rather than positions -- the honest unit, and the one that
    makes the threshold mean what it says.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    from .brain_factory import MIN_ML_EXAMPLES

    floor = MIN_ML_EXAMPLES if min_games is None else int(min_games)
    if len(games) < floor:
        raise NotEnoughGames(
            f"{len(games)} партий против порога {floor}. Позиций хватает, "
            f"но позиции одной партии не независимы: обучение на них "
            f"запомнит партии, а не выучит оценку.")

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"неизвестный набор признаков {feature_set!r}")
    rows, labels, groups = design_matrix(games, chess, feature_set)
    if not rows:
        raise NotEnoughGames("в партиях нет сохранённых позиций")

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=float)
    g = np.asarray(groups, dtype=int)

    # Grouped, so a game is wholly in one fold. Plain k-fold here reports
    # how well the model recognises a game it has already seen half of.
    folds = min(FOLDS, len(set(groups)))
    scores: List[float] = []
    if folds >= 2:
        for train_index, test_index in GroupKFold(n_splits=folds).split(X, y, g):
            scaler = StandardScaler().fit(X[train_index])
            model = Ridge(alpha=1.0).fit(scaler.transform(X[train_index]),
                                         y[train_index])
            scores.append(float(model.score(scaler.transform(X[test_index]),
                                            y[test_index])))

    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
    return LearnedEval(
        coefficients=tuple(float(c) for c in model.coef_),
        intercept=float(model.intercept_),
        mean=tuple(float(m) for m in scaler.mean_),
        scale=tuple(float(s) if s else 1.0 for s in scaler.scale_),
        names=FEATURE_SETS[feature_set], feature_set=feature_set,
        trained_games=len(games), trained_positions=len(rows),
        cv_r2=round(sum(scores) / len(scores), 4) if scores else 0.0,
        cv_folds=len(scores))


# --------------------------------------------------------------------------
# using it, and storing it
# --------------------------------------------------------------------------

def player(learned: LearnedEval, chess: Any, depth: int = 2,
           name: str = "learned") -> Any:
    """A search that uses the fitted evaluation."""
    from .chess_arena import SearchPlayer

    return SearchPlayer(evaluate=lambda board: learned.score(board, chess),
                        depth=depth, name=name)


def model_path() -> Path:
    from ..paths import data_root
    return Path(data_root()) / MODEL_DIRNAME / MODEL_FILENAME


def save(learned: LearnedEval, path: Optional[Path] = None) -> bool:
    target = Path(path) if path else model_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(learned.as_dict(), ensure_ascii=False,
                                     indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def load(path: Optional[Path] = None) -> Optional[LearnedEval]:
    source = Path(path) if path else model_path()
    try:
        return LearnedEval.from_dict(json.loads(
            source.read_text(encoding="utf-8")))
    except Exception:
        return None
