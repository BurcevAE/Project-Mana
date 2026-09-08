"""
mana.cognition.chess_arena — practice, on an oracle that was proved first.

Why self-play and not a program on the desktop
-----------------------------------------------
Playing through somebody else's chess GUI turns a clean problem into a
computer-vision problem: you end up debugging board recognition instead
of learning anything, and every game costs seconds of screen scraping.

The verified rules engine **is** the environment. It generates legal
moves, declares mate, stalemate, repetition and the fifty-move rule, and
it does so exactly -- which is what `acquire.py` established before a
single game was played here.

The refusal that makes the acquisition real
--------------------------------------------
`play` and `self_play` refuse to run unless `acquire.verify("chess_rules")`
says VERIFIED. Not a warning, a refusal. If practice ran on an unproved
oracle, the verification step would be decoration -- and a subtly wrong
engine produces a corpus that is wrong in a way nothing downstream can
detect.

Positions are not independent, and that matters
------------------------------------------------
A game yields eighty positions that share an opening, a pawn structure
and an outcome. Treating them as eighty observations is how a model comes
to memorise games and report a fine cross-validation score.

So two rules hold here and are enforced rather than hoped for:

  * only a few positions per game are kept, spread across it;
  * the **game** is the independent unit. Anything statistical -- the
    paired trials the acceptance gates want -- counts games, never
    positions.

`MIN_ML_EXAMPLES = 1000` in `brain_factory` counts examples. Reaching it
with correlated positions from thirty games would satisfy the number and
not the reason for it.

The baseline player is a floor, not a fix
------------------------------------------
Material counting with a shallow alpha-beta search is here so there is
something to play against and something to beat. It is the control arm,
and a learned evaluation that cannot beat it has not earned anything.
Naming it plainly matters: it is not a hand-written solution smuggled in
as a starting point.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Centipawn values. The one piece of chess knowledge put in by hand, and
#: it is the control arm the learned evaluation has to beat -- not a
#: solution, a floor.
PIECE_VALUE = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}

#: A game longer than this is not going anywhere useful. The rules engine
#: already declares repetition and the fifty-move rule; this only bounds
#: the pathological case so a batch cannot hang.
MAX_PLIES = 300

#: Positions kept per game, spread evenly across it. Small on purpose:
#: see the module docstring on independence.
POSITIONS_PER_GAME = 4

#: Plies at the very start and very end that are not sampled. Openings
#: are shared by every game and say nothing about the players; the last
#: few plies are decided already and label themselves.
SKIP_OPENING_PLIES = 12
SKIP_ENDING_PLIES = 4

#: Where the corpus accumulates, beside the rest of the user's state.
#:
#: Deliberately not "chess". In a development run `data_root()` is the
#: working directory, so a folder called `chess` there becomes an implicit
#: namespace package and shadows the real library: `import chess` returned
#: the corpus directory. The test suite only escaped it because
#: `acquire.ensure_importable()` happens to put the acquisition path ahead
#: of the working directory -- luck, not protection. A directory MANA
#: writes must never be able to answer an import.
CORPUS_DIRNAME = "chess_corpus"


class Unverified(RuntimeError):
    """Practice was attempted on an oracle that was not proved.

    Raised, not warned about: a corpus generated on a subtly wrong rules
    engine is wrong in a way nothing downstream can detect, so continuing
    is worse than stopping.
    """


def oracle(deep: bool = False) -> Any:
    """The chess module, but only if it has been verified.

    Checked on every entry rather than once at import: the provider can
    be replaced or damaged between runs, and the cheap suite costs
    milliseconds against games that cost seconds.
    """
    from .. import acquire

    checked = acquire.verify("chess_rules", deep=deep)
    if not checked.trusted:
        raise Unverified(checked.describe())
    module = acquire.installed("chess_rules")
    if module is None:                      # verified but vanished
        raise Unverified("правила прошли проверку, но модуль не импортируется")
    return module


# --------------------------------------------------------------------------
# evaluation and players
# --------------------------------------------------------------------------

Evaluate = Callable[[Any], float]


def material(board: Any) -> float:
    """Centipawns from White's point of view. The control arm."""
    total = 0
    for piece_type, value in PIECE_VALUE.items():
        if not value:
            continue
        total += value * len(board.pieces(piece_type, True))
        total -= value * len(board.pieces(piece_type, False))
    return float(total)


#: How a finished game scores, from White's point of view. Used both to
#: label positions and to score a match.
OUTCOME_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


class Player:
    """Something that picks a legal move. Named, so a result can say who."""
    name = "player"

    def choose(self, board: Any, rng: random.Random) -> Any:
        raise NotImplementedError


class RandomPlayer(Player):
    """The floor under the floor. Useful only as a control."""

    def __init__(self, name: str = "random") -> None:
        self.name = name

    def choose(self, board: Any, rng: random.Random) -> Any:
        return rng.choice(list(board.legal_moves))


@dataclass
class Thought:
    """Why one move was chosen, in the search's own terms.

    `margin` is the number that says whether the decision was close. A
    move ahead by two centipawns is a tie the evaluation happened to
    win; one ahead by four hundred is a position where the search saw
    something. Diagnosing a class of errors needs that difference: the
    first kind is noise to fix by evaluating better, the second is a
    judgement to fix by searching differently.
    """
    ply: int
    fen: str
    chosen: str
    score: float
    #: Every root move with its score, best first. The search computed
    #: these anyway; the trace is what it used to throw away.
    considered: List[Tuple[str, float]] = field(default_factory=list)
    depth: int = 0
    nodes: int = 0

    @property
    def margin(self) -> float:
        """How far ahead the chosen move was of the next one."""
        if len(self.considered) < 2:
            return 0.0
        return float(self.considered[0][1] - self.considered[1][1])

    @property
    def close_call(self) -> bool:
        return self.margin < 25.0

    def describe(self, top: int = 4) -> str:
        rivals = ", ".join(f"{san} {score:+.0f}"
                           for san, score in self.considered[1:top + 1])
        head = (f"{self.chosen} {self.score:+.0f} "
                f"(глубина {self.depth}, узлов {self.nodes}, "
                f"отрыв {self.margin:+.0f})")
        return head + (f"; рассматривала: {rivals}" if rivals else "")

    def as_dict(self) -> Dict[str, Any]:
        return {"ply": self.ply, "fen": self.fen, "chosen": self.chosen,
                "score": round(self.score, 1),
                "considered": [[san, round(score, 1)]
                               for san, score in self.considered],
                "depth": self.depth, "nodes": self.nodes,
                "margin": round(self.margin, 1),
                "close_call": self.close_call}


class SearchPlayer(Player):
    """Alpha-beta negamax over a pluggable evaluation.

    The evaluation is the part that will be learned; the search is not.
    `brain_factory.choose_mechanism` says algorithmic for search -- it is
    exactly computable given an evaluation -- and classical_ml for the
    evaluation itself, which is the split this class exists to allow.
    """

    def __init__(self, evaluate: Evaluate = material, depth: int = 2,
                 name: str = "material", max_nodes: int = 0,
                 trace: bool = False) -> None:
        self.evaluate = evaluate
        #: Keep the reasoning behind every move. Off by default: the
        #: scores are computed either way, but a five-hundred-game run
        #: would hold several million rows for nobody.
        self.trace = bool(trace)
        self.thoughts: List["Thought"] = []
        self.depth = max(1, int(depth))
        self.name = name
        #: When set, depth is decided by a node budget instead: iterative
        #: deepening until the budget is spent. Two evaluations compared
        #: at equal depth make the expensive one free, and the feature
        #: set here costs 21.7 microseconds a position against 3.5 for
        #: material counting -- 6.2 times. Equal nodes isolates the
        #: question being asked (does this evaluation guide the search
        #: better, per unit of search) and stays deterministic, which
        #: paired trials need.
        self.max_nodes = max(0, int(max_nodes))
        self.nodes = 0
        self.reached_depth = 0
        #: Root moves and their scores from the last completed search.
        #: Filled only when tracing; the search computes them regardless.
        self._root_scores: List[Tuple[str, float]] = []

    def choose(self, board: Any, rng: random.Random) -> Any:
        before = self.nodes
        if self.max_nodes:
            move = self._choose_within_budget(board, rng)
        else:
            move = self._choose_at_depth(board, rng, self.depth)
        if self.trace:
            self._remember(board, move, before)
        return move

    def _remember(self, board: Any, move: Any, nodes_before: int) -> None:
        """Write down what the search just decided, and how closely.

        Read off `self._root_scores`, which `_choose_at_depth` fills as
        it goes -- so this records what the search actually did rather
        than a second search that might disagree with it.
        """
        scored = sorted(self._root_scores, key=lambda row: -row[1])
        self.thoughts.append(Thought(
            ply=board.ply() + 1, fen=board.fen(),
            chosen=board.san(move),
            score=float(dict(self._root_scores).get(board.san(move), 0.0)),
            considered=scored,
            depth=self.reached_depth or self.depth,
            nodes=self.nodes - nodes_before))

    def _choose_at_depth(self, board: Any, rng: random.Random,
                         depth: int) -> Any:
        moves = list(board.legal_moves)
        rng.shuffle(moves)              # break ties without a preference
        best_move, best_score = moves[0], float("-inf")
        scores: List[Tuple[str, float]] = []
        for move in moves:
            san = board.san(move) if self.trace else ""
            board.push(move)
            score = -self._search(board, depth - 1,
                                  float("-inf"), float("inf"))
            board.pop()
            if self.trace:
                scores.append((san, score))
            if score > best_score:
                best_move, best_score = move, score
        self._root_scores = scores
        return best_move

    def _choose_within_budget(self, board: Any, rng: random.Random) -> Any:
        """Iterative deepening until the node budget is spent.

        The move comes from the last **completed** iteration. Taking the
        best move from a half-searched depth would make the result depend
        on where the budget happened to run out, which is noise dressed
        as a decision.
        """
        seed = rng.random()
        spent_before = self.nodes
        best = None
        for depth in range(1, 64):
            # Same tie-breaking at every depth, so deepening does not
            # shuffle the answer around for reasons unrelated to search.
            best = self._choose_at_depth(board, random.Random(seed), depth)
            self.reached_depth = depth
            if self.nodes - spent_before >= self.max_nodes:
                break
        return best

    def _search(self, board: Any, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if depth <= 0 or board.is_game_over():
            return self._leaf(board)
        best = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            score = -self._search(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def _leaf(self, board: Any) -> float:
        """Score from the side to move's point of view, as negamax wants.

        Mate is scored by depth so a forced mate is preferred sooner and
        avoided later; a flat score makes the search wander in a won
        position instead of finishing.
        """
        if board.is_checkmate():
            return -100000.0 + board.ply()
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        value = self.evaluate(board)
        return value if board.turn else -value


# --------------------------------------------------------------------------
# playing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Game:
    """One finished game, and enough of it to learn from."""
    white: str
    black: str
    result: str                    # "1-0" | "0-1" | "1/2-1/2"
    reason: str                    # checkmate, stalemate, repetition, ...
    plies: int
    elapsed: float
    seed: int
    sampled: Tuple[str, ...] = ()  # FENs kept for learning

    @property
    def score(self) -> float:
        """From White's point of view."""
        return OUTCOME_SCORE.get(self.result, 0.5)

    def as_dict(self) -> Dict[str, Any]:
        return {"white": self.white, "black": self.black,
                "result": self.result, "reason": self.reason,
                "plies": self.plies, "elapsed": round(self.elapsed, 3),
                "seed": self.seed, "sampled": list(self.sampled)}

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Game":
        return cls(white=str(row.get("white", "")),
                   black=str(row.get("black", "")),
                   result=str(row.get("result", "1/2-1/2")),
                   reason=str(row.get("reason", "")),
                   plies=int(row.get("plies") or 0),
                   elapsed=float(row.get("elapsed") or 0.0),
                   seed=int(row.get("seed") or 0),
                   sampled=tuple(row.get("sampled") or ()))


def _sample_indices(plies: int, keep: int) -> List[int]:
    """Which plies to keep, spread across the middle of the game."""
    first = SKIP_OPENING_PLIES
    last = plies - SKIP_ENDING_PLIES
    if last - first < 1 or keep < 1:
        return []
    keep = min(keep, last - first)
    step = (last - first) / float(keep)
    return sorted({int(first + step * i) for i in range(keep)})


def play(white: Player, black: Player, seed: int = 0,
         max_plies: int = MAX_PLIES,
         positions_per_game: int = POSITIONS_PER_GAME,
         chess: Any = None) -> Game:
    """One game to a finish, on the verified oracle.

    Draw claims are asked for: without repetition and the fifty-move
    rule, two shallow searches shuffle pieces until the ply cap and the
    game is recorded as a draw for the wrong reason.
    """
    chess = chess or oracle()
    rng = random.Random(seed)
    board = chess.Board()
    started = time.perf_counter()
    fens: List[str] = []

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        fens.append(board.fen())
        mover = white if board.turn else black
        board.push(mover.choose(board, rng))

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result, reason = "1/2-1/2", "оборвано по лимиту ходов"
    else:
        result = outcome.result()
        reason = getattr(outcome.termination, "name", str(outcome.termination))

    keep = _sample_indices(len(fens), positions_per_game)
    return Game(white=white.name, black=black.name, result=result,
                reason=reason.lower(), plies=board.ply(),
                elapsed=time.perf_counter() - started, seed=seed,
                sampled=tuple(fens[i] for i in keep))


def match(white: Player, black: Player, games: int, seed: int = 0,
          swap_colours: bool = True, chess: Any = None,
          on_game: Optional[Callable[[Game], None]] = None) -> List[Game]:
    """A set of games between two players.

    Colours alternate. White has a real advantage, so a match played from
    one side measures the colour as much as the player -- and the paired
    comparison the gates want needs both arms to have met the same
    positions from both sides.
    """
    chess = chess or oracle()
    played: List[Game] = []
    for index in range(max(0, int(games))):
        first, second = ((white, black) if not (swap_colours and index % 2)
                         else (black, white))
        game = play(first, second, seed=seed + index, chess=chess)
        played.append(game)
        if on_game is not None:
            try:
                on_game(game)
            except Exception:
                pass
    return played


def self_play(player: Player, games: int, seed: int = 0,
              chess: Any = None,
              on_game: Optional[Callable[[Game], None]] = None) -> List[Game]:
    """A player against itself, to produce a corpus."""
    return match(player, player, games, seed=seed, swap_colours=False,
                 chess=chess, on_game=on_game)


# --------------------------------------------------------------------------
# what comes out
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Labelled:
    """One position, with the outcome of the game it came from.

    `game_id` travels with it so anything statistical can group by game.
    Losing that is how correlated positions get counted as independent
    observations.
    """
    fen: str
    label: float                   # 1.0 white won, 0.5 draw, 0.0 black won
    game_id: int

    def as_dict(self) -> Dict[str, Any]:
        return {"fen": self.fen, "label": self.label, "game_id": self.game_id}


def labelled(games: Sequence[Game]) -> List[Labelled]:
    """Sampled positions from these games, each carrying its game."""
    out: List[Labelled] = []
    for index, game in enumerate(games):
        for fen in game.sampled:
            out.append(Labelled(fen, game.score, index))
    return out


def score_of(games: Sequence[Game], player: str) -> Dict[str, Any]:
    """How a named player did, counting games and never positions."""
    wins = draws = losses = 0
    itself = 0
    for game in games:
        if game.white == player and game.black == player:
            # A player against itself measures nothing. Counted and
            # reported rather than folded in as a draw, which would drag
            # every score towards 0.5 and quietly flatter a weak player.
            itself += 1
            continue
        if player == game.white:
            got = game.score
        elif player == game.black:
            got = 1.0 - game.score
        else:
            continue
        wins += got == 1.0
        draws += got == 0.5
        losses += got == 0.0
    played = wins + draws + losses
    return {"player": player, "games": played, "wins": wins, "draws": draws,
            "losses": losses, "self_play_skipped": itself,
            "score": round((wins + 0.5 * draws) / played, 4) if played else 0.0}


# --------------------------------------------------------------------------
# the corpus on disk
# --------------------------------------------------------------------------

def corpus_path() -> Path:
    from ..paths import data_root
    return Path(data_root()) / CORPUS_DIRNAME / "games.jsonl"


class Corpus:
    """Games accumulated across runs, appended and read back.

    Deliberately plain. This is training material, not state anything
    depends on being present, so a missing or damaged file costs games
    and never a crash.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else corpus_path()

    def append(self, games: Sequence[Game]) -> int:
        if not games:
            return 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                for game in games:
                    handle.write(json.dumps(game.as_dict(),
                                            ensure_ascii=False) + "\n")
            return len(games)
        except Exception:
            return 0

    def games(self, limit: int = 0) -> List[Game]:
        rows: List[Game] = []
        try:
            handle = self.path.open("r", encoding="utf-8")
        except Exception:
            return rows
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(Game.from_dict(json.loads(line)))
                except Exception:
                    continue
        return rows[-limit:] if limit and limit > 0 else rows

    def stats(self) -> Dict[str, Any]:
        games = self.games()
        positions = sum(len(g.sampled) for g in games)
        by_result: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        for game in games:
            by_result[game.result] = by_result.get(game.result, 0) + 1
            by_reason[game.reason] = by_reason.get(game.reason, 0) + 1
        return {
            "path": str(self.path), "exists": self.path.exists(),
            "games": len(games),
            "positions": positions,
            # Stated together on purpose: positions from one game share an
            # opening and an outcome, so the second number is not a count
            # of independent examples and must never be read as one.
            "independent_units": len(games),
            "by_result": by_result,
            "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
            "plies_total": sum(g.plies for g in games),
            "seconds_total": round(sum(g.elapsed for g in games), 2),
        }
