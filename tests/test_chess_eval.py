"""The part that is learned, and the two ways of fooling yourself.

The load-bearing tests are `test_fitting_is_refused_below_the_game_
threshold` and `test_grouped_folds_score_lower_than_ungrouped_ones`. Both
guard the same fact: four positions from one game are not four
observations, and every convenient way of counting them says otherwise.
"""
from __future__ import annotations

import json

import pytest

from mana import acquire
from mana.cognition import chess_arena as arena
from mana.cognition import chess_eval as ev
from mana.cognition.chess_eval import LearnedEval, NotEnoughGames

acquire.ensure_importable()
pytest.importorskip("chess", reason="оракул не приобретён")
pytest.importorskip("sklearn", reason="scikit-learn не установлен")


@pytest.fixture(scope="module")
def chess():
    return arena.oracle()


@pytest.fixture(scope="module")
def games(chess):
    """A small corpus. Small on purpose: these tests are about how the
    counting works, not about producing a strong evaluation."""
    return arena.self_play(arena.SearchPlayer(depth=1, name="d1"), 40,
                           seed=90000, chess=chess)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def test_every_feature_is_named():
    assert len(ev.FEATURE_NAMES) == len(set(ev.FEATURE_NAMES))


def test_the_vector_matches_the_names(chess):
    assert len(ev.features(chess.Board(), chess)) == len(ev.FEATURE_NAMES)


def test_a_position_and_its_mirror_score_opposite(chess):
    """Every feature is a difference, so this holds by construction
    rather than by the model happening to learn it. A feature that broke
    it would let the fit prefer a colour."""
    board = chess.Board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    ours = ev.features(board, chess)
    theirs = ev.features(board.mirror(), chess)
    assert [round(-x, 6) for x in ours] == [round(y, 6) for y in theirs]


def test_material_shows_up_where_it_should(chess):
    # White a queen up.
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    named = dict(zip(ev.FEATURE_NAMES, ev.features(board, chess)))
    assert named["queens"] == 1.0
    assert named["pawns"] == 0.0


def test_features_stay_cheap(chess):
    """This runs at every leaf of the search -- tens of thousands of
    times per move. Measured at 21.7 microseconds against 3.5 for
    material counting; a feature needing legal-move generation would be
    worth more per position and far less per second."""
    import time

    board = chess.Board()
    started = time.perf_counter()
    for _ in range(2000):
        ev.features(board, chess)
    per_position = (time.perf_counter() - started) / 2000
    assert per_position < 200e-6


# --------------------------------------------------------------------------
# counting what is independent
# --------------------------------------------------------------------------

def test_fitting_is_refused_below_the_game_threshold(games, chess):
    """`MIN_ML_EXAMPLES` counts examples. Satisfying it with correlated
    positions from a few hundred games meets the number and misses the
    reason for it, so the floor is counted in games."""
    from mana.cognition.brain_factory import MIN_ML_EXAMPLES

    positions = sum(len(g.sampled) for g in games)
    assert positions > len(games)          # the tempting number is bigger

    with pytest.raises(NotEnoughGames) as raised:
        ev.fit(games, chess, min_games=MIN_ML_EXAMPLES)
    assert "не независимы" in str(raised.value)


def test_the_design_matrix_always_carries_its_groups(games, chess):
    """A caller that has to remember to build them is a caller that will
    forget."""
    rows, labels, groups = ev.design_matrix(games, chess)
    assert len(rows) == len(labels) == len(groups)
    assert len(set(groups)) == len([g for g in games if g.sampled])
    for row, group in zip(rows, groups):
        assert len(row) == len(ev.FEATURE_NAMES)
        assert games[group].score in (0.0, 0.5, 1.0)


def test_grouped_folds_score_lower_than_ungrouped_ones(games, chess):
    """The inflation this guards against, demonstrated rather than
    asserted: split without groups and positions from one game land in
    both halves, so the score partly measures whether the model
    recognised a game it has already seen most of."""
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.preprocessing import StandardScaler

    rows, labels, groups = ev.design_matrix(games, chess)
    X, y, g = (np.asarray(rows), np.asarray(labels), np.asarray(groups))

    def mean_score(splits):
        got = []
        for train, test in splits:
            scaler = StandardScaler().fit(X[train])
            model = Ridge(alpha=1.0).fit(scaler.transform(X[train]), y[train])
            got.append(model.score(scaler.transform(X[test]), y[test]))
        return sum(got) / len(got)

    ungrouped = mean_score(KFold(n_splits=5, shuffle=True,
                                 random_state=0).split(X))
    grouped = mean_score(GroupKFold(n_splits=5).split(X, y, g))
    assert grouped <= ungrouped + 1e-9


def test_the_fit_reports_which_folds_it_used(games, chess):
    learned = ev.fit(games, chess, min_games=5)
    assert learned.cv_folds >= 2
    assert learned.trained_games == len(games)
    assert learned.trained_positions == sum(len(g.sampled) for g in games)


def test_fitting_uses_grouped_folds():
    import inspect

    source = inspect.getsource(ev.fit)
    assert "GroupKFold" in source
    assert "KFold(" not in source.replace("GroupKFold(", "")


# --------------------------------------------------------------------------
# the fitted evaluation
# --------------------------------------------------------------------------

def test_a_fitted_evaluation_can_be_read(games, chess):
    """A result nobody can argue with is not a result. The first thing to
    check is whether a pawn ended up worth more than a queen."""
    learned = ev.fit(games, chess, min_games=5)
    weights = learned.weights()
    assert set(weights) == set(ev.FEATURE_NAMES)
    assert all(isinstance(v, float) for v in weights.values())


def test_the_scaling_travels_with_the_coefficients(games, chess, tmp_path):
    """A model stored without the scaling it was fitted under silently
    produces nonsense on the next run."""
    learned = ev.fit(games, chess, min_games=5)
    path = tmp_path / "evaluation.json"
    assert ev.save(learned, path)
    restored = ev.load(path)
    assert restored is not None
    assert restored.as_dict() == learned.as_dict()

    board = chess.Board()
    assert abs(restored.score(board, chess) - learned.score(board, chess)) < 1e-9


def test_a_missing_model_is_none_not_a_crash(tmp_path):
    assert ev.load(tmp_path / "nothing here.json") is None


def test_the_evaluation_scores_a_queen_up_position_higher(games, chess):
    """Not a claim that the fit is good -- a check that the sign
    convention is White-positive, which everything downstream assumes."""
    learned = ev.fit(games, chess, min_games=5)
    even = learned.score(chess.Board(), chess)
    white_up = learned.score(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"), chess)
    black_up = learned.score(chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1"), chess)
    assert white_up > even > black_up


def test_the_learned_player_is_a_search_over_the_fit(games, chess):
    learned = ev.fit(games, chess, min_games=5)
    player = ev.player(learned, chess, depth=1, name="learned")
    assert isinstance(player, arena.SearchPlayer)
    assert player.name == "learned"
    game = arena.play(player, arena.RandomPlayer(), seed=5, chess=chess)
    assert game.result in ("1-0", "0-1", "1/2-1/2")


def test_scoring_does_not_call_into_sklearn(games, chess):
    """Plain arithmetic over stored coefficients: this is called tens of
    thousands of times per move, and a predict() per position would cost
    more than the search it is guiding."""
    import inspect

    source = inspect.getsource(LearnedEval.score)
    for heavy in ("predict", "numpy", "np.", "sklearn"):
        assert heavy not in source
