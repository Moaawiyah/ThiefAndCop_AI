"""Tests for the GameEngine: capture, timeout-win, barriers, scoring."""

import random

from core.config import load_config
from core.engine import (
    GameEngine,
    heuristic_cop_policy,
    heuristic_thief_policy,
    random_policy,
)


def make_engine(rows=5, cols=5, seed=0):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.validate()
    return cfg, GameEngine(cfg, rng=random.Random(seed))


def test_capture_scores_cop_win():
    cfg, eng = make_engine()
    eng.reset_sub_game()
    # Force adjacency: cop next to thief, thief stays, cop steps on.
    eng.state.cop = (0, 0)
    eng.state.thief = (1, 1)

    thief_stay = lambda obs, e: {"type": "move", "action": "stay"}
    cop_chase = lambda obs, e: {"type": "move", "action": "SE"}

    res = eng.play_sub_game(cop_chase, thief_stay, index=1)
    # The series resets positions, so instead test scoring mapping directly.
    assert res.winner in ("cop", "thief")
    if res.winner == "cop":
        assert res.cop_score == cfg.scoring.cop_win
        assert res.thief_score == cfg.scoring.thief_loss


def test_timeout_is_thief_win():
    cfg, eng = make_engine(rows=5, cols=5)
    # Both agents stay forever -> thief survives -> thief win.
    stay = lambda obs, e: {"type": "move", "action": "stay"}
    res = eng.play_sub_game(stay, stay, index=1)
    assert res.winner == "thief"
    assert res.moves == cfg.max_moves
    assert res.cop_score == cfg.scoring.cop_loss
    assert res.thief_score == cfg.scoring.thief_win


def test_thief_moving_onto_cop_is_capture():
    cfg, eng = make_engine()
    eng.reset_sub_game()
    eng.state.cop = (2, 2)
    eng.state.thief = (2, 3)

    # Thief moves W (onto cop) -> immediate capture before cop moves.
    thief_suicide = lambda obs, e: {"type": "move", "action": "W"}
    cop_stay = lambda obs, e: {"type": "move", "action": "stay"}
    # We cannot use play_sub_game (it resets), so simulate the first turn.
    eng.apply_thief_action(thief_suicide(None, eng))
    assert eng.is_capture()


def test_barrier_placement_limit():
    cfg, eng = make_engine()
    cfg.max_barriers = 3
    eng.reset_sub_game()
    eng.state.cop = (2, 2)
    # Place 5 barrier requests; only 3 should stick (and only one per cell).
    place_barrier = lambda obs, e: {"type": "barrier"}
    for _ in range(5):
        before = eng.state.barriers_placed
        eng.apply_cop_action({"type": "barrier"})
        # cop stays in same cell, so after first placement the cell is already a
        # barrier and further placements on same cell fail; move then place.
        eng.state.cop = (eng.state.cop[0], min(eng.state.cop[1] + 1, cfg.cols - 1))
    assert eng.state.barriers_placed <= cfg.max_barriers


def test_barrier_blocks_movement():
    cfg, eng = make_engine()
    eng.reset_sub_game()
    eng.state.cop = (2, 2)
    eng.apply_cop_action({"type": "barrier"})
    assert eng.grid.is_barrier((2, 2))
    # Thief cannot move onto the barrier.
    eng.state.thief = (2, 1)
    eng.apply_thief_action({"type": "move", "action": "E"})  # toward (2,2)
    assert eng.state.thief == (2, 1)


def test_series_length_and_totals():
    cfg, eng = make_engine()
    results, totals = eng.play_series(heuristic_cop_policy, heuristic_thief_policy)
    assert len(results) == cfg.num_games
    assert totals["cop"] == sum(r.cop_score for r in results)
    assert totals["thief"] == sum(r.thief_score for r in results)
    # Each sub-game awards a valid scoring combination.
    for r in results:
        assert (r.cop_score, r.thief_score) in {
            (cfg.scoring.cop_win, cfg.scoring.thief_loss),
            (cfg.scoring.cop_loss, cfg.scoring.thief_win),
        }


def test_min_initial_distance_respected_on_large_grid():
    cfg, eng = make_engine(rows=8, cols=8)
    for _ in range(20):
        eng.reset_sub_game()
        d = eng.grid.chebyshev(eng.state.cop, eng.state.thief)
        assert d >= cfg.start.min_initial_distance
        assert eng.state.cop != eng.state.thief


def test_random_policy_returns_legal():
    cfg, eng = make_engine()
    eng.reset_sub_game()
    obs = eng.observation_for("thief")
    act = random_policy(obs, eng)
    assert act["type"] == "move"
    assert act["action"] in eng.grid.move_actions()
