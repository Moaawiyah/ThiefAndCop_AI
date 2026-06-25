"""Tests for agents/train_utils.py: legal_mask, apply_action, write_curve, evaluate."""

import os

from agents.qlearning import QTable
from agents.train_utils import apply_action, evaluate, legal_mask, write_curve
from core.config import load_config
from core.engine import GameEngine


def small_config(rows=4, cols=4, max_moves=8):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.max_moves = max_moves
    cfg.validate()
    return cfg


def test_legal_mask_thief_excludes_illegal_moves_off_board():
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.thief = (0, 0)
    q = QTable("thief", cfg.num_cells, cfg.allow_diagonal)
    mask = legal_mask(q, eng, eng.state.thief, barriers_left=0)
    n_idx = q.actions.index("N")
    assert not mask[n_idx]
    stay_idx = q.actions.index("stay")
    assert mask[stay_idx]


def test_legal_mask_cop_barrier_allowed_when_barriers_left():
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (2, 2)
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    mask = legal_mask(q, eng, eng.state.cop, barriers_left=2)
    assert mask[q.actions.index("barrier")]


def test_apply_action_barrier_and_moves():
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 1)
    apply_action(eng, "cop", "barrier")
    assert eng.grid.is_barrier((1, 1))

    eng.state.thief = (2, 2)
    apply_action(eng, "thief", "E")
    assert eng.state.thief == (2, 3)


def test_write_curve_creates_csv_and_png(tmp_path):
    history = [(i, float(i), float(-i), int(i % 5 == 0), max(0.05, 1.0 - i * 0.01))
               for i in range(20)]
    csv_path, png_path = write_curve(history, str(tmp_path))
    assert os.path.exists(csv_path)
    assert os.path.exists(png_path)
    with open(csv_path) as fh:
        lines = fh.readlines()
    assert lines[0].strip() == "episode,cop_reward,thief_reward,captured,epsilon"
    assert len(lines) == 21


def test_evaluate_with_missing_qtable_returns_none_trained_rate(tmp_path):
    cfg = small_config()
    missing_path = str(tmp_path / "missing_q_cop.npy")
    metrics = evaluate(cfg, missing_path, n=5, seed=1)
    assert metrics["trained_capture_rate"] is None
    assert 0.0 <= metrics["random_capture_rate"] <= 1.0


def test_evaluate_with_saved_qtable_returns_float_rate(tmp_path):
    cfg = small_config()
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    metrics = evaluate(cfg, path, n=5, seed=1)
    assert metrics["trained_capture_rate"] is not None
    assert 0.0 <= metrics["trained_capture_rate"] <= 1.0
