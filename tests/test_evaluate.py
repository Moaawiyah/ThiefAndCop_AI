"""Tests for agents/evaluate.py: the symmetric trained-vs-trained metrics."""

from agents.evaluate import evaluate
from agents.qlearning import QTable
from core.config import load_config


def small_config(rows=4, cols=4, max_moves=8):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.max_moves = max_moves
    cfg.validate()
    return cfg


def test_evaluate_without_thief_path_leaves_new_metrics_none(tmp_path):
    cfg = small_config()
    cop_path = str(tmp_path / "q_cop.npy")
    QTable("cop", cfg.num_cells, cfg.allow_diagonal).save(cop_path)
    metrics = evaluate(cfg, cop_path, n=5, seed=1)
    assert metrics["thief_survival_rate"] is None
    assert metrics["self_play_capture_rate"] is None


def test_evaluate_with_both_qtables_returns_float_rates(tmp_path):
    cfg = small_config()
    cop_path = str(tmp_path / "q_cop.npy")
    thief_path = str(tmp_path / "q_thief.npy")
    QTable("cop", cfg.num_cells, cfg.allow_diagonal).save(cop_path)
    QTable("thief", cfg.num_cells, cfg.allow_diagonal).save(thief_path)
    metrics = evaluate(cfg, cop_path, thief_path, n=5, seed=1)
    assert 0.0 <= metrics["thief_survival_rate"] <= 1.0
    assert 0.0 <= metrics["self_play_capture_rate"] <= 1.0


def test_evaluate_missing_thief_qtable_leaves_new_metrics_none(tmp_path):
    cfg = small_config()
    cop_path = str(tmp_path / "q_cop.npy")
    QTable("cop", cfg.num_cells, cfg.allow_diagonal).save(cop_path)
    missing_thief = str(tmp_path / "missing_q_thief.npy")
    metrics = evaluate(cfg, cop_path, missing_thief, n=5, seed=1)
    assert metrics["thief_survival_rate"] is None
    assert metrics["self_play_capture_rate"] is None
