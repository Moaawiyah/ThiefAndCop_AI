"""Tests for agents/train.py: real, fast self-play training with shrunk episodes."""

import os
import sys

import numpy as np

import agents.train as train_mod
from core.config import load_config


def small_config(tmp_path, rows=4, cols=4):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.max_moves = 10
    cfg.qlearning.episodes = 30
    cfg.qlearning.q_dir = str(tmp_path)
    cfg.validate()
    return cfg


def test_train_returns_qtables_and_history(tmp_path):
    cfg = small_config(tmp_path)
    cop_q, thief_q, history = train_mod.train(cfg, episodes=20, seed=1)
    assert len(history) == 20
    assert cop_q.table.shape[0] == cop_q.num_states
    assert thief_q.table.shape[0] == thief_q.num_states
    # Some Q-values should have moved away from zero after 20 episodes.
    assert np.any(cop_q.table != 0) or np.any(thief_q.table != 0)


def test_train_history_fields_are_well_formed(tmp_path):
    cfg = small_config(tmp_path)
    _, _, history = train_mod.train(cfg, episodes=5, seed=2)
    for ep, cop_r, thief_r, captured, epsilon in history:
        assert isinstance(ep, int)
        assert isinstance(captured, int) and captured in (0, 1)
        assert 0.0 <= epsilon <= 1.0


def test_main_full_pipeline_writes_artifacts(tmp_path, monkeypatch):
    cfg = small_config(tmp_path, rows=3, cols=3)
    cfg.qlearning.episodes = 10
    monkeypatch.setattr(train_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(sys, "argv", ["train.py", "--episodes", "10", "--eval-n", "5"])
    train_mod.main()

    q_dir = cfg.q_dir_abs()
    assert os.path.exists(os.path.join(q_dir, "q_cop.npy"))
    assert os.path.exists(os.path.join(q_dir, "q_thief.npy"))
    assert os.path.exists(os.path.join(q_dir, "learning_curve.csv"))
    assert os.path.exists(os.path.join(q_dir, "learning_curve.png"))


def test_main_eval_only_path_does_not_overwrite_missing_tables(tmp_path, monkeypatch):
    cfg = small_config(tmp_path, rows=3, cols=3)
    monkeypatch.setattr(train_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(sys, "argv", ["train.py", "--eval-only", "--eval-n", "3"])
    train_mod.main()
    q_dir = cfg.q_dir_abs()
    assert not os.path.exists(os.path.join(q_dir, "q_cop.npy"))
