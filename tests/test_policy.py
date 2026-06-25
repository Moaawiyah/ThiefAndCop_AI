"""Tests for agents/policy.py: QPolicy belief tracking + build_policy factory."""

import os

from agents.policy import QPolicy, build_policy
from agents.qlearning import QTable
from core.config import load_config
from core.engine import GameEngine, heuristic_cop_policy, heuristic_thief_policy


def small_config(rows=4, cols=4):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.validate()
    return cfg


def test_qpolicy_without_qtable_file_falls_back_to_heuristic(tmp_path):
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    policy = QPolicy("cop", cfg, q_path=str(tmp_path / "missing.npy"))
    assert policy.loaded is False
    obs = eng.observation_for("cop")
    action = policy(obs, eng)
    assert action["type"] in ("move", "barrier")


def test_qpolicy_loads_existing_table_and_acts_greedily(tmp_path):
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (0, 0)
    eng.state.thief = (3, 3)

    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    # Force the SE action to be the obvious best from state encoding for (0,0)->thief visible.
    path = str(tmp_path / "q_cop.npy")
    q.save(path)

    policy = QPolicy("cop", cfg, q_path=path)
    assert policy.loaded is True
    obs = eng.observation_for("cop")
    action = policy(obs, eng)
    assert action["type"] in ("move", "barrier")


def test_qpolicy_reset_belief_clears_last_known_opponent(tmp_path):
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    policy._last_known_opp = 5
    policy.reset_belief()
    assert policy._last_known_opp is None


def test_qpolicy_updates_belief_when_opponent_visible():
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (0, 0)
    eng.state.thief = (0, 1)  # within vision radius
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    obs = eng.observation_for("cop")
    assert obs.opponent_visible is True
    policy(obs, eng)
    assert policy._last_known_opp == eng.grid.cell_index((0, 1))


def test_qpolicy_barrier_action_returned_when_greedy_picks_barrier(tmp_path):
    cfg = small_config()
    cfg.vision_radius = 0
    cfg.validate()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 1)
    eng.state.thief = (3, 3)  # out of vision -> opponent belief stays None
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    barrier_idx = q.actions.index("barrier")
    obs = eng.observation_for("cop")
    assert obs.opponent_visible is False
    self_cell = eng.grid.cell_index(obs.self_pos)
    state = q.encode_state(self_cell, None)
    q.table[state, barrier_idx] = 1000.0
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    action = policy(obs, eng)
    assert action == {"type": "barrier"}


def test_build_policy_prefers_qtable_when_present(tmp_path):
    cfg = small_config()
    cfg.qlearning.q_dir = str(tmp_path)
    q = QTable("thief", cfg.num_cells, cfg.allow_diagonal)
    q.save(os.path.join(cfg.q_dir_abs(), "q_thief.npy"))
    policy = build_policy("thief", cfg)
    assert isinstance(policy, QPolicy)
    assert policy.loaded is True


def test_build_policy_falls_back_to_heuristic_when_missing(tmp_path):
    cfg = small_config()
    cfg.qlearning.q_dir = str(tmp_path)
    policy = build_policy("cop", cfg)
    assert policy is heuristic_cop_policy


def test_build_policy_prefer_qtable_false_returns_heuristic(tmp_path):
    cfg = small_config()
    cfg.qlearning.q_dir = str(tmp_path)
    q = QTable("thief", cfg.num_cells, cfg.allow_diagonal)
    q.save(os.path.join(cfg.q_dir_abs(), "q_thief.npy"))
    policy = build_policy("thief", cfg, prefer_qtable=False)
    assert policy is heuristic_thief_policy
