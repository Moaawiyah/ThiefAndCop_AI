"""Tests for agents/train_utils.py: legal_mask, apply_action, write_curve, evaluate."""

import os

from agents.qlearning import QTable
from agents.train_utils import (
    apply_action, cop_step_reward, evaluate, idle_or_move, legal_mask,
    thief_step_reward, write_curve,
)
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


def test_legal_mask_barrier_gated_on_thief_visibility():
    # A barrier is only legal while the thief is currently in vision.
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (2, 2)
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    b = q.actions.index("barrier")
    assert not legal_mask(q, eng, eng.state.cop, 2, opp_visible=False)[b]
    assert legal_mask(q, eng, eng.state.cop, 2, opp_visible=True)[b]
    # With the gate relaxed, a blind barrier is legal again.
    assert legal_mask(q, eng, eng.state.cop, 2, opp_visible=False,
                      require_visible=False)[b]


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


def test_idle_or_move_penalises_standing_still_rewards_moving():
    # Keyed on actual movement: a real move is rewarded, standing still penalised.
    assert idle_or_move(moved=True, idle_penalty=0.15) > 0
    assert idle_or_move(moved=False, idle_penalty=0.15) < 0


def test_thief_step_reward_rewards_keeping_open_space():
    # Gaining reachable free cells is rewarded; losing them is penalised.
    gain = thief_step_reward(True, 2, 2, 8, 12, True, False, 0.1, 0.15, 0.1)
    lose = thief_step_reward(True, 2, 2, 12, 8, True, False, 0.1, 0.15, 0.1)
    assert gain > lose
    # Blind branch: moving beats staying (no freeze), all else equal.
    move = thief_step_reward(False, 2, 2, 10, 10, True, False, 0.1, 0.15, 0.1)
    idle = thief_step_reward(False, 2, 2, 10, 10, False, False, 0.1, 0.15, 0.1)
    assert move > idle
    # Stepping back into a recent cell costs the soft revisit penalty.
    revis = thief_step_reward(False, 2, 2, 10, 10, True, True, 0.1, 0.15, 0.1)
    assert revis < move


def test_cop_step_reward_penalises_idle_and_revisits():
    # A blind cop that stays is worse than one that moves.
    idle = cop_step_reward(False, 1, 1, 10, 10, "stay", False, False, 0.15, 0.15, 1.0, 0.1)
    move = cop_step_reward(False, 1, 1, 10, 10, "SE", True, False, 0.15, 0.15, 1.0, 0.1)
    assert move > idle
    # Re-entering a recent cell is penalised relative to a fresh move.
    revis = cop_step_reward(False, 1, 1, 10, 10, "SE", True, True, 0.15, 0.15, 1.0, 0.1)
    assert revis < move
    # Visible branch: closing distance yields a positive reward.
    assert cop_step_reward(True, 3, 1, 10, 10, "SE", True, False, 0.15, 0.15, 1.0, 0.1) > 0


def test_cop_barrier_gets_confinement_plus_bonus_else_penalised():
    # A confining barrier earns the confinement credit PLUS the flat bonus.
    reward = cop_step_reward(True, 1, 1, 12, 8, "barrier", False, False, 0.4, 0.15, 1.0, 0.1)
    assert reward == 0.4 * (12 - 8) + 1.0
    # The bonus makes it strictly better than the bare confinement credit alone.
    assert reward > 0.4 * (12 - 8)
    # A wasted barrier (shrinks nothing) is penalised like idling, not rewarded.
    assert cop_step_reward(True, 1, 1, 10, 10, "barrier", False, False, 0.4, 0.15, 1.0, 0.1) < 0
