"""Tests for agents/turn.py: step_agent."""

from collections import deque

from agents.qlearning import QTable
from agents.turn import step_agent
from core.config import load_config
from core.engine import GameEngine


def small_config(rows=4, cols=4, max_moves=8):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.max_moves = max_moves
    cfg.validate()
    return cfg


def _cop_qtable(cfg):
    return QTable("cop", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)


def test_step_agent_reports_capture_on_collision():
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 1)
    eng.state.thief = (1, 2)  # visible; an E move lands on the thief -> capture
    q = _cop_qtable(cfg)
    q.table[:, q.actions.index("E")] = 1000.0  # E is greedy everywhere
    belief, captured, reward = step_agent(eng, "cop", q, None, deque(maxlen=3), cfg)
    assert captured is True
    assert reward == cfg.scoring.cop_win


def test_step_agent_learns_without_capture():
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (0, 0)
    eng.state.thief = (3, 3)  # far apart -> no capture this turn
    q = _cop_qtable(cfg)
    belief, captured, reward = step_agent(eng, "cop", q, None, deque(maxlen=3), cfg)
    assert captured is False
    assert isinstance(reward, float)
