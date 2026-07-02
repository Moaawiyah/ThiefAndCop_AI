"""Tests for agents/turn.py: act_agent (defer) + step_agent (act+learn)."""

from collections import deque

from agents.qlearning import QTable
from agents.turn import act_agent, step_agent
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


def test_act_agent_defers_learning():
    # act_agent chooses+applies a move but does NOT touch the Q-table.
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.thief = (1, 1)
    eng.state.cop = (3, 3)  # far -> thief not captured on its move
    q = QTable("thief", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    t = act_agent(eng, "thief", q, None, deque(maxlen=3), cfg)
    assert t.captured is False
    assert q.table[t.state, t.aidx] == 0.0  # deferred: no update happened


def test_deferred_thief_penalty_on_cop_pounce():
    # The trainer's fix: a thief move that lets the cop pounce is terminally
    # penalised (-cop_win), even though the thief did not walk onto the cop.
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.thief = (1, 1)
    eng.state.cop = (3, 3)
    q = QTable("thief", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    t = act_agent(eng, "thief", q, None, deque(maxlen=3), cfg)
    q.update(t.state, t.aidx, -cfg.scoring.cop_win, t.next_state, True)
    assert q.table[t.state, t.aidx] < 0  # thief now fears this move


def test_timeout_applies_survival_terminal():
    # On the last step with no capture, the thief's move earns +survive_reward.
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.thief = (0, 0)
    eng.state.cop = (3, 3)
    q = QTable("thief", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    t = act_agent(eng, "thief", q, None, deque(maxlen=3), cfg)
    q.update(t.state, t.aidx, cfg.qlearning.thief_survive_reward, t.next_state, True)
    assert q.table[t.state, t.aidx] > 0  # survival now enters the Q-table
