"""One self-play turn: legal mask + shaping + Bellman update.

Extracted from :mod:`agents.train` so the trainer's two near-identical agent
turns collapse to a single :func:`step_agent` call each and the module stays
within the project's per-file line budget.
"""

from __future__ import annotations

from typing import Optional

from core.engine import GameEngine
from agents.qlearning import QTable
from agents.train_utils import (
    apply_action, cop_step_reward, legal_mask, thief_step_reward,
)


def step_agent(engine: GameEngine, role: str, qtable: QTable,
               belief: Optional[int], history, cfg) -> tuple:
    """Play one turn for ``role``; learn from it. Returns (belief, captured, reward).

    ``history`` is a deque of the agent's recent cells; stepping back into one
    incurs a soft revisit penalty (discourages ping-pong loops without banning
    revisits).
    """
    s = engine.state
    grid = engine.grid
    cell = grid.cell_index
    self_pos = s.cop if role == "cop" else s.thief
    opp_pos = s.thief if role == "cop" else s.cop

    visible = grid.chebyshev(self_pos, opp_pos) <= cfg.vision_radius
    if visible:
        belief = cell(opp_pos)
    state = qtable.encode_state(cell(self_pos), belief)

    barriers_left = cfg.max_barriers - s.barriers_placed if role == "cop" else 0
    mask = legal_mask(qtable, engine, self_pos, barriers_left, visible,
                      cfg.qlearning.barrier_requires_visible)
    aidx = qtable.select_action_index(state, mask)
    action = qtable.actions[aidx]

    dist_before = grid.chebyshev(self_pos, opp_pos)
    free_before = grid.reachable_free_count(s.thief)
    apply_action(engine, role, action)
    new_pos = s.cop if role == "cop" else s.thief
    new_cell = cell(new_pos)
    moved = new_pos != self_pos
    revisited = new_cell in history
    history.append(new_cell)

    next_state = qtable.encode_state(new_cell, belief)
    if engine.is_capture():
        reward = cfg.scoring.cop_win * (1 if role == "cop" else -1)
        qtable.update(state, aidx, reward, next_state, True)
        return belief, True, reward

    dist_after = grid.chebyshev(new_pos, opp_pos)
    free_after = grid.reachable_free_count(s.thief)
    ql = cfg.qlearning
    if role == "cop":
        reward = cop_step_reward(visible, dist_before, dist_after, free_before,
                                 free_after, action, moved, revisited, ql.confinement_weight,
                                 ql.idle_penalty, ql.barrier_bonus, ql.revisit_penalty)
    else:
        reward = thief_step_reward(visible, dist_before, dist_after, free_before,
                                   free_after, moved, revisited, ql.thief_freedom_weight,
                                   ql.idle_penalty, ql.revisit_penalty)
    qtable.update(state, aidx, reward, next_state, False)
    return belief, False, reward
