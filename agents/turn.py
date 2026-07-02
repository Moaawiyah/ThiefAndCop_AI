"""One self-play turn: legal mask + shaping, split into act vs. learn.

Extracted from :mod:`agents.train`. :func:`act_agent` chooses+applies an action
and returns its transition *without* learning, so the trainer can apply the
Bellman update with the correct terminal flag once the opponent's response (and
thus the episode outcome) is known — crucial so the thief is penalised when the
cop pounces on it, not only when it walks onto the cop. :func:`step_agent`
(act + immediate learn) is kept for callers/tests where the outcome is known now.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from core.engine import GameEngine
from agents.qlearning import QTable
from agents.train_utils import (
    apply_action, cop_step_reward, legal_mask, thief_step_reward,
)


class Transition(NamedTuple):
    """One agent move; ``reward``/``captured`` are ±cop_win terminal on a capture."""
    belief: Optional[int]
    captured: bool
    reward: float
    state: int
    aidx: int
    next_state: int


def act_agent(engine: GameEngine, role: str, qtable: QTable,
              belief: Optional[int], history, cfg) -> Transition:
    """Choose+apply one turn for ``role`` and return its transition (no learning).

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
        return Transition(belief, True, reward, state, aidx, next_state)

    dist_after = grid.chebyshev(new_pos, opp_pos)
    free_after = grid.reachable_free_count(s.thief)
    ql = cfg.qlearning
    if role == "cop":
        reward = cop_step_reward(visible, dist_before, dist_after, free_before,
                                 free_after, action, moved, revisited, ql.confinement_weight,
                                 ql.idle_penalty, ql.barrier_bonus, ql.revisit_penalty,
                                 dist_coef=ql.cop_dist_coef, search_bonus=ql.search_move_bonus)
    else:
        # Blind directional flee: extra Chebyshev distance opened from the cop's
        # last-known cell (belief); None until the thief has ever seen the cop.
        blind_flee_delta = None
        if belief is not None:
            bpos = grid.index_to_cell(belief)
            blind_flee_delta = grid.chebyshev(new_pos, bpos) - grid.chebyshev(self_pos, bpos)
        mob = len(grid.legal_moves(new_pos))  # open escape routes (incl. stay)
        reward = thief_step_reward(visible, dist_before, dist_after, free_before,
                                   free_after, moved, revisited, ql.thief_freedom_weight,
                                   ql.idle_penalty, ql.revisit_penalty,
                                   dist_coef=ql.thief_dist_coef,
                                   survive_bonus=ql.survive_bonus,
                                   search_bonus=ql.search_move_bonus,
                                   blind_flee_coef=ql.thief_blind_flee_coef,
                                   blind_flee_delta=blind_flee_delta,
                                   mobility=mob, mobility_coef=ql.thief_mobility_coef,
                                   corner=mob <= 4, corner_penalty=ql.thief_corner_penalty)
    return Transition(belief, False, reward, state, aidx, next_state)


def step_agent(engine: GameEngine, role: str, qtable: QTable,
               belief: Optional[int], history, cfg) -> tuple:
    """Act + immediately learn (terminal iff this move captured).

    Returns ``(belief, captured, reward)``. Used where the outcome is known at
    act time; the trainer uses :func:`act_agent` directly so it can defer the
    thief's update until the cop has responded.
    """
    t = act_agent(engine, role, qtable, belief, history, cfg)
    qtable.update(t.state, t.aidx, t.reward, t.next_state, t.captured)
    return t.belief, t.captured, t.reward
