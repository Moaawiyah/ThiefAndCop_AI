"""Policy wrappers that turn an Observation into an Action.

Two concrete policies are provided:

* :class:`QPolicy` — loads a trained Q-table and selects greedily, tracking the
  last-known opponent cell so the table can be queried even when the opponent is
  out of vision (partial observability).
* The heuristic policies in :mod:`core.engine` remain the dependency-free
  fallback used when no Q-table has been trained yet.

A small factory, :func:`build_policy`, returns a callable with the
``(observation, engine) -> action`` signature expected by the engine and the
orchestrator. It degrades gracefully to the heuristic if the ``.npy`` file is
missing.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from core.config import Config
from core.engine import GameEngine, heuristic_cop_policy, heuristic_thief_policy
from core.observation import Observation
from .qlearning import QTable


class QPolicy:
    """Greedy Q-table policy with last-known-opponent belief tracking."""

    def __init__(self, role: str, config: Config, q_path: Optional[str] = None):
        self.role = role
        self.config = config
        self.qtable = QTable(
            role=role,
            num_cells=config.num_cells,
            allow_diagonal=config.allow_diagonal,
            learning_rate=config.qlearning.learning_rate,
            discount_factor=config.qlearning.discount_factor,
            epsilon=0.0,            # inference: no exploration
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )
        self.loaded = False
        if q_path and os.path.exists(q_path):
            self.loaded = self.qtable.load(q_path)
        self._last_known_opp: Optional[int] = None

    def reset_belief(self) -> None:
        self._last_known_opp = None

    def _legal_mask(self, engine: GameEngine, obs: Observation) -> np.ndarray:
        """Boolean mask over the action list (movement legality + barrier rule)."""
        mask = np.zeros(self.qtable.num_actions, dtype=bool)
        legal_moves = set(engine.grid.legal_moves(obs.self_pos))
        for i, a in enumerate(self.qtable.actions):
            if a == "barrier":
                mask[i] = (
                    self.role == "cop"
                    and obs.barriers_left > 0
                    and not engine.grid.is_barrier(obs.self_pos)
                )
            else:
                mask[i] = a in legal_moves
        if not mask.any():
            mask[:] = True
        return mask

    def __call__(self, obs: Observation, engine: GameEngine) -> dict:
        if not self.loaded:
            # No trained table: fall back to a sensible heuristic.
            fb = heuristic_cop_policy if self.role == "cop" else heuristic_thief_policy
            return fb(obs, engine)

        self_cell = engine.grid.cell_index(obs.self_pos)
        # Update belief from current observation.
        if obs.opponent_visible and obs.opponent_pos is not None:
            self._last_known_opp = engine.grid.cell_index(obs.opponent_pos)
        opp_cell = self._last_known_opp  # may be None -> "unknown" slot

        state = self.qtable.encode_state(self_cell, opp_cell)
        mask = self._legal_mask(engine, obs)
        a_idx = self.qtable.greedy_action_index(state, mask)
        action = self.qtable.actions[a_idx]
        if action == "barrier":
            return {"type": "barrier"}
        return {"type": "move", "action": action}


def build_policy(
    role: str,
    config: Config,
    prefer_qtable: bool = True,
) -> Callable[[Observation, GameEngine], dict]:
    """Return a policy callable for ``role``.

    Uses a trained Q-table when available (and ``prefer_qtable``), otherwise the
    heuristic. The returned callable matches the engine's policy signature.
    """
    if prefer_qtable:
        fname = "q_cop.npy" if role == "cop" else "q_thief.npy"
        q_path = os.path.join(config.q_dir_abs(), fname)
        policy = QPolicy(role, config, q_path=q_path)
        if policy.loaded:
            return policy
    return heuristic_cop_policy if role == "cop" else heuristic_thief_policy
