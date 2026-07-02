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
        """Boolean mask over the action list (movement legality + barrier rule).

        A barrier is legal only while the thief is *currently* in vision (unless
        the config relaxes that), so the cop walls off space as a deliberate
        response to a seen thief rather than blindly.
        """
        require_visible = self.config.qlearning.barrier_requires_visible
        mask = np.zeros(self.qtable.num_actions, dtype=bool)
        legal_moves = set(engine.grid.legal_moves(obs.self_pos))
        for i, a in enumerate(self.qtable.actions):
            if a == "barrier":
                mask[i] = (
                    self.role == "cop"
                    and obs.barriers_left > 0
                    and not engine.grid.is_barrier(obs.self_pos)
                    and (obs.opponent_visible or not require_visible)
                )
            else:
                mask[i] = a in legal_moves
        if not mask.any():
            mask[:] = True
        return mask

    def _suppress_idle(self, mask: np.ndarray, engine: GameEngine,
                       obs: Observation) -> None:
        """Drop position-preserving actions while blind, keeping >=1 action.

        With the opponent out of vision, any action that leaves the agent on its
        current cell — ``stay`` or a move blocked by a wall/barrier — is
        state-preserving: a greedy agent that picks it idles forever and never
        closes in. Banning them forces real progress; the >=1 fallback keeps the
        mask legal if the agent is genuinely boxed in.

        A ``barrier`` is *not* idle when ``barrier_requires_visible`` is off: it
        walls off an escape corridor during blind search (and is self-limiting —
        the cop can't re-barrier its own cell and runs out of barriers), so it is
        left available in that mode. When the gate requires visibility a blind
        barrier is already illegal, so it is suppressed as before.
        """
        allow_blind_barrier = not self.config.qlearning.barrier_requires_visible
        trial = mask.copy()
        for i, a in enumerate(self.qtable.actions):
            if a == "barrier":
                if not allow_blind_barrier:
                    trial[i] = False
            elif engine.grid.apply_move(obs.self_pos, a) == obs.self_pos:
                trial[i] = False
        if trial.any():
            mask[:] = trial

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
        if not obs.opponent_visible:
            # Opponent out of vision. Occasionally lurk in place (human-like);
            # otherwise never freeze on a state-preserving action.
            if self.qtable.rng.random() < self.config.qlearning.blind_stay_prob:
                return {"type": "move", "action": "stay"}
            self._suppress_idle(mask, engine, obs)
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
