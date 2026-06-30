"""Tabular Q-Learning for the Cop & Thief agents (assignment §8).

State encoding (partial-observation friendly):
    The full joint state ``(self_cell, opponent_cell)`` would be exact but is not
    what an agent perceives under partial observability. We therefore encode the
    *belief* state actually available to a policy:

        index = self_cell * (num_cells + 1) + opp_slot

    where ``opp_slot`` is the opponent's cell index when visible/known, or a
    dedicated "unknown" slot (== num_cells) when the opponent is out of vision.
    The policy feeds the **last-known** opponent cell, so the table generalises
    across turns where the opponent is briefly invisible.

Actions:
    Cop:   8 moves + "stay" + "barrier"   (len = move_actions + 1)
    Thief: 8 moves + "stay"               (len = move_actions)

Update rule — the Bellman equation (assignment §8.2):
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from core.grid import MOVE_ACTIONS_4, MOVE_ACTIONS_8


def action_set(role: str, allow_diagonal: bool) -> List[str]:
    """Ordered action labels for a role. Cop gets an extra 'barrier' action."""
    moves = list(MOVE_ACTIONS_8 if allow_diagonal else MOVE_ACTIONS_4)
    if role == "cop":
        return moves + ["barrier"]
    return moves


class QTable:
    """A numpy-backed Q-table with epsilon-greedy selection and Bellman update."""

    def __init__(
        self,
        role: str,
        num_cells: int,
        allow_diagonal: bool = True,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        rng: Optional[np.random.Generator] = None,
    ):
        self.role = role
        self.num_cells = num_cells
        self.allow_diagonal = allow_diagonal
        self.actions = action_set(role, allow_diagonal)
        self.num_actions = len(self.actions)
        # opp_slot in [0, num_cells]: num_cells == "unknown".
        self.num_states = num_cells * (num_cells + 1)

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.rng = rng or np.random.default_rng()

    # ----- state encoding --------------------------------------------------
    def encode_state(self, self_cell: int, opp_cell: Optional[int]) -> int:
        """Map (self cell, known opponent cell or None) to a row index."""
        opp_slot = self.num_cells if opp_cell is None else opp_cell
        return self_cell * (self.num_cells + 1) + opp_slot

    # ----- action selection -----------------------------------------------
    def select_action_index(self, state: int, legal_mask: Optional[np.ndarray] = None) -> int:
        """Epsilon-greedy choice among (optionally masked) legal actions."""
        if legal_mask is None:
            legal_mask = np.ones(self.num_actions, dtype=bool)
        legal_idx = np.flatnonzero(legal_mask)
        if legal_idx.size == 0:
            legal_idx = np.arange(self.num_actions)

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_idx))
        q = self.table[state].copy()
        q[~legal_mask] = -np.inf
        # Break ties randomly among the best legal actions.
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def greedy_action_index(self, state: int, legal_mask: Optional[np.ndarray] = None) -> int:
        """Pure exploitation (no exploration) — used at inference time."""
        if legal_mask is None:
            legal_mask = np.ones(self.num_actions, dtype=bool)
        q = self.table[state].copy()
        q[~legal_mask] = -np.inf
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    # ----- learning --------------------------------------------------------
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Apply the Bellman update; returns the TD error (for diagnostics)."""
        best_next = 0.0 if done else float(np.max(self.table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.table[state, action]
        self.table[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ----- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.table)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        loaded = np.load(path)
        if loaded.shape != self.table.shape:
            raise ValueError(
                f"Q-table shape mismatch: file {loaded.shape} vs expected "
                f"{self.table.shape}. Grid size / role may differ from training."
            )
        self.table = loaded
        return True
