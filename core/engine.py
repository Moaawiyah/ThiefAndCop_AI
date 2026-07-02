"""Game engine: turn loop, win detection, sub-game and series scoring.

The engine is policy-agnostic: callers supply policy callables. It owns the
authoritative board state and delegates the longer run loop to
``core.engine_runner`` so this module stays focused on state operations.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

from . import engine_runner
from .config import Config
from .grid import Grid
from .observation import Observation, observe
from .state import GameState, SubGameResult, initial_positions
from .policies import (  # noqa: F401  (re-exported for callers/tests)
    Action,
    Policy,
    heuristic_cop_policy,
    heuristic_thief_policy,
    random_policy,
)


class GameEngine:
    """Drives sub-games and the full series for a given config."""

    def __init__(self, config: Config, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random()
        self.grid = Grid(config.rows, config.cols, config.allow_diagonal)
        self.state: Optional[GameState] = None

    def reset_sub_game(self) -> GameState:
        """Start a fresh sub-game."""
        self.grid.clear_barriers()
        cop, thief = initial_positions(self.grid, self.config.start, self.rng)
        self.state = GameState(cop=cop, thief=thief)
        return self.state

    def observation_for(self, agent: str) -> Observation:
        """Return the partial observation for one agent."""
        assert self.state is not None, "No active sub-game"
        s = self.state
        if agent == "cop":
            self_pos, opp = s.cop, s.thief
            barriers_left = self.config.max_barriers - s.barriers_placed
        else:
            self_pos, opp = s.thief, s.cop
            barriers_left = 0
        return observe(
            self.grid, agent, self_pos, opp,
            vision_radius=self.config.vision_radius,
            move_number=s.move_number,
            max_moves=self.config.max_moves,
            barriers_left=barriers_left,
        )

    def apply_thief_action(self, action: Action) -> None:
        """Apply the thief movement action."""
        assert self.state is not None
        act = action.get("action", "stay")
        self.state.thief = self.grid.apply_move(self.state.thief, act)

    def apply_cop_action(self, action: Action) -> None:
        """Apply a cop move or barrier placement."""
        assert self.state is not None
        s = self.state
        if action.get("type") == "barrier":
            if s.barriers_placed < self.config.max_barriers and self.grid.add_barrier(s.cop):
                s.barriers_placed += 1
            return
        act = action.get("action", "stay")
        s.cop = self.grid.apply_move(s.cop, act)

    def is_capture(self) -> bool:
        assert self.state is not None
        return self.state.cop == self.state.thief

    def play_sub_game(
        self,
        cop_policy: Policy,
        thief_policy: Policy,
        index: int = 0,
        on_step: Optional[Callable[["GameEngine"], None]] = None,
    ) -> SubGameResult:
        """Play one full sub-game and return its scored result."""
        return engine_runner.play_sub_game(
            self, cop_policy, thief_policy, index=index, on_step=on_step
        )

    def play_series(
        self,
        cop_policy: Policy,
        thief_policy: Policy,
        on_step: Optional[Callable[["GameEngine"], None]] = None,
        on_sub_game: Optional[Callable[[SubGameResult], None]] = None,
    ) -> Tuple[List[SubGameResult], dict]:
        """Play all configured sub-games; return results and totals."""
        return engine_runner.play_series(
            self, cop_policy, thief_policy, on_step=on_step, on_sub_game=on_sub_game
        )
