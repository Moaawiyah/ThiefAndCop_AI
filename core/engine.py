"""Game engine: turn loop, win detection, sub-game & game-series scoring.

Terminology (assignment §4.1):
  * **sub-game** (משחקון): a single pursuit round, capped at ``max_moves``.
  * **game**      (משחק):  a series of ``num_games`` sub-games.

Turn order: the **thief moves first**, then the cop (§4.1). The cop wins by
landing exactly on the thief's cell (§4.3). The thief wins by surviving the move
cap uncaught. The cop may, instead of moving, place a barrier on its current
cell (up to ``max_barriers`` per sub-game); the thief cannot.

The engine is deliberately *policy-agnostic*: callers supply two policy callables
(or use the built-in random/heuristic helpers). This lets the same engine back
the headless sanity checks, the MCP orchestrator, and Q-Learning self-play.

The dataclasses live in :mod:`core.state` and the reference policies in
:mod:`core.policies`; both are re-exported here for backward compatibility.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

from .config import Config
from .grid import Grid
from .observation import Observation, observe
from .state import GameState, Position, SubGameResult, initial_positions
from .policies import (  # noqa: F401  (re-exported for callers/tests)
    Action,
    Policy,
    heuristic_cop_policy,
    heuristic_thief_policy,
    random_policy,
)


class GameEngine:
    """Drives sub-games and the full series for a given :class:`Config`."""

    def __init__(self, config: Config, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random()
        self.grid = Grid(config.rows, config.cols, config.allow_diagonal)
        self.state: Optional[GameState] = None

    # ----- placement -------------------------------------------------------
    def reset_sub_game(self) -> GameState:
        """Start a fresh sub-game (clears barriers, re-places agents)."""
        self.grid.clear_barriers()
        cop, thief = initial_positions(self.grid, self.config.start, self.rng)
        self.state = GameState(cop=cop, thief=thief)
        return self.state

    # ----- observations ----------------------------------------------------
    def observation_for(self, agent: str) -> Observation:
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

    # ----- action application ---------------------------------------------
    def apply_thief_action(self, action: Action) -> None:
        """Thief may only move (no barriers)."""
        assert self.state is not None
        act = action.get("action", "stay")
        self.state.thief = self.grid.apply_move(self.state.thief, act)

    def apply_cop_action(self, action: Action) -> None:
        """Cop may move or place a barrier on its current cell."""
        assert self.state is not None
        s = self.state
        if action.get("type") == "barrier":
            if s.barriers_placed < self.config.max_barriers and self.grid.add_barrier(s.cop):
                s.barriers_placed += 1
            # If barrier illegal/exhausted, the cop simply stays this turn.
            return
        act = action.get("action", "stay")
        s.cop = self.grid.apply_move(s.cop, act)

    def is_capture(self) -> bool:
        assert self.state is not None
        return self.state.cop == self.state.thief

    # ----- sub-game driver -------------------------------------------------
    def play_sub_game(
        self,
        cop_policy: Policy,
        thief_policy: Policy,
        index: int = 0,
        on_step: Optional[Callable[["GameEngine"], None]] = None,
    ) -> SubGameResult:
        """Play one full sub-game and return its scored result.

        ``on_step`` (optional) is invoked after each completed move pair — handy
        for GUI rendering or logging.
        """
        self.reset_sub_game()
        s = self.state
        cfg = self.config
        winner = "thief"          # default: thief survives
        capture_pos: Optional[Position] = None

        while s.move_number < cfg.max_moves:
            s.move_number += 1

            # 1) Thief moves first.
            thief_obs = self.observation_for("thief")
            thief_act = thief_policy(thief_obs, self)
            self.apply_thief_action(thief_act)
            if self.is_capture():
                # Thief walked onto the cop -> capture.
                winner = "cop"
                capture_pos = s.cop
                if on_step:
                    on_step(self)
                break

            # 2) Cop moves (or places a barrier).
            cop_obs = self.observation_for("cop")
            cop_act = cop_policy(cop_obs, self)
            self.apply_cop_action(cop_act)
            if self.is_capture():
                winner = "cop"
                capture_pos = s.cop
                if on_step:
                    on_step(self)
                break

            if on_step:
                on_step(self)

        if winner == "cop":
            cop_score = cfg.scoring.cop_win
            thief_score = cfg.scoring.thief_loss
        else:
            cop_score = cfg.scoring.cop_loss
            thief_score = cfg.scoring.thief_win

        return SubGameResult(
            index=index,
            winner=winner,
            moves=s.move_number,
            cop_score=cop_score,
            thief_score=thief_score,
            capture_pos=capture_pos,
            barriers_placed=s.barriers_placed,
        )

    # ----- full series -----------------------------------------------------
    def play_series(
        self,
        cop_policy: Policy,
        thief_policy: Policy,
        on_step: Optional[Callable[["GameEngine"], None]] = None,
        on_sub_game: Optional[Callable[[SubGameResult], None]] = None,
    ) -> Tuple[List[SubGameResult], dict]:
        """Play ``num_games`` sub-games; return per-sub-game results + totals."""
        results: List[SubGameResult] = []
        for i in range(self.config.num_games):
            res = self.play_sub_game(cop_policy, thief_policy, index=i + 1, on_step=on_step)
            results.append(res)
            if on_sub_game:
                on_sub_game(res)
        totals = {
            "cop": sum(r.cop_score for r in results),
            "thief": sum(r.thief_score for r in results),
        }
        return results, totals
