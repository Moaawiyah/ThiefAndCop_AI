"""Partial-observation function O(state, agent) (assignment §4.5 / §5.1).

Each agent only perceives the board within a Chebyshev ``vision_radius`` of its
own cell. The opponent's position is revealed only when inside that radius;
otherwise the observation reports ``opponent_visible = False`` and the agent must
reason about the opponent's likely location (this is what the NL dialogue and the
Q-Learning "last-known position" feature are for).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .grid import Grid

Position = Tuple[int, int]


@dataclass
class Observation:
    """What a single agent perceives on a given turn."""

    agent: str                       # "cop" or "thief"
    self_pos: Position
    opponent_visible: bool
    opponent_pos: Optional[Position]  # only set when visible
    visible_barriers: List[Position] = field(default_factory=list)
    vision_radius: int = 2
    grid_size: Tuple[int, int] = (5, 5)
    move_number: int = 0
    max_moves: int = 25
    barriers_left: int = 0           # cop only; thief always 0

    def as_dict(self) -> dict:
        """JSON-serialisable view (used by the MCP get_observation tool)."""
        return {
            "agent": self.agent,
            "self_pos": list(self.self_pos),
            "opponent_visible": self.opponent_visible,
            "opponent_pos": list(self.opponent_pos) if self.opponent_pos else None,
            "visible_barriers": [list(b) for b in self.visible_barriers],
            "vision_radius": self.vision_radius,
            "grid_size": list(self.grid_size),
            "move_number": self.move_number,
            "max_moves": self.max_moves,
            "barriers_left": self.barriers_left,
        }


def observe(
    grid: Grid,
    agent: str,
    self_pos: Position,
    opponent_pos: Position,
    vision_radius: int,
    move_number: int = 0,
    max_moves: int = 25,
    barriers_left: int = 0,
) -> Observation:
    """Compute the partial observation for ``agent``.

    The opponent is visible only if within ``vision_radius`` (Chebyshev). Only
    barriers within the vision window are reported.
    """
    visible = grid.chebyshev(self_pos, opponent_pos) <= vision_radius
    visible_barriers = [
        b for b in grid.barriers
        if grid.chebyshev(self_pos, b) <= vision_radius
    ]
    return Observation(
        agent=agent,
        self_pos=self_pos,
        opponent_visible=visible,
        opponent_pos=opponent_pos if visible else None,
        visible_barriers=visible_barriers,
        vision_radius=vision_radius,
        grid_size=(grid.rows, grid.cols),
        move_number=move_number,
        max_moves=max_moves,
        barriers_left=barriers_left,
    )
