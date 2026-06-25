"""Engine state dataclasses (sub-game result + mutable per-sub-game state).

Split out of :mod:`core.engine` to keep the engine module under the project's
line budget. :class:`SubGameResult` is the scored outcome of one sub-game;
:class:`GameState` is the mutable state shared across a single sub-game.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Position = Tuple[int, int]


def initial_positions(grid, start, rng: random.Random) -> Tuple[Position, Position]:
    """Pick start positions honouring ``start`` config + min_initial_distance."""
    min_d = start.min_initial_distance

    def resolve(spec):
        if isinstance(spec, (list, tuple)):
            return (int(spec[0]), int(spec[1]))
        return None

    def rnd():
        return (rng.randrange(grid.rows), rng.randrange(grid.cols))

    fixed_cop = resolve(start.cop)
    fixed_thief = resolve(start.thief)

    # Try to satisfy the distance constraint with random sampling.
    for _ in range(500):
        cop = fixed_cop or rnd()
        thief = fixed_thief or rnd()
        if cop == thief:
            continue
        if grid.chebyshev(cop, thief) >= min_d:
            return cop, thief
    # Fallback: opposite corners (always max distance, distinct).
    cop = fixed_cop or (0, 0)
    thief = fixed_thief or (grid.rows - 1, grid.cols - 1)
    if cop == thief:
        thief = (grid.rows - 1, grid.cols - 1)
    return cop, thief


@dataclass
class SubGameResult:
    """Outcome of one sub-game."""

    index: int
    winner: str               # "cop" or "thief"
    moves: int
    cop_score: int
    thief_score: int
    capture_pos: Optional[Position] = None
    barriers_placed: int = 0

    def as_dict(self) -> dict:
        return {
            "sub_game": self.index,
            "winner": self.winner,
            "moves": self.moves,
            "cop_score": self.cop_score,
            "thief_score": self.thief_score,
            "capture_pos": list(self.capture_pos) if self.capture_pos else None,
            "barriers_placed": self.barriers_placed,
        }


@dataclass
class GameState:
    """Mutable state shared across a single sub-game."""

    cop: Position
    thief: Position
    move_number: int = 0
    barriers_placed: int = 0
    history: List[dict] = field(default_factory=list)
