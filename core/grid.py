"""Grid representation for the Cop & Thief pursuit game.

The grid is a configurable 2D board. Movement is 8-directional (incl. diagonals)
when enabled. The cop may turn cells into impassable *barriers*; barriers block
both agents, just like the edges of the board (assignment §4.3).
"""

from __future__ import annotations

from typing import List, Set, Tuple

Position = Tuple[int, int]

# Action labels. The first eight are movements (8-directional). "stay" keeps the
# current cell. "barrier" is a cop-only special action (handled by the engine).
DIRECTIONS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
    "NE": (-1, 1),
    "NW": (-1, -1),
    "SE": (1, 1),
    "SW": (1, -1),
    "stay": (0, 0),
}

# Canonical, ordered action lists used by the Q-Learning agents.
MOVE_ACTIONS_8 = ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "stay"]
MOVE_ACTIONS_4 = ["N", "S", "E", "W", "stay"]


class Grid:
    """A bounded rectangular board with a set of barrier cells."""

    def __init__(self, rows: int, cols: int, allow_diagonal: bool = True):
        if rows < 1 or cols < 1:
            raise ValueError("Grid dimensions must be >= 1")
        self.rows = rows
        self.cols = cols
        self.allow_diagonal = allow_diagonal
        self.barriers: Set[Position] = set()

    # ----- queries ---------------------------------------------------------
    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_barrier(self, pos: Position) -> bool:
        return pos in self.barriers

    def is_passable(self, pos: Position) -> bool:
        """A cell is passable if it is on the board and not a barrier."""
        return self.in_bounds(pos) and not self.is_barrier(pos)

    def move_actions(self) -> List[str]:
        """Available movement action labels given the diagonal setting."""
        return list(MOVE_ACTIONS_8) if self.allow_diagonal else list(MOVE_ACTIONS_4)

    def apply_move(self, pos: Position, action: str) -> Position:
        """Return the resulting position for ``action`` from ``pos``.

        If the move would leave the board or hit a barrier, the agent stays put
        (the move is treated as a no-op rather than an error).
        """
        if action not in DIRECTIONS:
            raise ValueError(f"Unknown action: {action!r}")
        if not self.allow_diagonal and action in ("NE", "NW", "SE", "SW"):
            # Diagonal disabled: treat as stay.
            return pos
        dr, dc = DIRECTIONS[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if self.is_passable(nxt):
            return nxt
        return pos

    def legal_moves(self, pos: Position) -> List[str]:
        """Movement actions that actually change/keep a valid position."""
        out = []
        for a in self.move_actions():
            if a == "stay":
                out.append(a)
                continue
            dr, dc = DIRECTIONS[a]
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.is_passable(nxt):
                out.append(a)
        return out

    # ----- mutation --------------------------------------------------------
    def add_barrier(self, pos: Position) -> bool:
        """Place a barrier on ``pos``. Returns False if it cannot be placed."""
        if not self.in_bounds(pos) or self.is_barrier(pos):
            return False
        self.barriers.add(pos)
        return True

    def clear_barriers(self) -> None:
        self.barriers.clear()

    def reachable_free_count(self, pos: Position) -> int:
        """Number of passable cells reachable from ``pos`` (incl. ``pos`` itself)
        via legal single steps. This is the thief's "escape freedom": barriers
        that wall it in shrink this count, which the trainer rewards the cop for.
        """
        if not self.is_passable(pos):
            return 0
        seen = {pos}
        stack = [pos]
        while stack:
            cur = stack.pop()
            for a in self.legal_moves(cur):
                if a == "stay":
                    continue
                dr, dc = DIRECTIONS[a]
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return len(seen)

    def chebyshev(self, a: Position, b: Position) -> int:
        """Chebyshev (king-move) distance — natural metric for 8-dir movement."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def manhattan(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def cell_index(self, pos: Position) -> int:
        """Flatten a position to a single integer in [0, rows*cols)."""
        return pos[0] * self.cols + pos[1]

    def index_to_cell(self, idx: int) -> Position:
        return (idx // self.cols, idx % self.cols)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"Grid({self.rows}x{self.cols}, diagonal={self.allow_diagonal}, "
                f"barriers={sorted(self.barriers)})")
