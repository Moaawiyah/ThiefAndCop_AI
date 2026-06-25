"""Tests for the Grid: bounds, barriers, 8-directional movement."""

from core.grid import Grid, MOVE_ACTIONS_4, MOVE_ACTIONS_8


def test_bounds():
    g = Grid(5, 5)
    assert g.in_bounds((0, 0))
    assert g.in_bounds((4, 4))
    assert not g.in_bounds((-1, 0))
    assert not g.in_bounds((5, 0))
    assert not g.in_bounds((0, 5))


def test_move_within_bounds():
    g = Grid(5, 5)
    assert g.apply_move((2, 2), "N") == (1, 2)
    assert g.apply_move((2, 2), "S") == (3, 2)
    assert g.apply_move((2, 2), "E") == (2, 3)
    assert g.apply_move((2, 2), "W") == (2, 1)
    assert g.apply_move((2, 2), "NE") == (1, 3)
    assert g.apply_move((2, 2), "SW") == (3, 1)
    assert g.apply_move((2, 2), "stay") == (2, 2)


def test_move_off_board_is_noop():
    g = Grid(5, 5)
    # Top-left corner; moving N or W keeps position.
    assert g.apply_move((0, 0), "N") == (0, 0)
    assert g.apply_move((0, 0), "W") == (0, 0)
    assert g.apply_move((0, 0), "NW") == (0, 0)


def test_barrier_blocks_both():
    g = Grid(5, 5)
    assert g.add_barrier((1, 2))
    assert g.is_barrier((1, 2))
    assert not g.is_passable((1, 2))
    # Moving into the barrier is a no-op for any agent.
    assert g.apply_move((2, 2), "N") == (2, 2)  # (1,2) is blocked
    # Cannot place the same barrier twice.
    assert not g.add_barrier((1, 2))
    # Cannot place off-board.
    assert not g.add_barrier((9, 9))


def test_diagonal_toggle():
    g4 = Grid(5, 5, allow_diagonal=False)
    assert set(g4.move_actions()) == set(MOVE_ACTIONS_4)
    # Diagonal move treated as stay when disabled.
    assert g4.apply_move((2, 2), "NE") == (2, 2)

    g8 = Grid(5, 5, allow_diagonal=True)
    assert set(g8.move_actions()) == set(MOVE_ACTIONS_8)


def test_legal_moves_corner():
    g = Grid(3, 3)
    legal = set(g.legal_moves((0, 0)))
    # From corner: stay, S, E, SE are valid; N/W/NW/NE/SW are not.
    assert "stay" in legal
    assert "S" in legal and "E" in legal and "SE" in legal
    assert "N" not in legal and "W" not in legal and "NW" not in legal


def test_distance_metrics():
    g = Grid(5, 5)
    assert g.chebyshev((0, 0), (2, 3)) == 3
    assert g.manhattan((0, 0), (2, 3)) == 5


def test_cell_index_roundtrip():
    g = Grid(5, 4)
    for r in range(5):
        for c in range(4):
            idx = g.cell_index((r, c))
            assert g.index_to_cell(idx) == (r, c)
