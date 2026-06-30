"""Tests for core/state.py: start-position resolution + SubGameResult.as_dict."""

import random

from core.state import GameState, SubGameResult, initial_positions


class _FakeGrid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def chebyshev(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class _FakeStart:
    def __init__(self, cop, thief, min_initial_distance):
        self.cop = cop
        self.thief = thief
        self.min_initial_distance = min_initial_distance


def test_initial_positions_honours_fixed_specs():
    grid = _FakeGrid(5, 5)
    start = _FakeStart([0, 0], [4, 4], 1)
    cop, thief = initial_positions(grid, start, random.Random(0))
    assert cop == (0, 0)
    assert thief == (4, 4)


def test_initial_positions_random_respects_min_distance():
    grid = _FakeGrid(6, 6)
    start = _FakeStart("random", "random", 3)
    cop, thief = initial_positions(grid, start, random.Random(1))
    assert cop != thief
    assert grid.chebyshev(cop, thief) >= 3


def test_initial_positions_falls_back_to_corners_when_unsatisfiable():
    # A 1x1 grid can never satisfy a positive distance -> corner fallback path.
    grid = _FakeGrid(1, 1)
    start = _FakeStart("random", "random", 3)
    cop, thief = initial_positions(grid, start, random.Random(2))
    assert cop == (0, 0)
    assert thief == (0, 0)


def test_subgameresult_as_dict_includes_capture_pos():
    r = SubGameResult(index=2, winner="cop", moves=7, cop_score=20,
                      thief_score=5, capture_pos=(1, 3), barriers_placed=2)
    d = r.as_dict()
    assert d["sub_game"] == 2
    assert d["winner"] == "cop"
    assert d["capture_pos"] == [1, 3]
    assert d["barriers_placed"] == 2


def test_subgameresult_as_dict_without_capture_pos():
    r = SubGameResult(index=1, winner="thief", moves=25, cop_score=5, thief_score=10)
    assert r.as_dict()["capture_pos"] is None


def test_gamestate_defaults():
    s = GameState(cop=(0, 0), thief=(4, 4))
    assert s.move_number == 0
    assert s.barriers_placed == 0
    assert s.history == []
