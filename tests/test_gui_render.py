"""Headless tests for gui/render.py using the SDL dummy video driver."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from core.config import load_config
from core.engine import GameEngine
from gui.render import BoardRenderer


def make_renderer(rows=4, cols=4):
    return BoardRenderer(rows, cols, headless=True)


def test_board_renderer_initialises_screen_dimensions():
    r = make_renderer(4, 4)
    assert r.rows == 4 and r.cols == 4
    assert r.width >= 640
    assert r.height > 0


def test_cell_rect_matches_grid_position():
    r = make_renderer(4, 4)
    rect = r.cell_rect(1, 2)
    assert rect.width == 86  # CELL - 4
    assert rect.x > 0 and rect.y > 0


def test_draw_runs_without_error_and_save_screenshot(tmp_path):
    cfg = load_config()
    cfg.grid_size = (4, 4)
    cfg.validate()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.apply_cop_action({"type": "barrier"})

    r = make_renderer(cfg.rows, cfg.cols)
    info = {
        "sub_game": 1, "num_games": 6, "max_moves": cfg.max_moves,
        "totals": {"cop": 0, "thief": 0}, "cop_msg": "hi", "thief_msg": "hey",
        "llm_available": False, "status_text": "running",
    }
    r.draw(eng, info)
    path = str(tmp_path / "shot.png")
    r.save_screenshot(path)
    assert os.path.exists(path)


def test_draw_panel_handles_long_messages():
    r = make_renderer(4, 4)
    cfg = load_config()
    cfg.grid_size = (4, 4)
    cfg.validate()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    info = {
        "sub_game": 2, "num_games": 6, "max_moves": cfg.max_moves,
        "totals": {"cop": 5, "thief": 10}, "cop_msg": "x" * 200,
        "thief_msg": "y" * 200, "llm_available": True, "status_text": "ok",
    }
    r.draw(eng, info)  # should not raise despite long messages
