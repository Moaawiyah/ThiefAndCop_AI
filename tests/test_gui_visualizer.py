"""Headless tests for gui/visualizer.py using the SDL dummy video driver."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gui.visualizer as viz_mod
from core.config import load_config


def small_config(tmp_path, num_games=1, max_moves=6, rows=4, cols=4):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.num_games = num_games
    cfg.max_moves = max_moves
    cfg.llm.enabled = False
    cfg.qlearning.q_dir = str(tmp_path)
    cfg.validate()
    return cfg


def test_visualizer_run_completes_series_headless(tmp_path):
    cfg = small_config(tmp_path)
    v = viz_mod.Visualizer(cfg, headless=True, fps=1000, screenshots=0)
    totals = v.run()
    assert totals["cop"] + totals["thief"] > 0


def test_visualizer_takes_screenshots(tmp_path):
    cfg = small_config(tmp_path)
    v = viz_mod.Visualizer(cfg, headless=True, fps=1000, screenshots=2)
    v.run()
    assert v._shots_saved == 2
    saved = [f for f in os.listdir(cfg.q_dir_abs()) if f.startswith("gui_")]
    assert len(saved) == 2


def test_visualizer_draw_updates_messages(tmp_path):
    cfg = small_config(tmp_path)
    v = viz_mod.Visualizer(cfg, headless=True, fps=1000, screenshots=0)
    v.engine.reset_sub_game()
    v.cop_msg = "hello"
    v.thief_msg = "world"
    v.draw("status")  # should not raise


def test_main_headless_cli_runs(tmp_path, monkeypatch, capsys):
    cfg = small_config(tmp_path, num_games=1, max_moves=4, rows=3, cols=3)
    monkeypatch.setattr(viz_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(sys, "argv", ["visualizer.py", "--headless", "--screenshots", "1"])
    viz_mod.main()
    out = capsys.readouterr().out
    assert "GUI series complete" in out


def test_pump_returns_true_with_no_events(tmp_path):
    cfg = small_config(tmp_path)
    v = viz_mod.Visualizer(cfg, headless=True, fps=1000, screenshots=0)
    assert v._pump() is True
