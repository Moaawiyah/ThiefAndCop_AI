"""Headless tests for gui/visualizer.py using the SDL dummy video driver."""

import os
import sys
import types

import pytest

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


def test_pump_returns_false_on_quit(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)
    v = viz_mod.Visualizer(cfg, headless=True, fps=1000, screenshots=0)
    quit_event = types.SimpleNamespace(type=viz_mod.pygame.QUIT)
    monkeypatch.setattr(viz_mod.pygame.event, "get", lambda: [quit_event])
    assert v._pump() is False


def test_pump_returns_false_on_escape(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)
    v = viz_mod.Visualizer(cfg, headless=True, fps=1000, screenshots=0)
    esc = types.SimpleNamespace(type=viz_mod.pygame.KEYDOWN, key=viz_mod.pygame.K_ESCAPE)
    monkeypatch.setattr(viz_mod.pygame.event, "get", lambda: [esc])
    assert v._pump() is False


def test_visualizer_requires_pygame(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)
    monkeypatch.setattr(viz_mod, "_PYGAME_OK", False)
    with pytest.raises(RuntimeError):
        viz_mod.Visualizer(cfg, headless=True)


def test_main_without_pygame_exits(monkeypatch):
    monkeypatch.setattr(viz_mod, "_PYGAME_OK", False)
    monkeypatch.setattr(sys, "argv", ["visualizer.py"])
    with pytest.raises(SystemExit):
        viz_mod.main()


def test_visualizer_thief_win_scoring(tmp_path, monkeypatch):
    cfg = small_config(tmp_path, num_games=1, max_moves=3, rows=5, cols=5)
    v = viz_mod.Visualizer(cfg, headless=True, fps=1000, screenshots=0)
    # Force both agents to stay put: no capture -> thief survives -> thief wins.
    stay = {"type": "move", "action": "stay"}
    monkeypatch.setattr(v, "cop_policy", lambda obs, eng: stay)
    monkeypatch.setattr(v, "thief_policy", lambda obs, eng: stay)
    totals = v.run()
    assert totals["thief"] == cfg.scoring.thief_win
    assert totals["cop"] == cfg.scoring.cop_loss
