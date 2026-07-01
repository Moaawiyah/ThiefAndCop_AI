"""Tests for the live web GUI: gui/live_game.py and gui/live_server.py."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from core.config import load_config
from core.engine import GameEngine
from gui.live_game import LiveGame
from gui.live_server import make_handler, serve_in_background
from gui.series_gif import SeriesGif


def small_config(tmp_path, rows=4, cols=4, max_moves=6, num_games=1):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.max_moves = max_moves
    cfg.num_games = num_games
    cfg.llm.enabled = False                # templated NL, no network
    cfg.qlearning.q_dir = str(tmp_path)    # no trained tables -> heuristic policies
    cfg.validate()
    return cfg


def test_live_game_snapshot_has_board_keys(tmp_path):
    cfg = small_config(tmp_path)
    g = LiveGame(cfg, delay=0)
    snap = g.snapshot()
    for k in ("rows", "cols", "cop", "thief", "barriers", "cop_msg",
              "thief_msg", "move", "totals", "sub_game", "status"):
        assert k in snap
    assert 0 <= snap["cop"][0] < cfg.rows and 0 <= snap["cop"][1] < cfg.cols


def test_live_game_play_sub_game_runs_and_scores(tmp_path):
    cfg = small_config(tmp_path)
    g = LiveGame(cfg, delay=0)
    winner = g.play_sub_game(1)
    assert winner in ("cop", "thief")
    totals = g.snapshot()["totals"]
    assert totals["cop"] + totals["thief"] > 0


def test_series_gif_snapshot_reflects_engine(tmp_path):
    cfg = small_config(tmp_path)
    view = SeriesGif(cfg, enabled=False)   # no pygame frames, live snapshot still works
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 2)
    eng.state.thief = (3, 3)
    view.start_sub_game(2)
    view.turn("cop", "freeze!", eng)
    snap = view.snapshot()
    assert snap["cop"] == [1, 2] and snap["thief"] == [3, 3]
    assert snap["cop_msg"] == "freeze!" and snap["sub_game"] == 2


def test_serve_in_background_returns_live_url(tmp_path):
    cfg = small_config(tmp_path)
    view = SeriesGif(cfg, enabled=False)
    url = serve_in_background(view, port=0)   # ephemeral port
    assert url.startswith("http://127.0.0.1:")


def test_live_server_serves_page_and_state(tmp_path):
    cfg = small_config(tmp_path)
    g = LiveGame(cfg, delay=0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(g))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "Cop" in page and "board" in page
        raw = urllib.request.urlopen(f"http://127.0.0.1:{port}/state").read()
        state = json.loads(raw)
        assert state["rows"] == cfg.rows and "cop" in state
    finally:
        httpd.shutdown()
        httpd.server_close()
