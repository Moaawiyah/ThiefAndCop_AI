"""GIF capture + optional live web view for an orchestrator-driven series.

Wraps a headless :class:`~gui.render.BoardRenderer` + :class:`~gui.animate.GifRecorder`
so the MCP orchestrator can emit ``artifacts/game_full.gif``, and optionally serves
a live browser view of the very same game (the web server polls :meth:`snapshot`).
It remembers the latest cop/thief NL messages + running scores so the orchestrator
just hands it one turn at a time. The GIF degrades to a no-op when pygame/Pillow are
unavailable; the live view works regardless.
"""

from __future__ import annotations

import threading
import time

from gui.render import _PYGAME_OK, BoardRenderer
from gui.animate import GifRecorder


class SeriesGif:
    """Per-turn frame recorder + thread-safe live snapshot provider."""

    def __init__(self, config, enabled: bool = True, serve: bool = False,
                 host: str = "127.0.0.1", port: int = 8000, delay: float = 0.4):
        self.config = config
        self.available = bool(enabled) and _PYGAME_OK
        self.cop_msg = ""
        self.thief_msg = ""
        self.sub_game = 0
        self.totals = {"cop": 0, "thief": 0}
        self.url = ""
        self._delay = delay if serve else 0.0
        self._engine = None
        self._status = "starting"
        self._lock = threading.Lock()
        if self.available:
            self.renderer = BoardRenderer(config.rows, config.cols, headless=True)
            self.recorder = GifRecorder(config.q_dir_abs(), enabled=True)
        if serve:
            from gui.live_server import serve_in_background
            self.url = serve_in_background(self, host, port)
            print(f"[live] watch the game at: {self.url}")

    def start_sub_game(self, index: int) -> None:
        self.sub_game = index

    def add_score(self, cop: int, thief: int) -> None:
        self.totals["cop"] += cop
        self.totals["thief"] += thief

    def turn(self, agent: str, msg: str, engine, llm_available: bool = False) -> None:
        """Record ``agent``'s message + board; capture a GIF frame; pace the view."""
        setattr(self, f"{agent}_msg", msg)
        with self._lock:
            self._engine = engine
            self._status = f"{agent} moved"
        if self.available:
            self.renderer.draw(engine, {
                "sub_game": self.sub_game, "num_games": self.config.num_games,
                "max_moves": self.config.max_moves, "totals": self.totals,
                "cop_msg": self.cop_msg, "thief_msg": self.thief_msg,
                "llm_available": llm_available, "status_text": self._status,
            })
            self.recorder.capture(self.renderer.screen)
        if self._delay:
            time.sleep(self._delay)

    def snapshot(self) -> dict:
        """Current board + dialogue as a JSON-friendly dict (for the live view)."""
        with self._lock:
            eng = self._engine
            cop = list(eng.state.cop) if eng else [0, 0]
            thief = list(eng.state.thief) if eng else [0, 0]
            barriers = [list(b) for b in eng.grid.barriers] if eng else []
            move = eng.state.move_number if eng else 0
            status = self._status
        return {
            "rows": self.config.rows, "cols": self.config.cols,
            "cop": cop, "thief": thief, "barriers": barriers,
            "cop_msg": self.cop_msg, "thief_msg": self.thief_msg,
            "move": move, "max_moves": self.config.max_moves,
            "sub_game": self.sub_game, "num_games": self.config.num_games,
            "totals": dict(self.totals), "status": status,
        }

    def save(self) -> str:
        return self.recorder.save("game_full") if self.available else ""
