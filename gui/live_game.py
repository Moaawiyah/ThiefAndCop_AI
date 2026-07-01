"""Background game runner for the live web GUI (assignment §6).

Plays the trained cop-vs-thief series in a daemon thread, refreshing a
thread-safe JSON snapshot after every half-move (board + NL trash-talk) so the
web server can stream it to the browser. Loops forever so the live URL always
shows action.
"""

from __future__ import annotations

import random
import threading
import time

from core.engine import GameEngine
from agents.policy import build_policy
from llm.glm_client import LLMClient


class LiveGame:
    """Runs the series in a thread and exposes the latest board as a dict."""

    def __init__(self, config, delay: float = 0.7):
        self.config = config
        self.delay = delay
        self.lock = threading.Lock()
        self.engine = GameEngine(config, rng=random.Random())
        self.cop = build_policy("cop", config)
        self.thief = build_policy("thief", config)
        self.llm = LLMClient(config.llm)
        self.cop_msg = self.thief_msg = ""
        self.sub_game = 0
        self.totals = {"cop": 0, "thief": 0}
        self._snap: dict = {}
        self.engine.reset_sub_game()
        self._snapshot("starting")

    def _snapshot(self, status: str) -> None:
        s = self.engine.state
        with self.lock:
            self._snap = {
                "rows": self.config.rows, "cols": self.config.cols,
                "cop": list(s.cop), "thief": list(s.thief),
                "barriers": [list(b) for b in self.engine.grid.barriers],
                "cop_msg": self.cop_msg, "thief_msg": self.thief_msg,
                "move": s.move_number, "max_moves": self.config.max_moves,
                "sub_game": self.sub_game, "num_games": self.config.num_games,
                "totals": dict(self.totals), "status": status,
                "llm": self.llm.available,
            }

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self._snap)

    def _half(self, role: str, pol, status: str) -> bool:
        """Play one agent's move, update its message + the snapshot, pause."""
        obs = self.engine.observation_for(role)
        opp_msg = self.thief_msg if role == "cop" else self.cop_msg
        setattr(self, f"{role}_msg", self.llm.generate_message(obs, opp_msg))
        act = pol(obs, self.engine)
        apply = self.engine.apply_cop_action if role == "cop" else self.engine.apply_thief_action
        apply(act)
        self._snapshot(status)
        time.sleep(self.delay)
        return self.engine.is_capture()

    def play_sub_game(self, gi: int) -> str:
        self.sub_game = gi
        self.engine.reset_sub_game()
        for p in (self.cop, self.thief):
            if hasattr(p, "reset_belief"):
                p.reset_belief()
        self.cop_msg = self.thief_msg = ""
        s = self.engine.state
        winner = "thief"
        self._snapshot(f"sub-game {gi} start")
        time.sleep(self.delay)
        while s.move_number < self.config.max_moves:
            s.move_number += 1
            if self._half("thief", self.thief, "thief moved") or \
               self._half("cop", self.cop, "cop moved"):
                winner = "cop"
                break
        if winner == "cop":
            self.totals["cop"] += self.config.scoring.cop_win
            self.totals["thief"] += self.config.scoring.thief_loss
        else:
            self.totals["cop"] += self.config.scoring.cop_loss
            self.totals["thief"] += self.config.scoring.thief_win
        self._snapshot(f"sub-game {gi}: {winner} wins")
        time.sleep(self.delay)
        return winner

    def run_forever(self) -> None:  # pragma: no cover - infinite daemon loop
        while True:
            self.totals = {"cop": 0, "thief": 0}
            for gi in range(1, self.config.num_games + 1):
                self.play_sub_game(gi)
