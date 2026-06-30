"""Pygame real-time visualizer (assignment §6 / Phase 6).

Renders the grid with the cop, thief, barriers, the latest NL messages, the move
counter and live scores. It drives a full in-process series through the engine,
animating each step.

The pygame import is guarded so importing this module never fails on a headless
box. Two run modes:

    python3 gui/visualizer.py                 # live window (needs a display)
    python3 gui/visualizer.py --headless      # render frames to artifacts/*.png
    python3 gui/visualizer.py --screenshots 3 # live + save N screenshots

On a headless machine pass ``--headless`` (uses SDL's dummy video driver) to
prove the renderer works and to capture screenshots for the report.

Pure drawing primitives live in :mod:`gui.render`.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from core.engine import GameEngine
from agents.policy import build_policy
from llm.glm_client import LLMClient
from gui.render import _PYGAME_OK, BoardRenderer, pygame


class Visualizer:
    """Animates an in-process series with the engine + Q/heuristic policies."""

    def __init__(self, config, headless: bool = False, fps: int = 4,
                 screenshots: int = 0):
        if not _PYGAME_OK:
            raise RuntimeError("pygame is not installed; cannot run the GUI.")
        self.config = config
        self.headless = headless
        self.fps = fps
        self.screenshots = screenshots
        self._shots_saved = 0

        self.renderer = BoardRenderer(config.rows, config.cols, headless=headless)
        self.clock = pygame.time.Clock()

        self.engine = GameEngine(config, rng=random.Random(7))
        self.cop_policy = build_policy("cop", config)
        self.thief_policy = build_policy("thief", config)
        self.llm = LLMClient(config.llm)

        self.cop_msg = ""
        self.thief_msg = ""
        self.totals = {"cop": 0, "thief": 0}
        self.sub_game = 0
        self.q_dir = config.q_dir_abs()

    # ----- drawing ---------------------------------------------------------
    def draw(self, status_text: str = ""):
        self.renderer.draw(self.engine, {
            "sub_game": self.sub_game,
            "num_games": self.config.num_games,
            "max_moves": self.config.max_moves,
            "totals": self.totals,
            "cop_msg": self.cop_msg,
            "thief_msg": self.thief_msg,
            "llm_available": self.llm.available,
            "status_text": status_text,
        })

    def _maybe_screenshot(self, tag: str):
        if self.screenshots and self._shots_saved < self.screenshots:
            path = os.path.join(self.q_dir, f"gui_{tag}.png")
            self.renderer.save_screenshot(path)
            self._shots_saved += 1
            print(f"[gui] saved screenshot {path}")

    def _pump(self) -> bool:
        """Process events; return False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    # ----- main loop -------------------------------------------------------
    def run(self):
        running = True
        for gi in range(1, self.config.num_games + 1):
            if not running:
                break
            self.sub_game = gi
            self.engine.reset_sub_game()
            if hasattr(self.cop_policy, "reset_belief"):
                self.cop_policy.reset_belief()
            if hasattr(self.thief_policy, "reset_belief"):
                self.thief_policy.reset_belief()
            s = self.engine.state
            winner = "thief"

            while s.move_number < self.config.max_moves and running:
                running = self._pump()
                s.move_number += 1

                t_obs = self.engine.observation_for("thief")
                self.thief_msg = self.llm.generate_message(t_obs, self.cop_msg)
                self.engine.apply_thief_action(self.thief_policy(t_obs, self.engine))
                self.draw("thief moved")
                self._maybe_screenshot(f"g{gi}_m{s.move_number}_thief")
                self.clock.tick(self.fps)
                if self.engine.is_capture():
                    winner = "cop"
                    break

                c_obs = self.engine.observation_for("cop")
                self.cop_msg = self.llm.generate_message(c_obs, self.thief_msg)
                self.engine.apply_cop_action(self.cop_policy(c_obs, self.engine))
                self.draw("cop moved")
                self._maybe_screenshot(f"g{gi}_m{s.move_number}_cop")
                self.clock.tick(self.fps)
                if self.engine.is_capture():
                    winner = "cop"
                    break

            if winner == "cop":
                self.totals["cop"] += self.config.scoring.cop_win
                self.totals["thief"] += self.config.scoring.thief_loss
            else:
                self.totals["cop"] += self.config.scoring.cop_loss
                self.totals["thief"] += self.config.scoring.thief_win
            self.draw(f"sub-game {gi}: {winner} wins")
            self._maybe_screenshot(f"g{gi}_result")
            if not self.headless:
                pygame.time.wait(600)

        self.draw("series complete")
        self._maybe_screenshot("final")
        if not self.headless:
            # Keep window open until closed.
            waiting = True
            while waiting:
                waiting = self._pump()
                self.clock.tick(30)
        pygame.quit()
        return self.totals


def main():
    p = argparse.ArgumentParser(description="HW6 Pygame visualizer")
    p.add_argument("--headless", action="store_true",
                   help="use SDL dummy driver (no display needed)")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--screenshots", type=int, default=0,
                   help="save up to N screenshots into the artifacts dir")
    args = p.parse_args()

    if not _PYGAME_OK:
        print("pygame is not installed — GUI unavailable.")
        sys.exit(1)

    config = load_config()
    # In headless mode, always capture a few screenshots for the report.
    shots = args.screenshots or (5 if args.headless else 0)
    viz = Visualizer(config, headless=args.headless, fps=args.fps, screenshots=shots)
    totals = viz.run()
    print(f"GUI series complete. totals={totals}")


if __name__ == "__main__":
    main()
