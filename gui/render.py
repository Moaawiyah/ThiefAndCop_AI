"""Pure pygame drawing primitives for the visualizer (assignment §6 / Phase 6).

Split out of :mod:`gui.visualizer` to keep the run/loop/CLI module small. The
:class:`BoardRenderer` owns the pygame surface, fonts and palette and knows how
to paint a single frame (grid, barriers, agents, info panel). The visualizer
keeps all game-driving logic and just hands frame data to the renderer.
"""

from __future__ import annotations

import os

# pygame is optional at import time (headless-safe).
try:
    import pygame
    _PYGAME_OK = True
except Exception:  # pragma: no cover
    pygame = None
    _PYGAME_OK = False

CELL = 90
MARGIN = 20
PANEL_H = 150
COLORS = {
    "bg": (24, 26, 32),
    "grid": (60, 64, 74),
    "cell": (40, 43, 51),
    "cop": (66, 135, 245),
    "thief": (235, 87, 87),
    "barrier": (120, 120, 120),
    "text": (230, 230, 230),
    "dim": (160, 160, 160),
    "panel": (32, 34, 41),
}


class BoardRenderer:
    """Owns the pygame surface/fonts and paints one frame on demand."""

    def __init__(self, rows: int, cols: int, headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        self.rows, self.cols = rows, cols
        # Ensure the panel has room for the NL message lines.
        self.width = max(cols * CELL + 2 * MARGIN, 640)
        self.height = rows * CELL + 2 * MARGIN + PANEL_H
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("HW6 — Cop & Thief MCP pursuit")
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 16)
        self.big = pygame.font.SysFont("menlo,consolas,monospace", 22, bold=True)

    def cell_rect(self, r, c):
        return pygame.Rect(MARGIN + c * CELL, MARGIN + r * CELL, CELL - 4, CELL - 4)

    def draw(self, engine, info: dict) -> None:
        """Paint a full frame from the engine state + an ``info`` dict.

        ``info`` keys: sub_game, num_games, totals, cop_msg, thief_msg,
        llm_available, status_text.
        """
        s = engine.state
        self.screen.fill(COLORS["bg"])
        # Cells + barriers.
        for r in range(self.rows):
            for c in range(self.cols):
                rect = self.cell_rect(r, c)
                color = COLORS["barrier"] if (r, c) in engine.grid.barriers else COLORS["cell"]
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, COLORS["grid"], rect, width=1, border_radius=8)

        # Agents (circles).
        cr = self.cell_rect(*s.cop)
        pygame.draw.circle(self.screen, COLORS["cop"], cr.center, CELL // 3)
        self.screen.blit(self.big.render("C", True, (255, 255, 255)),
                         (cr.centerx - 8, cr.centery - 12))
        tr = self.cell_rect(*s.thief)
        pygame.draw.circle(self.screen, COLORS["thief"], tr.center, CELL // 3)
        self.screen.blit(self.big.render("T", True, (255, 255, 255)),
                         (tr.centerx - 8, tr.centery - 12))

        self._draw_panel(s, info)
        pygame.display.flip()

    def _draw_panel(self, s, info: dict) -> None:
        """Panel: scores, move counter, NL messages."""
        py = MARGIN + self.rows * CELL + 8
        pygame.draw.rect(self.screen, COLORS["panel"],
                         pygame.Rect(0, py, self.width, PANEL_H))
        llm = "on" if info["llm_available"] else "off (template NL)"
        lines = [
            (self.big, COLORS["text"],
             f"Sub-game {info['sub_game']}/{info['num_games']}   "
             f"move {s.move_number}/{info['max_moves']}   "
             f"cop {info['totals']['cop']}  thief {info['totals']['thief']}"),
            (self.font, COLORS["cop"], f"COP:   {info['cop_msg'][:70]}"),
            (self.font, COLORS["thief"], f"THIEF: {info['thief_msg'][:70]}"),
            (self.font, COLORS["dim"], f"LLM(GLM): {llm}   {info['status_text']}"),
        ]
        for i, (font, color, text) in enumerate(lines):
            self.screen.blit(font.render(text, True, color), (MARGIN, py + 10 + i * 30))

    def save_screenshot(self, path: str) -> None:
        pygame.image.save(self.screen, path)
