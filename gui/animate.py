"""Animated-GIF recorder for the visualizer (optional report evidence).

Buffers every rendered pygame frame across the whole series and writes a single
animated GIF (``artifacts/game_full.gif``) once the full run completes, using
Pillow — no extra runtime services needed. Degrades to a no-op when
Pillow/pygame are unavailable, so the GUI never breaks on its account.
"""

from __future__ import annotations

import os

try:  # optional dependency guard — animation is a nice-to-have
    import pygame
    from PIL import Image
    _ANIM_OK = True
except Exception:  # pragma: no cover
    _ANIM_OK = False


class GifRecorder:
    """Captures frames from a pygame surface and writes them as animated GIFs."""

    def __init__(self, out_dir: str, fps: int = 4, enabled: bool = True):
        self.out_dir = out_dir
        self.duration = int(1000 / max(1, fps))
        self.frames: list = []
        self.available = _ANIM_OK and enabled

    def capture(self, surface) -> None:
        """Append the current surface as one GIF frame."""
        if not self.available:
            return
        raw = pygame.image.tostring(surface, "RGB")
        self.frames.append(Image.frombytes("RGB", surface.get_size(), raw))

    def save(self, name: str) -> str:
        """Write the buffered frames to ``<out_dir>/<name>.gif`` and reset."""
        if not self.available or not self.frames:
            self.frames = []
            return ""
        path = os.path.join(self.out_dir, f"{name}.gif")
        first, rest = self.frames[0], self.frames[1:]
        first.save(path, save_all=True, append_images=rest,
                   duration=self.duration, loop=0)
        self.frames = []
        print(f"[gui] saved animation {path}")
        return path
