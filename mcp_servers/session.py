"""Per-server session state for the MCP tools (assignment §5.1).

Split out of :mod:`mcp_servers.tools` to keep that module small. Holds the
authoritative engine state for one agent's server, plus a tiny two-slot mailbox
for the free-NL messages exchanged between agents. Re-exported from
:mod:`mcp_servers.tools` so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from core.config import Config
from core.engine import GameEngine, SubGameResult


class AuthError(Exception):
    """Raised when a tool is called with an invalid token."""


@dataclass
class Mailbox:
    """A tiny two-slot mailbox for free-NL messages between agents."""

    cop_to_thief: List[str] = field(default_factory=list)
    thief_to_cop: List[str] = field(default_factory=list)

    def post(self, sender: str, text: str) -> None:
        if sender == "cop":
            self.cop_to_thief.append(text)
        else:
            self.thief_to_cop.append(text)

    def latest_for(self, reader: str) -> str:
        """Latest message addressed to ``reader``."""
        box = self.thief_to_cop if reader == "cop" else self.cop_to_thief
        return box[-1] if box else ""


@dataclass
class GameSession:
    """Authoritative state for one agent's MCP server."""

    config: Config
    agent: str                       # "cop" or "thief" (this server's owner)
    engine: GameEngine
    mailbox: Mailbox = field(default_factory=Mailbox)
    sub_game_index: int = 0
    cop_total: int = 0
    thief_total: int = 0
    last_result: Optional[SubGameResult] = None
    done: bool = False               # current sub-game finished?
    winner: Optional[str] = None

    def start_sub_game(self) -> None:
        self.engine.reset_sub_game()
        self.sub_game_index += 1
        self.done = False
        self.winner = None

    def set_agent_position(self, agent: str, row: int, col: int) -> None:
        """Override one agent's position (either role) with ground truth from
        their own server — used for inter-group bonus play (§12)."""
        pos = (int(row), int(col))
        if agent == "cop":
            self.engine.state.cop = pos
        else:
            self.engine.state.thief = pos


def check_token(token: str, config: Config) -> None:
    """Raise :class:`AuthError` unless ``token`` matches the configured token."""
    if token != config.mcp.auth_token:
        raise AuthError("invalid auth token")
