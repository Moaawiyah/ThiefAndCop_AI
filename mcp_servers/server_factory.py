"""Factory that builds a FastMCP server exposing the shared game tools.

Both ``cop_server.py`` and ``thief_server.py`` call :func:`build_server` with
their role. The server holds a :class:`GameSession` and registers the tools from
:mod:`mcp_servers.tools`. Per the architecture rule (§5.2) the server exposes
*tools only* — no LLM lives here.

The two servers run on separate ports (from ``config.mcp``) over HTTP so the
orchestrator (MCP client) can connect to both, exactly as the assignment
requires two separate MCP servers (one per agent).
"""

from __future__ import annotations

from typing import List

from fastmcp import FastMCP

from core.config import Config
from core.engine import GameEngine
from . import tools as T
from .tools import GameSession


def build_server(config: Config, role: str) -> FastMCP:
    """Create a FastMCP server for ``role`` ('cop' or 'thief')."""
    engine = GameEngine(config)
    engine.reset_sub_game()
    session = GameSession(config=config, agent=role, engine=engine)

    mcp = FastMCP(name=f"{role}-mcp-server")

    @mcp.tool
    def send_message(token: str, sender: str, text: str) -> dict:
        """Post a free natural-language message to the opponent's mailbox."""
        return T.send_message(session, token, sender, text)

    @mcp.tool
    def read_message(token: str, reader: str) -> dict:
        """Read the latest natural-language message addressed to `reader`."""
        return T.read_message(session, token, reader)

    @mcp.tool
    def get_observation(token: str, agent: str) -> dict:
        """Return the partial observation (vision-limited) for `agent`."""
        return T.get_observation(session, token, agent)

    @mcp.tool
    def verify_position(token: str, agent: str, row: int, col: int) -> dict:
        """Confirm whether `agent` is at the claimed (row, col) cell."""
        return T.verify_position(session, token, agent, (row, col))

    @mcp.tool
    def submit_move(token: str, agent: str, action: str) -> dict:
        """Apply a movement action (N/S/E/W/diagonals/stay) for `agent`."""
        return T.submit_move(session, token, agent, action)

    @mcp.tool
    def place_barrier(token: str, agent: str) -> dict:
        """Cop-only: place an impassable barrier on the cop's current cell."""
        return T.place_barrier(session, token, agent)

    @mcp.tool
    def game_status(token: str) -> dict:
        """Return scores, move counter, winner and done flag."""
        return T.game_status(session, token)

    @mcp.tool
    def advance_move_counter(token: str) -> dict:
        """Advance the move counter; flags a thief survival win on timeout."""
        return T.advance_move_counter(session, token)

    @mcp.tool
    def start_sub_game(token: str, cop_row: int = -1, cop_col: int = -1,
                       thief_row: int = -1, thief_col: int = -1) -> dict:
        """Begin a fresh sub-game, optionally at explicit canonical positions."""
        return T.start_sub_game(session, token, cop_row, cop_col, thief_row, thief_col)

    @mcp.tool
    def sync_opponent_position(token: str, agent: str, row: int, col: int) -> dict:
        """Inter-group bonus play (§12): push `agent`'s real position, learned
        from their own server, into this server's engine."""
        return T.sync_opponent_position(session, token, agent, row, col)

    # Expose the session for in-process integration tests.
    mcp._game_session = session  # type: ignore[attr-defined]
    return mcp


TOOL_NAMES: List[str] = [
    "send_message", "read_message", "get_observation", "verify_position",
    "submit_move", "place_barrier", "game_status", "advance_move_counter",
    "start_sub_game", "sync_opponent_position",
]
