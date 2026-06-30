"""Shared MCP tool implementations (assignment §5.1).

These functions are the *logic* behind the tools exposed by both FastMCP servers
(``cop_server.py`` / ``thief_server.py``). They operate on a process-local
:class:`GameSession` holding the authoritative engine state for one agent's
server; the servers never run the LLM (§5.2) — the orchestrator keeps the two in
sync by mirroring submitted moves. Every tool takes a ``token`` and is rejected
unless it matches the configured ``mcp.auth_token`` (token-based auth, §6).

The session/mailbox dataclasses live in :mod:`mcp_servers.session` and are
re-exported here so existing imports keep working.
"""

from __future__ import annotations

from typing import Tuple

from .session import (  # noqa: F401  (re-exported for callers)
    AuthError,
    GameSession,
    Mailbox,
    check_token as _check,
)


# ---------------------------------------------------------------------------
# Tool implementations. Each returns a plain JSON-serialisable dict.
# ---------------------------------------------------------------------------
def send_message(session: GameSession, token: str, sender: str, text: str) -> dict:
    """Post a free natural-language message (never raw coordinates)."""
    _check(token, session.config)
    session.mailbox.post(sender, text)
    return {"ok": True, "sender": sender, "text": text}


def read_message(session: GameSession, token: str, reader: str) -> dict:
    """Read the latest message addressed to ``reader``."""
    _check(token, session.config)
    return {"ok": True, "reader": reader, "message": session.mailbox.latest_for(reader)}


def get_observation(session: GameSession, token: str, agent: str) -> dict:
    """Partial observation for ``agent`` from this server's engine."""
    _check(token, session.config)
    return session.engine.observation_for(agent).as_dict()


def verify_position(session: GameSession, token: str, agent: str,
                    claimed_pos: Tuple[int, int]) -> dict:
    """Mutual position confirmation (§5.1).

    The agent claims where it believes it is; the server confirms against the
    authoritative state. Used by agents to double-check their own bookkeeping.
    """
    _check(token, session.config)
    s = session.engine.state
    actual = s.cop if agent == "cop" else s.thief
    claimed = (int(claimed_pos[0]), int(claimed_pos[1]))
    return {
        "ok": True,
        "agent": agent,
        "confirmed": actual == claimed,
        "actual_pos": list(actual),
    }


def submit_move(session: GameSession, token: str, agent: str, action: str) -> dict:
    """Apply a movement action for ``agent`` and report capture/status."""
    _check(token, session.config)
    eng = session.engine
    if agent == "thief":
        eng.apply_thief_action({"type": "move", "action": action})
    else:
        eng.apply_cop_action({"type": "move", "action": action})
    captured = eng.is_capture()
    if captured:
        session.done = True
        session.winner = "cop"
    return {
        "ok": True,
        "agent": agent,
        "action": action,
        "captured": captured,
        "self_pos": list(eng.state.cop if agent == "cop" else eng.state.thief),
    }


def place_barrier(session: GameSession, token: str, agent: str) -> dict:
    """Cop-only: place a barrier on the cop's current cell."""
    _check(token, session.config)
    if agent != "cop":
        return {"ok": False, "error": "only the cop may place barriers"}
    eng = session.engine
    before = eng.state.barriers_placed
    eng.apply_cop_action({"type": "barrier"})
    placed = eng.state.barriers_placed > before
    return {
        "ok": True,
        "placed": placed,
        "barriers_placed": eng.state.barriers_placed,
        "barriers_left": session.config.max_barriers - eng.state.barriers_placed,
    }


def game_status(session: GameSession, token: str) -> dict:
    """Scores, move counter, current winner, and done flag."""
    _check(token, session.config)
    eng = session.engine
    return {
        "ok": True,
        "sub_game_index": session.sub_game_index,
        "move_number": eng.state.move_number if eng.state else 0,
        "max_moves": session.config.max_moves,
        "done": session.done,
        "winner": session.winner,
        "cop_total": session.cop_total,
        "thief_total": session.thief_total,
        "cop_pos": list(eng.state.cop) if eng.state else None,
        "thief_pos": list(eng.state.thief) if eng.state else None,
    }


def start_sub_game(session: GameSession, token: str,
                   cop_row: int = -1, cop_col: int = -1,
                   thief_row: int = -1, thief_col: int = -1) -> dict:
    """Begin a fresh sub-game.

    If explicit positions are supplied (>= 0) they are used so the orchestrator
    can force both servers to share identical canonical start cells. Otherwise
    the server samples positions itself.
    """
    _check(token, session.config)
    session.start_sub_game()
    if cop_row >= 0 and thief_row >= 0:
        session.engine.state.cop = (int(cop_row), int(cop_col))
        session.engine.state.thief = (int(thief_row), int(thief_col))
    s = session.engine.state
    return {
        "ok": True,
        "sub_game_index": session.sub_game_index,
        "cop_pos": list(s.cop),
        "thief_pos": list(s.thief),
    }


def advance_move_counter(session: GameSession, token: str) -> dict:
    """Increment the move counter and detect a timeout (thief survival).

    Called by the orchestrator after each full thief+cop turn pair.
    """
    _check(token, session.config)
    eng = session.engine
    eng.state.move_number += 1
    if not session.done and eng.state.move_number >= session.config.max_moves:
        session.done = True
        session.winner = "thief"
    return {
        "ok": True,
        "move_number": eng.state.move_number,
        "done": session.done,
        "winner": session.winner,
    }


def sync_opponent_position(session: GameSession, token: str, agent: str, row: int, col: int) -> dict:
    """Push ``agent``'s ground-truth position, learned from their own server,
    into this server's engine (inter-group bonus play, §12). Works for either
    role via ``agent``, same as :func:`submit_move`."""
    _check(token, session.config)
    session.set_agent_position(agent, row, col)
    return {"ok": True, "agent": agent, "pos": [int(row), int(col)]}
