"""Tests for mcp_servers/session.py: Mailbox, GameSession, token check."""

import pytest

from core.config import load_config
from core.engine import GameEngine
from mcp_servers.session import AuthError, GameSession, Mailbox, check_token


def make_session(cfg=None):
    cfg = cfg or load_config()
    eng = GameEngine(cfg)
    return GameSession(config=cfg, agent="cop", engine=eng)


def test_mailbox_post_and_latest_for():
    box = Mailbox()
    assert box.latest_for("cop") == ""
    assert box.latest_for("thief") == ""
    box.post("thief", "hello cop")
    assert box.latest_for("cop") == "hello cop"
    assert box.latest_for("thief") == ""
    box.post("cop", "hi thief")
    assert box.latest_for("thief") == "hi thief"


def test_mailbox_latest_for_returns_most_recent():
    box = Mailbox()
    box.post("cop", "first")
    box.post("cop", "second")
    assert box.latest_for("thief") == "second"


def test_check_token_accepts_matching_token():
    cfg = load_config()
    check_token(cfg.mcp.auth_token, cfg)  # should not raise


def test_check_token_rejects_bad_token():
    cfg = load_config()
    with pytest.raises(AuthError):
        check_token("wrong-token", cfg)


def test_game_session_start_sub_game_resets_state():
    session = make_session()
    session.done = True
    session.winner = "cop"
    session.start_sub_game()
    assert session.done is False
    assert session.winner is None
    assert session.sub_game_index == 1
    assert session.engine.state is not None


def test_game_session_defaults():
    session = make_session()
    assert session.cop_total == 0
    assert session.thief_total == 0
    assert session.last_result is None
    assert isinstance(session.mailbox, Mailbox)
