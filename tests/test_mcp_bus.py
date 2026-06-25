"""Tests for mcp_client/bus.py: InProcessBus (real) + light NetworkedBus checks."""

from unittest.mock import MagicMock, patch

import pytest

from core.config import load_config
from mcp_client.bus import InProcessBus, NetworkedBus, ToolBus


def test_inprocess_bus_builds_two_sessions():
    cfg = load_config()
    bus = InProcessBus(cfg)
    assert set(bus.sessions.keys()) == {"cop", "thief"}
    assert bus.sessions["cop"].agent == "cop"
    assert bus.sessions["thief"].agent == "thief"


def test_inprocess_bus_call_dispatches_to_session_and_tool():
    cfg = load_config()
    bus = InProcessBus(cfg)
    token = cfg.mcp.auth_token
    res = bus.call("cop", "send_message", token=token, sender="cop", text="hi")
    assert res == {"ok": True, "sender": "cop", "text": "hi"}
    read = bus.call("thief", "read_message", token=token, reader="thief")
    # NetworkedBus/InProcessBus keep separate sessions, so thief server's own
    # mailbox is independent unless explicitly mirrored by the orchestrator.
    assert read["ok"] is True


def test_inprocess_bus_submit_move_and_status():
    cfg = load_config()
    bus = InProcessBus(cfg)
    token = cfg.mcp.auth_token
    bus.call("cop", "start_sub_game", token=token)
    status = bus.call("cop", "game_status", token=token)
    assert status["ok"] is True
    assert "cop_pos" in status


def test_toolbus_base_call_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        ToolBus().call("cop", "game_status")


def test_networked_bus_opens_clients_and_calls_via_event_loop():
    cfg = load_config()
    fake_client_cls = MagicMock()
    fake_client = fake_client_cls.return_value

    async def fake_aenter(*a, **k):
        return fake_client

    fake_client.__aenter__ = MagicMock(side_effect=fake_aenter)

    async def fake_aexit(*a, **k):
        return None

    fake_client.__aexit__ = MagicMock(side_effect=fake_aexit)

    class FakeResult:
        data = {"ok": True}

    async def fake_call_tool(tool, kwargs):
        return FakeResult()

    fake_client.call_tool = fake_call_tool

    with patch("fastmcp.Client", fake_client_cls):
        bus = NetworkedBus(cfg)
        try:
            res = bus.call("cop", "game_status", token=cfg.mcp.auth_token)
            assert res == {"ok": True}
        finally:
            bus.close()
    assert fake_client_cls.call_count == 2
