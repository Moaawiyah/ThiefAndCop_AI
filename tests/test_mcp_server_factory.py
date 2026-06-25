"""Real in-process MCP integration tests for server_factory.py via fastmcp.Client.

No mocking: connects to an actual FastMCP server object over the in-memory
transport and exercises the real MCP protocol (tool listing + calling).
"""

import asyncio

from fastmcp import Client

from core.config import load_config
from mcp_servers.server_factory import TOOL_NAMES, build_server

TOKEN = load_config().mcp.auth_token


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_build_server_registers_all_tools():
    cfg = load_config()
    server = build_server(cfg, role="cop")

    async def _go():
        async with Client(server) as client:
            tools = await client.list_tools()
            return {t.name for t in tools}

    names = run(_go())
    assert names == set(TOOL_NAMES)


def test_call_tool_get_observation_over_mcp():
    cfg = load_config()
    server = build_server(cfg, role="thief")

    async def _go():
        async with Client(server) as client:
            res = await client.call_tool("get_observation", {"token": TOKEN, "agent": "thief"})
            return res.data

    data = run(_go())
    assert data["agent"] == "thief"
    assert "self_pos" in data


def test_call_tool_submit_move_and_status_round_trip():
    cfg = load_config()
    server = build_server(cfg, role="cop")
    session = server._game_session
    session.engine.state.cop = (0, 0)
    session.engine.state.thief = (4, 4)

    async def _go():
        async with Client(server) as client:
            move = await client.call_tool(
                "submit_move", {"token": TOKEN, "agent": "cop", "action": "E"})
            status = await client.call_tool("game_status", {"token": TOKEN})
            return move.data, status.data

    move_res, status_res = run(_go())
    assert move_res["captured"] is False
    assert status_res["cop_pos"] == [0, 1]


def test_call_tool_invalid_token_raises_tool_error():
    cfg = load_config()
    server = build_server(cfg, role="cop")

    async def _go():
        async with Client(server) as client:
            await client.call_tool("game_status", {"token": "wrong"})

    raised = False
    try:
        run(_go())
    except Exception:
        raised = True
    assert raised


def test_start_sub_game_and_place_barrier_over_mcp():
    cfg = load_config()
    server = build_server(cfg, role="cop")

    async def _go():
        async with Client(server) as client:
            start = await client.call_tool(
                "start_sub_game",
                {"token": TOKEN, "cop_row": 1, "cop_col": 1, "thief_row": 3, "thief_col": 3},
            )
            barrier = await client.call_tool(
                "place_barrier", {"token": TOKEN, "agent": "cop"})
            return start.data, barrier.data

    start_res, barrier_res = run(_go())
    assert start_res["cop_pos"] == [1, 1]
    assert barrier_res["placed"] is True


def test_send_and_read_message_over_mcp():
    cfg = load_config()
    server = build_server(cfg, role="cop")

    async def _go():
        async with Client(server) as client:
            await client.call_tool(
                "send_message", {"token": TOKEN, "sender": "cop", "text": "hi thief"})
            read = await client.call_tool(
                "read_message", {"token": TOKEN, "reader": "thief"})
            return read.data

    data = run(_go())
    assert data["message"] == "hi thief"


def test_verify_position_and_advance_move_counter_over_mcp():
    cfg = load_config()
    server = build_server(cfg, role="cop")
    session = server._game_session
    actual = session.engine.state.cop

    async def _go():
        async with Client(server) as client:
            verify = await client.call_tool(
                "verify_position",
                {"token": TOKEN, "agent": "cop", "row": actual[0], "col": actual[1]},
            )
            advance = await client.call_tool("advance_move_counter", {"token": TOKEN})
            return verify.data, advance.data

    verify_res, advance_res = run(_go())
    assert verify_res["confirmed"] is True
    assert advance_res["move_number"] == 1


def test_sync_opponent_position_over_mcp():
    cfg = load_config()
    server = build_server(cfg, role="cop")
    session = server._game_session

    async def _go():
        async with Client(server) as client:
            res = await client.call_tool(
                "sync_opponent_position",
                {"token": TOKEN, "agent": "thief", "row": 2, "col": 2},
            )
            return res.data

    data = run(_go())
    assert data == {"ok": True, "agent": "thief", "pos": [2, 2]}
    assert session.engine.state.thief == (2, 2)
