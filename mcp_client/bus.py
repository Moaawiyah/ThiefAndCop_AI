"""Tool transport abstraction for the orchestrator (assignment §5.2).

Split out of :mod:`orchestrator` to keep that module small. The orchestrator's
turn loop talks to the two MCP servers exclusively through a :class:`ToolBus`,
which has two implementations:

* :class:`InProcessBus` — calls the shared :mod:`mcp_servers.tools` functions
  directly on two :class:`~mcp_servers.tools.GameSession` objects. No network,
  no event loop — ideal for CI and the autograder (the ``--inprocess`` default).
* :class:`NetworkedBus` — connects over HTTP to the two live FastMCP servers
  using ``fastmcp.Client`` and issues real MCP ``call_tool`` requests.
"""

from __future__ import annotations

import asyncio
from typing import Dict

from core.config import Config
from core.engine import GameEngine
from mcp_servers import tools as T
from mcp_servers.tools import GameSession


class ToolBus:
    """Common interface the loop uses; implemented in-process or over MCP."""

    def call(self, server: str, tool: str, **kwargs) -> dict:  # pragma: no cover
        raise NotImplementedError


class InProcessBus(ToolBus):
    """Calls the shared tool functions directly on two GameSessions."""

    def __init__(self, config: Config):
        self.config = config
        cop_engine = GameEngine(config)
        thief_engine = GameEngine(config)
        self.sessions: Dict[str, GameSession] = {
            "cop": GameSession(config=config, agent="cop", engine=cop_engine),
            "thief": GameSession(config=config, agent="thief", engine=thief_engine),
        }

    def call(self, server: str, tool: str, **kwargs) -> dict:
        session = self.sessions[server]
        fn = getattr(T, tool)
        return fn(session, **kwargs)


class NetworkedBus(ToolBus):
    """Calls tools over HTTP against the two live FastMCP servers."""

    def __init__(self, config: Config):
        from fastmcp import Client
        self.config = config
        cop_url = f"http://{config.mcp.cop.host}:{config.mcp.cop.port}/mcp"
        thief_url = f"http://{config.mcp.thief.host}:{config.mcp.thief.port}/mcp"
        self.clients = {"cop": Client(cop_url), "thief": Client(thief_url)}
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._open())

    async def _open(self):
        for c in self.clients.values():
            await c.__aenter__()

    def call(self, server: str, tool: str, **kwargs) -> dict:
        return self._loop.run_until_complete(self._call(server, tool, **kwargs))

    async def _call(self, server: str, tool: str, **kwargs) -> dict:
        res = await self.clients[server].call_tool(tool, kwargs)
        return res.data if hasattr(res, "data") else res

    def close(self):
        async def _close():
            for c in self.clients.values():
                await c.__aexit__(None, None, None)
        try:
            self._loop.run_until_complete(_close())
        finally:
            self._loop.close()
