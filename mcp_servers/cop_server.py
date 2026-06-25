"""FastMCP server for the COP agent (assignment §5 — one server per agent).

Runs on its own port (``config.mcp.cop``) over streamable-HTTP so the
orchestrator (MCP client) can connect remotely. Exposes tools only — the LLM is
owned by the client (§5.2).

Usage:
    python3 -m mcp_servers.cop_server
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from mcp_servers.server_factory import build_server


def main() -> None:
    config = load_config()
    server = build_server(config, role="cop")
    ep = config.mcp.cop
    print(f"[cop-server] starting on http://{ep.host}:{ep.port}/mcp")
    server.run(transport="http", host=ep.host, port=ep.port)


if __name__ == "__main__":
    main()
