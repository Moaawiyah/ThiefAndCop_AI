"""FastMCP server for the THIEF agent (assignment §5 — one server per agent).

Runs on its own port (``config.mcp.thief``) over streamable-HTTP. Exposes tools
only; the LLM is owned by the orchestrator/client (§5.2).

Usage:
    python3 -m mcp_servers.thief_server
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from mcp_servers.server_factory import build_server


def main() -> None:
    config = load_config()
    server = build_server(config, role="thief")
    ep = config.mcp.thief
    print(f"[thief-server] starting on http://{ep.host}:{ep.port}/mcp")
    server.run(transport="http", host=ep.host, port=ep.port)


if __name__ == "__main__":
    main()
