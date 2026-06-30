"""MCP Client / orchestrator entry point — the brain that drives the game (§5.2).

This is the component the assignment grades most heavily: an autonomous client
that connects to the **two separate MCP servers** (cop + thief), owns the LLM and
all dialogue/decision logic, and runs the full turn loop by calling MCP **tools**.
The servers never run the LLM (§5.2).

Two execution paths share the exact same loop logic via a small ``ToolBus``
abstraction (see :mod:`mcp_client.bus`):

* **In-process** (``--inprocess``, default for tests): tool calls go straight to
  the shared :mod:`mcp_servers.tools` functions on two ``GameSession`` objects.
  No network, no event loop friction — ideal for CI and the autograder.
* **Networked** (``--networked``): connects over HTTP to the two live FastMCP
  servers using ``fastmcp.Client`` and issues real MCP ``call_tool`` requests.

The :class:`Orchestrator` turn/dialogue logic lives in
:mod:`mcp_client.orchestrator_runner` and is re-exported here so existing imports
(``from orchestrator import Orchestrator``) keep working. This module keeps the
``run(...)`` entry point and the ``--inprocess`` / ``--networked`` CLI.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import load_config
from mcp_client.bus import InProcessBus, NetworkedBus, ToolBus
from mcp_client.orchestrator_runner import Orchestrator  # noqa: F401  (re-export)


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from .env so GLM_API_KEY reaches the LLM client,
    even when the shell hasn't sourced it. Existing env vars win (setdefault)."""
    here = os.path.dirname(os.path.abspath(__file__))
    full = path if os.path.isabs(path) else os.path.join(here, path)
    if not os.path.exists(full):
        return
    with open(full, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def run(networked: bool = False, verbose: bool = True) -> dict:
    """Top-level entry: build a bus + orchestrator and play the series."""
    _load_dotenv()
    config = load_config()
    bus: ToolBus
    if networked:
        bus = NetworkedBus(config)
    else:
        bus = InProcessBus(config)
    orch = Orchestrator(config, bus, verbose=verbose)
    try:
        summary = orch.play_series()
    finally:
        if isinstance(bus, NetworkedBus):
            bus.close()
    return summary


def main():
    p = argparse.ArgumentParser(description="HW6 MCP orchestrator (game client)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--inprocess", action="store_true",
                      help="run tools in-process (no servers needed; for CI/tests)")
    mode.add_argument("--networked", action="store_true",
                      help="connect to the two live FastMCP servers over HTTP (default)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    # Real MCP (networked) is the default; --inprocess opts out.
    run(networked=not args.inprocess, verbose=not args.quiet)


if __name__ == "__main__":
    main()
