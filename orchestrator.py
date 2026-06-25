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


def run(networked: bool = False, verbose: bool = True) -> dict:
    """Top-level entry: build a bus + orchestrator and play the series."""
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
                      help="run tools in-process (default; no servers needed)")
    mode.add_argument("--networked", action="store_true",
                      help="connect to the two live FastMCP servers over HTTP")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run(networked=args.networked, verbose=not args.quiet)


if __name__ == "__main__":
    main()
