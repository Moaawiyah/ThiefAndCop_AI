"""Prefect Cloud deployment scaffold for the two MCP servers (assignment §6/§7).

Phase 7 (cloud deploy) is OPTIONAL for this submission and is not exercised by
the local pipeline. This module documents and scaffolds the intended deployment:
each MCP server runs as a Prefect-managed process with a public URL and
token-based auth (the same ``config.mcp.auth_token`` enforced by every tool, so
unauthenticated callers are rejected and access can be revoked by rotating it).

Run locally (smoke):
    python3 deploy/prefect_flow.py --role cop

The ``prefect`` import is guarded so the rest of the repo does not depend on it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from mcp_servers.server_factory import build_server

try:
    from prefect import flow, task
    _PREFECT_OK = True
except Exception:  # pragma: no cover
    _PREFECT_OK = False

    def flow(*a, **k):  # type: ignore
        def deco(fn):
            return fn
        return deco

    def task(*a, **k):  # type: ignore
        def deco(fn):
            return fn
        return deco


@task
def serve_mcp(role: str) -> None:
    """Start one FastMCP server (cop or thief) over HTTP."""
    config = load_config()
    server = build_server(config, role=role)
    ep = config.mcp.cop if role == "cop" else config.mcp.thief
    print(f"[prefect:{role}] serving on http://{ep.host}:{ep.port}/mcp "
          f"(token-gated, rotate config.mcp.auth_token to revoke)")
    server.run(transport="http", host=ep.host, port=ep.port)


@flow(name="hw6-mcp-deploy")
def deploy_flow(role: str = "cop") -> None:
    """Prefect flow wrapping a single MCP server deployment."""
    serve_mcp(role)


def main():
    p = argparse.ArgumentParser(description="Prefect MCP deployment scaffold")
    p.add_argument("--role", choices=["cop", "thief"], default="cop")
    p.add_argument("--serve", action="store_true",
                    help="register a live Prefect Cloud deployment (no work "
                         "pool needed) instead of just running the flow once")
    args = p.parse_args()
    if not _PREFECT_OK:
        print("prefect not installed; running the server without Prefect wrapping.")
        deploy_flow(args.role)
        return
    if args.serve:
        deploy_flow.serve(name=f"{args.role}-mcp-server",
                           parameters={"role": args.role})
    else:
        deploy_flow(args.role)


if __name__ == "__main__":
    main()
