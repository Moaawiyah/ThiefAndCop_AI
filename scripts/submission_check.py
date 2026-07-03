"""Submission-readiness checks for HW6.

Run before final delivery:
    python scripts/submission_check.py

Use ``--skip-url-check`` for an offline metadata/token-only check.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config, load_config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


PLACEHOLDER_GROUPS = {"", "Team-Name", "team-name", "TODO", "TBD"}
PLACEHOLDER_TOKENS = {"", "change-me-dev-token", "dev-token", "token"}


def _present(value: str) -> bool:
    return bool(str(value).strip())


def metadata_checks(config: Config) -> list[Check]:
    """Validate non-network submission fields."""
    report = config.report
    return [
        Check(
            "group_name",
            report.group_name not in PLACEHOLDER_GROUPS,
            f"group_name={report.group_name!r}",
        ),
        Check(
            "students",
            bool(report.students),
            f"students={report.students!r}",
        ),
        Check(
            "github_repo",
            _present(report.github_repo),
            f"github_repo={report.github_repo!r}",
        ),
        Check(
            "cop_mcp_url",
            _present(report.cop_mcp_url),
            f"cop_mcp_url={report.cop_mcp_url!r}",
        ),
        Check(
            "thief_mcp_url",
            _present(report.thief_mcp_url),
            f"thief_mcp_url={report.thief_mcp_url!r}",
        ),
        Check(
            "mcp_auth_token",
            config.mcp.auth_token not in PLACEHOLDER_TOKENS,
            "token is set" if config.mcp.auth_token else "token is empty",
        ),
    ]


async def _probe_mcp_url(name: str, url: str, timeout: float) -> Check:
    """Connect to one MCP URL and list tools."""
    if not _present(url):
        return Check(name, False, "missing URL")
    try:
        from fastmcp import Client

        async with Client(url) as client:
            tools = await asyncio.wait_for(client.list_tools(), timeout=timeout)
        names = sorted(t.name for t in tools)
        return Check(name, True, f"{len(names)} tools: {', '.join(names[:4])}")
    except Exception as exc:
        return Check(name, False, f"{type(exc).__name__}: {exc}")


async def url_checks(config: Config, timeout: float = 10.0) -> list[Check]:
    """Validate that configured public MCP URLs are reachable."""
    report = config.report
    return await asyncio.gather(
        _probe_mcp_url("cop_mcp_reachable", report.cop_mcp_url, timeout),
        _probe_mcp_url("thief_mcp_reachable", report.thief_mcp_url, timeout),
    )


async def run_checks(config: Config, check_urls: bool = True) -> list[Check]:
    checks = metadata_checks(config)
    if check_urls:
        checks.extend(await url_checks(config))
    return checks


def print_report(checks: list[Check]) -> None:
    for check in checks:
        mark = "PASS" if check.ok else "FAIL"
        print(f"[{mark}] {check.name}: {check.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="HW6 submission readiness check")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--skip-url-check", action="store_true")
    args = parser.parse_args()

    checks = asyncio.run(run_checks(load_config(args.config), not args.skip_url_check))
    print_report(checks)
    failures = [c for c in checks if not c.ok]
    if failures:
        print(f"\n{len(failures)} submission check(s) failed.")
        return 1
    print("\nAll submission checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
