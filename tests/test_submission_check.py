"""Tests for scripts/submission_check.py."""

import asyncio

from core.config_schema import Config, MCPConfig, ReportConfig
from scripts.submission_check import metadata_checks, run_checks


def test_metadata_checks_reject_placeholders():
    cfg = Config(
        report=ReportConfig(
            group_name="Team-Name",
            students=[],
            github_repo="",
            cop_mcp_url="https://cop.example/mcp",
            thief_mcp_url="https://thief.example/mcp",
        ),
        mcp=MCPConfig(auth_token="change-me-dev-token"),
    )

    checks = {c.name: c.ok for c in metadata_checks(cfg)}

    assert checks["group_name"] is False
    assert checks["students"] is False
    assert checks["github_repo"] is False
    assert checks["mcp_auth_token"] is False
    assert checks["cop_mcp_url"] is True
    assert checks["thief_mcp_url"] is True


def test_metadata_checks_accept_real_values():
    cfg = Config(
        report=ReportConfig(
            group_name="Team Alpha",
            students=["Alice", "Bob"],
            github_repo="https://github.com/example/repo",
            cop_mcp_url="https://cop.example/mcp",
            thief_mcp_url="https://thief.example/mcp",
        ),
        mcp=MCPConfig(auth_token="real-token-value"),
    )

    assert all(c.ok for c in metadata_checks(cfg))


def test_run_checks_can_skip_network():
    cfg = Config(
        report=ReportConfig(
            group_name="Team Alpha",
            students=["Alice"],
            github_repo="https://github.com/example/repo",
            cop_mcp_url="https://cop.example/mcp",
            thief_mcp_url="https://thief.example/mcp",
        ),
        mcp=MCPConfig(auth_token="real-token-value"),
    )

    checks = asyncio.run(run_checks(cfg, check_urls=False))

    assert all(c.ok for c in checks)
    assert "cop_mcp_reachable" not in {c.name for c in checks}
