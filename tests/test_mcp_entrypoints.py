"""Tests for cop_server.py / thief_server.py main() wiring.

These modules just call build_server(...) then block on server.run(...). We let
build_server run for real and only stub out the blocking server.run call (a true
external/blocking boundary), matching the project's stated mocking policy.
"""

from unittest.mock import patch

import mcp_servers.cop_server as cop_server
import mcp_servers.thief_server as thief_server


def test_cop_server_main_builds_and_runs():
    with patch("mcp_servers.server_factory.build_server") as build, \
         patch.object(cop_server, "build_server", build):
        fake_server = build.return_value
        cop_server.main()
        build.assert_called_once()
        args, kwargs = build.call_args
        assert kwargs.get("role", args[1] if len(args) > 1 else None) == "cop"
        fake_server.run.assert_called_once()
        _, run_kwargs = fake_server.run.call_args
        assert run_kwargs["transport"] == "http"


def test_thief_server_main_builds_and_runs():
    with patch("mcp_servers.server_factory.build_server") as build, \
         patch.object(thief_server, "build_server", build):
        fake_server = build.return_value
        thief_server.main()
        build.assert_called_once()
        args, kwargs = build.call_args
        assert kwargs.get("role", args[1] if len(args) > 1 else None) == "thief"
        fake_server.run.assert_called_once()
