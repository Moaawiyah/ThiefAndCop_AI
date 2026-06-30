"""Tests for deploy/prefect_flow.py: wiring only, never starting a real server."""

import sys
from unittest.mock import MagicMock, patch

import deploy.prefect_flow as pf


def test_prefect_is_importable_and_flagged_ok():
    assert pf._PREFECT_OK is True


def test_serve_mcp_builds_server_for_role_and_calls_run():
    fake_server = MagicMock()
    with patch.object(pf, "build_server", return_value=fake_server) as build, \
         patch.object(pf, "load_config", wraps=pf.load_config) as load_cfg:
        pf.serve_mcp.fn("cop")
        build.assert_called_once()
        args, kwargs = build.call_args
        assert kwargs.get("role", args[-1]) == "cop"
        fake_server.run.assert_called_once()
        _, run_kwargs = fake_server.run.call_args
        assert run_kwargs["transport"] == "http"
        assert load_cfg.called


def test_serve_mcp_thief_role_uses_thief_endpoint():
    fake_server = MagicMock()
    with patch.object(pf, "build_server", return_value=fake_server):
        cfg = pf.load_config()
        pf.serve_mcp.fn("thief")
        _, run_kwargs = fake_server.run.call_args
        assert run_kwargs["port"] == cfg.mcp.thief.port


def test_deploy_flow_calls_serve_mcp():
    with patch.object(pf, "serve_mcp") as serve:
        pf.deploy_flow.fn("cop")
        serve.fn.assert_not_called()  # ensure we patched the task wrapper, not .fn
        serve.assert_called_once_with("cop")


def test_main_serve_flag_calls_flow_serve(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prefect_flow.py", "--role", "cop", "--serve"])
    with patch.object(pf.deploy_flow, "serve") as serve_method:
        pf.main()
        serve_method.assert_called_once()
        _, kwargs = serve_method.call_args
        assert kwargs["parameters"] == {"role": "cop"}


def test_main_without_serve_calls_deploy_flow_directly(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prefect_flow.py", "--role", "thief"])
    with patch.object(pf, "deploy_flow") as deploy:
        pf.main()
        deploy.assert_called_once_with("thief")


def test_main_without_prefect_runs_deploy_flow_unwrapped(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prefect_flow.py"])
    monkeypatch.setattr(pf, "_PREFECT_OK", False)
    with patch.object(pf, "deploy_flow") as deploy:
        pf.main()
        deploy.assert_called_once_with("cop")
