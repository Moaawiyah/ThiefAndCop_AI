"""Tests for reporting/email_report.py: real schema build + mocked Gmail send."""

import sys
from unittest.mock import MagicMock

import pytest

import reporting.email_report as er
from core.config import load_config
from reporting.report_schema import validate_internal


def test_sample_results_shape_and_winner_split():
    cfg = load_config()
    results = er._sample_results(cfg)
    assert len(results) == cfg.num_games
    cop_wins = [r for r in results if r["winner"] == "cop"]
    thief_wins = [r for r in results if r["winner"] == "thief"]
    assert len(cop_wins) == 3
    assert len(thief_wins) == cfg.num_games - 3


def test_build_report_dry_run_is_schema_valid():
    cfg = load_config()
    report = er.build_report(cfg, play=False)
    assert validate_internal(report)
    assert report["group_name"] == cfg.report.group_name


def test_build_report_play_uses_orchestrator(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.num_games = 1
    cfg.max_moves = 4
    cfg.llm.enabled = False
    cfg.qlearning.q_dir = str(tmp_path)
    cfg.validate()

    fake_summary = {"results": [{"sub_game": 1, "winner": "cop", "moves": 3,
                                  "cop_score": cfg.scoring.cop_win,
                                  "thief_score": cfg.scoring.thief_loss}]}
    fake_orchestrator = MagicMock()
    fake_orchestrator.run.return_value = fake_summary
    monkeypatch.setitem(sys.modules, "orchestrator", fake_orchestrator)

    report = er.build_report(cfg, play=True)
    assert validate_internal(report)
    assert len(report["sub_games"]) == 1


def test_send_email_calls_gmail_service_and_returns_response(monkeypatch):
    cfg = load_config()
    report = er.build_report(cfg, play=False)

    fake_service = MagicMock()
    fake_service.users.return_value.messages.return_value.send.return_value.execute.return_value = (
        {"id": "fake123"}
    )
    monkeypatch.setattr(er, "_gmail_service", lambda c, t: fake_service)

    resp = er.send_email(cfg, report, credentials_path="unused.json", token_path="unused_token.json")
    assert resp == {"id": "fake123"}
    fake_service.users.return_value.messages.return_value.send.assert_called_once()
    _, kwargs = fake_service.users.return_value.messages.return_value.send.call_args
    assert kwargs["userId"] == "me"
    assert "raw" in kwargs["body"]


def test_main_dry_run_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["email_report.py", "--dry-run"])
    er.main()
    out = capsys.readouterr().out
    assert "Internal Game JSON" in out
    assert "schema valid: True" in out


def test_main_send_path_invokes_send_email(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["email_report.py", "--send"])
    fake_resp = {"id": "abc999"}
    monkeypatch.setattr(er, "send_email", lambda *a, **k: fake_resp)
    er.main()
    out = capsys.readouterr().out
    assert "Sent. Gmail message id: abc999" in out


# --- _gmail_service OAuth paths (the google-auth imports are real deps; we patch
#     the classes they resolve to at call time). ---
def test_gmail_service_uses_valid_cached_token(monkeypatch, tmp_path):
    token = tmp_path / "token.json"
    token.write_text("{}")
    fake_creds = MagicMock(valid=True)
    fake_credentials_cls = MagicMock()
    fake_credentials_cls.from_authorized_user_file.return_value = fake_creds
    monkeypatch.setattr("google.oauth2.credentials.Credentials", fake_credentials_cls)
    monkeypatch.setattr("googleapiclient.discovery.build", lambda *a, **k: "SERVICE")
    assert er._gmail_service("creds.json", str(token)) == "SERVICE"
    fake_credentials_cls.from_authorized_user_file.assert_called_once()


def test_gmail_service_missing_credentials_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("google.oauth2.credentials.Credentials", MagicMock())
    monkeypatch.setattr("google_auth_oauthlib.flow.InstalledAppFlow", MagicMock())
    monkeypatch.setattr("google.auth.transport.requests.Request", MagicMock())
    with pytest.raises(FileNotFoundError):
        er._gmail_service(str(tmp_path / "nope.json"), str(tmp_path / "absent.json"))


def test_gmail_service_runs_oauth_flow_and_caches_token(monkeypatch, tmp_path):
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text("{}")
    token_file = tmp_path / "token.json"  # does not exist yet
    new_creds = MagicMock()
    new_creds.to_json.return_value = "{}"
    flow = MagicMock()
    flow.run_local_server.return_value = new_creds
    fake_flow_cls = MagicMock()
    fake_flow_cls.from_client_secrets_file.return_value = flow
    monkeypatch.setattr("google.oauth2.credentials.Credentials", MagicMock())
    monkeypatch.setattr("google_auth_oauthlib.flow.InstalledAppFlow", fake_flow_cls)
    monkeypatch.setattr("google.auth.transport.requests.Request", MagicMock())
    monkeypatch.setattr("googleapiclient.discovery.build", lambda *a, **k: "SVC")
    assert er._gmail_service(str(creds_file), str(token_file)) == "SVC"
    assert token_file.exists()
    flow.run_local_server.assert_called_once()


def test_gmail_service_refreshes_expired_token(monkeypatch, tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}")
    expired = MagicMock(valid=False, expired=True, refresh_token="r")
    expired.to_json.return_value = "{}"
    fake_credentials_cls = MagicMock()
    fake_credentials_cls.from_authorized_user_file.return_value = expired
    monkeypatch.setattr("google.oauth2.credentials.Credentials", fake_credentials_cls)
    monkeypatch.setattr("google.auth.transport.requests.Request", MagicMock())
    monkeypatch.setattr("googleapiclient.discovery.build", lambda *a, **k: "SVC")
    assert er._gmail_service("creds.json", str(token_file)) == "SVC"
    expired.refresh.assert_called_once()
