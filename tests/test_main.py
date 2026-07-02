"""Tests for main.py: the train -> orchestrate -> email pipeline wiring."""

import sys

import main as main_mod
from core.config import load_config


def test_phase_email_dry_run_prints_target(capsys):
    cfg = load_config()
    main_mod.phase_email(cfg, results=None, send=False)
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert cfg.report.email_target in out


def test_phase_email_uses_provided_results(capsys):
    cfg = load_config()
    results = [{"sub_game": 1, "winner": "cop", "moves": 5,
                "cop_score": cfg.scoring.cop_win, "thief_score": cfg.scoring.thief_loss}]
    main_mod.phase_email(cfg, results=results, send=False)
    out = capsys.readouterr().out
    assert '"sub_games"' in out  # report built from the passed results


def test_main_runs_phases_in_order(monkeypatch):
    cfg = load_config()
    calls = []
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(main_mod, "phase_train", lambda *a, **k: calls.append("train"))
    monkeypatch.setattr(
        main_mod, "phase_orchestrate",
        lambda *a, **k: (calls.append("orch"), {"results": [], "totals": {}})[1])
    monkeypatch.setattr(main_mod, "phase_email",
                        lambda *a, **k: calls.append("email"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--episodes", "5"])
    main_mod.main()
    assert calls == ["train", "orch", "email"]


def test_main_skip_flags_run_only_email(monkeypatch):
    cfg = load_config()
    calls = []
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(main_mod, "phase_train", lambda *a, **k: calls.append("train"))
    monkeypatch.setattr(main_mod, "phase_orchestrate", lambda *a, **k: calls.append("orch"))
    monkeypatch.setattr(main_mod, "phase_email", lambda *a, **k: calls.append("email"))
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--skip-train", "--skip-orchestrate"])
    main_mod.main()
    assert calls == ["email"]
