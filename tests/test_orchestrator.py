"""Integration tests for mcp_client.orchestrator_runner.Orchestrator + orchestrator.run().

Uses a real InProcessBus with config.llm.enabled = False (deterministic template
fallback, no network). Shrinks num_games/max_moves for speed and redirects
q_dir to tmp_path so nothing is written into the real artifacts/ dir.
"""

import os

import orchestrator as orchestrator_module
from core.config import load_config
from mcp_client.bus import InProcessBus
from mcp_client.orchestrator_runner import Orchestrator


def make_test_config(tmp_path, num_games=2, max_moves=6):
    cfg = load_config()
    cfg.num_games = num_games
    cfg.max_moves = max_moves
    cfg.llm.enabled = False
    cfg.qlearning.q_dir = str(tmp_path)
    cfg.validate()
    return cfg


def test_orchestrator_play_sub_game_returns_expected_keys(tmp_path):
    cfg = make_test_config(tmp_path)
    bus = InProcessBus(cfg)
    orch = Orchestrator(cfg, bus, verbose=False)
    result = orch.play_sub_game(1)
    for key in ("sub_game", "winner", "moves", "cop_score", "thief_score",
                "barriers_placed"):
        assert key in result
    assert result["winner"] in ("cop", "thief")


def test_orchestrator_play_series_completes_and_matches_schema(tmp_path):
    cfg = make_test_config(tmp_path)
    bus = InProcessBus(cfg)
    orch = Orchestrator(cfg, bus, verbose=False)
    summary = orch.play_series()
    assert len(summary["results"]) == cfg.num_games
    assert summary["totals"]["cop"] == sum(r["cop_score"] for r in summary["results"])
    assert summary["totals"]["thief"] == sum(r["thief_score"] for r in summary["results"])
    assert summary["llm_active"] is False
    assert summary["llm_usage"] == {"prompt_tokens": 0, "completion_tokens": 0,
                                     "total_tokens": 0}
    md_path = os.path.join(cfg.q_dir_abs(), "full_game_log.md")
    assert os.path.exists(md_path)
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()
    assert "Series Complete" in content


def test_orchestrator_writes_log_file(tmp_path):
    cfg = make_test_config(tmp_path)
    bus = InProcessBus(cfg)
    orch = Orchestrator(cfg, bus, verbose=False)
    assert os.path.exists(orch.log_path)
    orch.play_sub_game(1)
    for h in orch.logger.handlers:
        h.flush()
    with open(orch.log_path, encoding="utf-8") as fh:
        content = fh.read()
    assert "Sub-game 1" in content


def test_orchestrator_exchange_messages_uses_template_fallback(tmp_path):
    cfg = make_test_config(tmp_path)
    bus = InProcessBus(cfg)
    orch = Orchestrator(cfg, bus, verbose=False)
    orch._start_sub_game_on_servers()
    orch._sync_mirror_start()
    obs = orch._obs("thief")
    msg = orch._exchange_messages("thief", obs)
    assert isinstance(msg, str) and len(msg) > 0


def test_run_inprocess_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(
        orchestrator_module, "load_config",
        lambda: make_test_config(tmp_path, num_games=1, max_moves=4),
    )
    summary = orchestrator_module.run(networked=False, verbose=False)
    assert len(summary["results"]) == 1
    assert "totals" in summary


def test_run_main_inprocess_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        orchestrator_module, "load_config",
        lambda: make_test_config(tmp_path, num_games=1, max_moves=4),
    )
    monkeypatch.setattr("sys.argv",
                        ["orchestrator.py", "--inprocess", "--quiet", "--no-watch"])
    orchestrator_module.main()
