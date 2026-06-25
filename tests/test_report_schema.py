"""Tests for the JSON report builders (§9.1 / §9.2)."""

import json

from core.config import load_config
from reporting.report_schema import (
    build_bonus_game_report,
    build_internal_game_report,
    to_json,
    validate_bonus,
    validate_internal,
)


def sample_results(cfg):
    return [
        {"sub_game": 1, "winner": "cop", "moves": 5,
         "cop_score": cfg.scoring.cop_win, "thief_score": cfg.scoring.thief_loss},
        {"sub_game": 2, "winner": "thief", "moves": cfg.max_moves,
         "cop_score": cfg.scoring.cop_loss, "thief_score": cfg.scoring.thief_win},
    ]


def test_internal_report_keys_and_totals():
    cfg = load_config()
    results = sample_results(cfg)
    rep = build_internal_game_report(cfg, results)
    assert validate_internal(rep)
    # Totals computed correctly.
    assert rep["totals"]["cop"] == cfg.scoring.cop_win + cfg.scoring.cop_loss
    assert rep["totals"]["thief"] == cfg.scoring.thief_loss + cfg.scoring.thief_win
    assert rep["timezone"] == cfg.report.timezone
    assert len(rep["sub_games"]) == 2


def test_internal_report_is_valid_json():
    cfg = load_config()
    rep = build_internal_game_report(cfg, sample_results(cfg))
    s = to_json(rep)
    parsed = json.loads(s)  # round-trips
    assert parsed["group_name"] == cfg.report.group_name


def test_internal_report_explicit_totals_override():
    cfg = load_config()
    rep = build_internal_game_report(cfg, sample_results(cfg), totals={"cop": 90, "thief": 40})
    assert rep["totals"] == {"cop": 90, "thief": 40}


def test_bonus_report_schema():
    cfg = load_config()
    rep = build_bonus_game_report(
        cfg,
        group_1="Team-Alpha", group_2="Team-Beta",
        github_repo_group_1="https://github.com/a/repo",
        github_repo_group_2="https://github.com/b/repo",
        mcp_urls={
            "mcp_url_group_1_cop": "https://cop-a.run",
            "mcp_url_group_1_thief": "https://thief-a.run",
            "mcp_url_group_2_cop": "https://cop-b.run",
            "mcp_url_group_2_thief": "https://thief-b.run",
        },
        students_group_1=["s1"], students_group_2=["s2"],
        results=sample_results(cfg),
        totals_by_group={"Team-Alpha": 60, "Team-Beta": 80},
        bonus_claim={"Team-Alpha": 7, "Team-Beta": 10},
        mutual_agreement=True,
    )
    assert validate_bonus(rep)
    assert rep["report_type"] == "bonus_game"
    assert json.loads(to_json(rep))["mutual_agreement"] is True


def test_validate_rejects_incomplete():
    assert not validate_internal({"group_name": "x"})
    assert not validate_bonus({"report_type": "bonus_game"})
