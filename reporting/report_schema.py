"""JSON report builders (assignment §9.1 and §9.2).

* :func:`build_internal_game_report` — the **Internal Game JSON** the cop emails
  after the 6 sub-games (group metadata, GitHub repo, the two MCP URLs, timezone,
  per-sub-game results, and totals).
* :func:`build_bonus_game_report` — the **Inter-Group Bonus JSON** for the
  optional cross-group competition (§9.2).

Both produce plain dicts (and a JSON string helper) so the email body can be
*JSON only* with no free text (§9).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from core.config import Config


def _sub_games_payload(results: List[dict]) -> List[dict]:
    """Normalise engine/orchestrator sub-game results into report records."""
    payload = []
    for r in results:
        payload.append({
            "sub_game": r.get("sub_game", r.get("index")),
            "winner": r["winner"],
            "moves": r["moves"],
            "cop_score": r["cop_score"],
            "thief_score": r["thief_score"],
        })
    return payload


def build_internal_game_report(
    config: Config,
    results: List[dict],
    totals: Optional[Dict[str, int]] = None,
) -> dict:
    """Internal Game JSON (§9.1)."""
    if totals is None:
        totals = {
            "cop": sum(r["cop_score"] for r in results),
            "thief": sum(r["thief_score"] for r in results),
        }
    rep = config.report
    return {
        "group_name": rep.group_name,
        "students": list(rep.students),
        "github_repo": rep.github_repo,
        "cop_mcp_url": rep.cop_mcp_url,
        "thief_mcp_url": rep.thief_mcp_url,
        "timezone": rep.timezone,
        "sub_games": _sub_games_payload(results),
        "totals": {"cop": int(totals["cop"]), "thief": int(totals["thief"])},
    }


def build_bonus_game_report(
    config: Config,
    group_1: str,
    group_2: str,
    github_repo_group_1: str,
    github_repo_group_2: str,
    mcp_urls: Dict[str, str],
    students_group_1: List[str],
    students_group_2: List[str],
    results: List[dict],
    totals_by_group: Dict[str, int],
    bonus_claim: Dict[str, int],
    mutual_agreement: bool,
) -> dict:
    """Inter-Group Bonus JSON (§9.2).

    ``mcp_urls`` keys: mcp_url_group_1_cop, mcp_url_group_1_thief,
    mcp_url_group_2_cop, mcp_url_group_2_thief.
    """
    return {
        "report_type": "bonus_game",
        "groups": {"group_1": group_1, "group_2": group_2},
        "github_repo_group_1": github_repo_group_1,
        "github_repo_group_2": github_repo_group_2,
        "mcp_url_group_1_cop": mcp_urls.get("mcp_url_group_1_cop", ""),
        "mcp_url_group_1_thief": mcp_urls.get("mcp_url_group_1_thief", ""),
        "mcp_url_group_2_cop": mcp_urls.get("mcp_url_group_2_cop", ""),
        "mcp_url_group_2_thief": mcp_urls.get("mcp_url_group_2_thief", ""),
        "timezone": config.report.timezone,
        "students_group_1": list(students_group_1),
        "students_group_2": list(students_group_2),
        "sub_games": _sub_games_payload(results),
        "totals_by_group": dict(totals_by_group),
        "bonus_claim": dict(bonus_claim),
        "mutual_agreement": bool(mutual_agreement),
    }


def to_json(report: dict) -> str:
    """Serialise a report dict to a stable, pretty JSON string."""
    return json.dumps(report, indent=2, ensure_ascii=False)


# Required top-level keys for validation/tests.
INTERNAL_REQUIRED_KEYS = {
    "group_name", "students", "github_repo", "cop_mcp_url", "thief_mcp_url",
    "timezone", "sub_games", "totals",
}
BONUS_REQUIRED_KEYS = {
    "report_type", "groups", "github_repo_group_1", "github_repo_group_2",
    "mcp_url_group_1_cop", "mcp_url_group_1_thief", "mcp_url_group_2_cop",
    "mcp_url_group_2_thief", "timezone", "students_group_1", "students_group_2",
    "sub_games", "totals_by_group", "bonus_claim", "mutual_agreement",
}


def validate_internal(report: dict) -> bool:
    """Return True iff ``report`` has all required Internal Game JSON keys."""
    if not INTERNAL_REQUIRED_KEYS.issubset(report.keys()):
        return False
    if not isinstance(report["totals"], dict):
        return False
    if {"cop", "thief"} - report["totals"].keys():
        return False
    return isinstance(report["sub_games"], list)


def validate_bonus(report: dict) -> bool:
    """Return True iff ``report`` has all required Bonus Game JSON keys."""
    return BONUS_REQUIRED_KEYS.issubset(report.keys())
