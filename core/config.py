"""Configuration loader for HW6 (Cop & Thief MCP pursuit).

All game parameters live in ``config.yaml`` (assignment §10 forbids hard-coding).
This module loads that file into a small set of validated dataclasses so the rest
of the codebase can rely on typed access instead of dict look-ups.

The schema dataclasses live in :mod:`core.config_schema` and are re-exported here
so existing imports (``from core.config import Config, LLMConfig`` ...) keep
working unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import yaml

from .config_schema import (  # noqa: F401  (re-exported for callers/tests)
    Config,
    LLMConfig,
    MCPConfig,
    MCPEndpoint,
    QLearningConfig,
    ReportConfig,
    Scoring,
    StartConfig,
)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


def _coerce_pos(value: Any) -> Any:
    """Normalise a position spec to either the string 'random' or a list."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [int(value[0]), int(value[1])]
    return "random"


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from .env so secrets (e.g. GLM_API_KEY) reach the
    LLM client from any entry point, even when the shell hasn't sourced it.
    Existing env vars win (setdefault)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = path if os.path.isabs(path) else os.path.join(here, path)
    if not os.path.exists(full):
        return
    with open(full, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_config(path: Optional[str] = None) -> Config:
    """Load and validate ``config.yaml`` into a :class:`Config`.

    Parameters
    ----------
    path:
        Optional explicit path. Defaults to ``config.yaml`` next to the repo root.
    """
    _load_dotenv()
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    grid = raw.get("grid_size", [5, 5])
    scoring_raw = raw.get("scoring", {}) or {}
    start_raw = raw.get("start", {}) or {}
    llm_raw = raw.get("llm", {}) or {}
    mcp_raw = raw.get("mcp", {}) or {}
    ql_raw = raw.get("qlearning", {}) or {}
    report_raw = raw.get("report", {}) or {}

    cop_ep = (mcp_raw.get("cop", {}) or {})
    thief_ep = (mcp_raw.get("thief", {}) or {})

    cfg = Config(
        grid_size=(int(grid[0]), int(grid[1])),
        max_moves=int(raw.get("max_moves", 25)),
        num_games=int(raw.get("num_games", 6)),
        max_barriers=int(raw.get("max_barriers", 5)),
        vision_radius=int(raw.get("vision_radius", 2)),
        allow_diagonal=bool(raw.get("allow_diagonal", True)),
        scoring=Scoring(
            cop_win=int(scoring_raw.get("cop_win", 20)),
            thief_win=int(scoring_raw.get("thief_win", 10)),
            cop_loss=int(scoring_raw.get("cop_loss", 5)),
            thief_loss=int(scoring_raw.get("thief_loss", 5)),
        ),
        start=StartConfig(
            cop=_coerce_pos(start_raw.get("cop", "random")),
            thief=_coerce_pos(start_raw.get("thief", "random")),
            min_initial_distance=int(start_raw.get("min_initial_distance", 3)),
        ),
        llm=LLMConfig(
            model=str(llm_raw.get("model", "glm-5")),
            base_url=str(llm_raw.get("base_url", "https://api.z.ai/api/paas/v4")),
            api_key=str(llm_raw.get("api_key", "")),
            enabled=bool(llm_raw.get("enabled", False)),
        ),
        mcp=MCPConfig(
            cop=MCPEndpoint(
                host=str(cop_ep.get("host", "127.0.0.1")),
                port=int(cop_ep.get("port", 8101)),
            ),
            thief=MCPEndpoint(
                host=str(thief_ep.get("host", "127.0.0.1")),
                port=int(thief_ep.get("port", 8102)),
            ),
            auth_token=str(mcp_raw.get("auth_token", "change-me-dev-token")),
        ),
        qlearning=QLearningConfig(
            learning_rate=float(ql_raw.get("learning_rate", 0.1)),
            discount_factor=float(ql_raw.get("discount_factor", 0.9)),
            epsilon_start=float(ql_raw.get("epsilon_start", 1.0)),
            epsilon_min=float(ql_raw.get("epsilon_min", 0.05)),
            epsilon_decay=float(ql_raw.get("epsilon_decay", 0.9995)),
            thief_epsilon_decay=float(
                ql_raw.get("thief_epsilon_decay", ql_raw.get("epsilon_decay", 0.9995))
            ),
            episodes=int(ql_raw.get("episodes", 20000)),
            q_dir=str(ql_raw.get("q_dir", "artifacts")),
            confinement_weight=float(ql_raw.get("confinement_weight", 0.15)),
            barrier_bonus=float(ql_raw.get("barrier_bonus", 1.0)),
            thief_freedom_weight=float(ql_raw.get("thief_freedom_weight", 0.1)),
            barrier_requires_visible=bool(ql_raw.get("barrier_requires_visible", True)),
            idle_penalty=float(ql_raw.get("idle_penalty", 0.15)),
            blind_stay_prob=float(ql_raw.get("blind_stay_prob", 0.15)),
            revisit_penalty=float(ql_raw.get("revisit_penalty", 0.15)),
            loop_window=int(ql_raw.get("loop_window", 3)),
            thief_survive_reward=float(ql_raw.get("thief_survive_reward", 10.0)),
            thief_dist_coef=float(ql_raw.get("thief_dist_coef", 0.1)),
            cop_dist_coef=float(ql_raw.get("cop_dist_coef", 0.1)),
            survive_bonus=float(ql_raw.get("survive_bonus", 0.05)),
            search_move_bonus=float(ql_raw.get("search_move_bonus", 0.05)),
            thief_blind_flee_coef=float(ql_raw.get("thief_blind_flee_coef", 0.0)),
            thief_mobility_coef=float(ql_raw.get("thief_mobility_coef", 0.0)),
            thief_corner_penalty=float(ql_raw.get("thief_corner_penalty", 0.0)),
        ),
        report=ReportConfig(
            email_target=str(report_raw.get("email_target", "rmisegal+uoh26b@gmail.com")),
            timezone=str(report_raw.get("timezone", "Asia/Jerusalem")),
            group_name=str(report_raw.get("group_name", "Team-Name")),
            students=list(report_raw.get("students", []) or []),
            github_repo=str(report_raw.get("github_repo", "")),
            cop_mcp_url=str(report_raw.get("cop_mcp_url", "")),
            thief_mcp_url=str(report_raw.get("thief_mcp_url", "")),
        ),
        base_dir=os.path.dirname(os.path.abspath(path)),
    )
    return cfg.validate()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    c = load_config()
    print(f"Loaded config: grid={c.grid_size} moves={c.max_moves} "
          f"games={c.num_games} barriers={c.max_barriers} "
          f"vision={c.vision_radius} scoring={c.scoring}")
