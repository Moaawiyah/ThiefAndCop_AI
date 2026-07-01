"""Configuration schema dataclasses for HW6.

Split out of :mod:`core.config` to keep the loader/validator module small. These
typed dataclasses describe every section of ``config.yaml``; the top-level
:class:`Config` (loading + validation) lives in :mod:`core.config` and is
re-exported there for backward compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scoring:
    """Per-sub-game scoring table (assignment §4.4 / Table 1)."""

    cop_win: int = 20
    thief_win: int = 10
    cop_loss: int = 5
    thief_loss: int = 5


@dataclass
class StartConfig:
    """Initial placement rules for the two agents."""

    cop: Any = "random"          # "random" or [row, col]
    thief: Any = "random"        # "random" or [row, col]
    min_initial_distance: int = 3


@dataclass
class LLMConfig:
    """Cloud LLM backend settings (Approach 1 — public GLM API, OpenAI-compatible).

    The API key is read from ``api_key`` below or, when empty, from the
    ``GLM_API_KEY`` environment variable. Keep the key OUT of committed config;
    prefer the env var.
    """

    model: str = "glm-5"
    base_url: str = "https://api.z.ai/api/paas/v4"
    api_key: str = ""
    enabled: bool = False


@dataclass
class MCPEndpoint:
    host: str = "127.0.0.1"
    port: int = 8101


@dataclass
class MCPConfig:
    cop: MCPEndpoint = field(default_factory=MCPEndpoint)
    thief: MCPEndpoint = field(default_factory=lambda: MCPEndpoint(port=8102))
    auth_token: str = "change-me-dev-token"


@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    episodes: int = 20000
    q_dir: str = "artifacts"
    # Cop reward weight for shrinking the thief's reachable free area (barriers).
    confinement_weight: float = 0.15
    # Flat reward added when a barrier actually confines the thief (strategic nudge).
    barrier_bonus: float = 1.0
    # Thief reward weight for keeping its own reachable free area (anti-cornering).
    thief_freedom_weight: float = 0.1
    # Cop may only place a barrier while the thief is currently in vision.
    barrier_requires_visible: bool = True
    # Cost for not changing cell while the opponent is out of vision.
    idle_penalty: float = 0.15
    # Chance an agent is allowed to lurk (stay) on a turn its opponent is unseen.
    blind_stay_prob: float = 0.15
    # Soft training cost for stepping back into a recently-visited cell (anti-loop).
    revisit_penalty: float = 0.15
    # How many recent own-cells count as "recently visited" for the penalty.
    loop_window: int = 3


@dataclass
class ReportConfig:
    email_target: str = "rmisegal+uoh26b@gmail.com"
    timezone: str = "Asia/Jerusalem"
    group_name: str = "Team-Name"
    students: list = field(default_factory=list)
    github_repo: str = ""
    cop_mcp_url: str = ""
    thief_mcp_url: str = ""


@dataclass
class Config:
    """Top-level validated configuration object."""

    grid_size: tuple = (5, 5)
    max_moves: int = 25
    num_games: int = 6
    max_barriers: int = 5
    vision_radius: int = 2
    allow_diagonal: bool = True

    scoring: Scoring = field(default_factory=Scoring)
    start: StartConfig = field(default_factory=StartConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    qlearning: QLearningConfig = field(default_factory=QLearningConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # Resolved absolute path of the directory holding config.yaml.
    base_dir: str = ""

    # ----- derived helpers -------------------------------------------------
    @property
    def rows(self) -> int:
        return int(self.grid_size[0])

    @property
    def cols(self) -> int:
        return int(self.grid_size[1])

    @property
    def num_cells(self) -> int:
        return self.rows * self.cols

    def q_dir_abs(self) -> str:
        """Absolute path to the artifacts/Q-table directory."""
        d = self.qlearning.q_dir
        if not os.path.isabs(d):
            d = os.path.join(self.base_dir, d)
        os.makedirs(d, exist_ok=True)
        return d

    def validate(self) -> "Config":
        """Sanity-check values; raise ValueError on misconfiguration."""
        if self.rows < 1 or self.cols < 1:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")
        if self.max_moves < 1:
            raise ValueError("max_moves must be >= 1")
        if self.num_games < 1:
            raise ValueError("num_games must be >= 1")
        if self.max_barriers < 0:
            raise ValueError("max_barriers must be >= 0")
        if self.vision_radius < 0:
            raise ValueError("vision_radius must be >= 0")
        if not (0 < self.qlearning.learning_rate <= 1):
            raise ValueError("qlearning.learning_rate must be in (0, 1]")
        if not (0 <= self.qlearning.discount_factor <= 1):
            raise ValueError("qlearning.discount_factor must be in [0, 1]")
        max_dist = max(self.rows, self.cols) - 1
        if self.start.min_initial_distance > max_dist:
            # Not fatal: clamp so tiny grids still work (sanity stages).
            self.start.min_initial_distance = max(0, max_dist)
        return self
