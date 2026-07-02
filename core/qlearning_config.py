"""Q-learning configuration schema for HW6."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    # Thief-only epsilon decay override; defaults to epsilon_decay.
    thief_epsilon_decay: float = 0.9995
    episodes: int = 20000
    q_dir: str = "artifacts"
    # Cop reward weight for shrinking the thief's reachable free area.
    confinement_weight: float = 0.15
    # Flat reward added when a barrier actually confines the thief.
    barrier_bonus: float = 1.0
    # Thief reward weight for keeping reachable free area.
    thief_freedom_weight: float = 0.1
    # Cop may only place a barrier while the thief is currently in vision.
    barrier_requires_visible: bool = True
    # Cost for not changing cell while the opponent is out of vision.
    idle_penalty: float = 0.15
    # Chance an agent is allowed to lurk when its opponent is unseen.
    blind_stay_prob: float = 0.15
    # Soft training cost for stepping back into a recently-visited cell.
    revisit_penalty: float = 0.15
    # Recent own-cells count for the loop penalty.
    loop_window: int = 3
    # Training-only survival terminal, decoupled from Table-1 scoring.
    thief_survive_reward: float = 10.0
    # Role-specific per-step distance-shaping coefficients.
    thief_dist_coef: float = 0.1
    cop_dist_coef: float = 0.1
    # Per-step survival bonus paid to the thief.
    survive_bonus: float = 0.05
    # Reward for real position change while opponent is out of vision.
    search_move_bonus: float = 0.05
    # Blind-evasion reward for opening distance from last-known cop cell.
    thief_blind_flee_coef: float = 0.0
    # Anti-corner reward per open escape route at the thief's new cell.
    thief_mobility_coef: float = 0.0
    # Flat cost for stepping into a corner / low-mobility pocket.
    thief_corner_penalty: float = 0.0
