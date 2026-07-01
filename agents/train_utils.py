"""Helpers for Q-Learning self-play training (assignment §8 / Phase 4).

Split out of :mod:`agents.train` to keep the trainer's CLI + loop small. Contains
the legal-action mask, the action applier, the learning-curve writers (CSV + PNG)
and the trained-vs-random evaluation used to confirm learning.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from core.engine import GameEngine, heuristic_thief_policy, random_policy
from agents.qlearning import QTable

# Headless-safe matplotlib backend.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Local shaping constants (search encouragement / thief survival), kept beside
# the distance coefficient below rather than in config — they are fixed nudges.
SEARCH_MOVE_BONUS = 0.05
DIST_COEF = 0.1
SURVIVE_BONUS = 0.05


def legal_mask(qtable: QTable, engine: GameEngine, self_pos, barriers_left: int,
               opp_visible: bool = True, require_visible: bool = True) -> np.ndarray:
    """Boolean mask of legal action indices for ``qtable`` at ``self_pos``.

    A cop may place a barrier only when it still has barriers, is not already on
    one, and — when ``require_visible`` — the opponent is currently in vision
    (``opp_visible``). This keeps barriers a strategic response to a seen thief
    rather than a blind move.
    """
    mask = np.zeros(qtable.num_actions, dtype=bool)
    legal_moves = set(engine.grid.legal_moves(self_pos))
    for i, a in enumerate(qtable.actions):
        if a == "barrier":
            mask[i] = (
                qtable.role == "cop"
                and barriers_left > 0
                and not engine.grid.is_barrier(self_pos)
                and (opp_visible or not require_visible)
            )
        else:
            mask[i] = a in legal_moves
    if not mask.any():
        mask[:] = True
    return mask


def idle_or_move(moved: bool, idle_penalty: float) -> float:
    """Blind-phase shaping keyed on *actual* movement, not the action label.

    Rewards any real position change (search) and penalises standing still, so a
    wall-bump that leaves the agent in place is penalised just like ``stay``.
    """
    return SEARCH_MOVE_BONUS if moved else -idle_penalty


def thief_step_reward(visible, dist_before, dist_after, free_before, free_after,
                      moved, revisited, free_w, idle_penalty, revisit_penalty) -> float:
    """Per-step thief reward: flee when the cop is visible, search when blind.

    Rewards keeping open escape space plus a survival bonus, penalises idling
    while blind, and applies a soft cost for stepping back into a recently-visited
    cell so the thief flees instead of ping-ponging (loops are discouraged, not
    forbidden).
    """
    motion = (DIST_COEF * (dist_after - dist_before) if visible
              else idle_or_move(moved, idle_penalty))
    revisit = revisit_penalty if revisited else 0.0
    return motion + free_w * (free_after - free_before) + SURVIVE_BONUS - revisit


def cop_step_reward(visible, dist_before, dist_after, free_before, free_after,
                    action, moved, revisited, conf_w, idle_penalty, barrier_bonus,
                    revisit_penalty) -> float:
    """Per-step cop reward: closes distance + confines when the thief is visible.

    A ``barrier`` that shrinks the thief's reachable area earns the confinement
    credit plus a flat ``barrier_bonus`` (a strategic nudge, since a barrier costs
    the cop its turn); one that confines nothing is wasted and penalised like
    idling. While blind the cop is rewarded for moving (search) and penalised for
    standing still. A move back into a recently-visited cell pays a soft revisit
    penalty (loops discouraged, not forbidden).
    """
    if action == "barrier":
        if free_after < free_before:
            return conf_w * (free_before - free_after) + barrier_bonus
        return -idle_penalty
    revisit = revisit_penalty if revisited else 0.0
    if not visible:
        return idle_or_move(moved, idle_penalty) - revisit
    return DIST_COEF * (dist_before - dist_after) + conf_w * (free_before - free_after) - revisit


def apply_action(engine: GameEngine, role: str, action_label: str) -> None:
    """Apply a Q-table action label to ``engine`` for ``role``."""
    if action_label == "barrier":
        engine.apply_cop_action({"type": "barrier"})
    elif role == "cop":
        engine.apply_cop_action({"type": "move", "action": action_label})
    else:
        engine.apply_thief_action({"type": "move", "action": action_label})


def write_curve(history, q_dir: str):
    """Write CSV + PNG learning curves."""
    csv_path = os.path.join(q_dir, "learning_curve.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "cop_reward", "thief_reward", "captured", "epsilon"])
        w.writerows(history)

    arr = np.array(history, dtype=float)
    eps_ax = arr[:, 0]
    cop_r = arr[:, 1]
    captured = arr[:, 3]
    eps = arr[:, 4]

    def moving_avg(x, w=None):
        if w is None:
            w = max(1, len(x) // 50)
        if len(x) < w or w < 1:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ma = moving_avg(cop_r)
    cap_ma = moving_avg(captured)
    ax1.plot(eps_ax[-len(ma):], ma, color="tab:blue", label="cop reward (MA)")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("cop episode reward (moving avg)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(eps_ax[-len(cap_ma):], cap_ma, color="tab:green", label="capture rate (MA)")
    ax2.plot(eps_ax, eps, color="tab:red", alpha=0.4, label="epsilon")
    ax2.set_ylabel("capture rate / epsilon", color="tab:green")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="tab:green")

    fig.suptitle("Q-Learning self-play — learning curves")
    fig.tight_layout()
    png_path = os.path.join(q_dir, "learning_curve.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return csv_path, png_path


def evaluate(config, cop_q_path: str, n: int = 200, seed: int = 123) -> dict:
    """Compare a trained cop vs a random cop, both chasing a heuristic thief.

    Returns capture rates so we can confirm the trained cop beats the baseline.
    """
    import random as _random

    from agents.policy import QPolicy  # local: avoids a policy<->turn import cycle

    def run(cop_policy, label):
        eng = GameEngine(config, rng=_random.Random(seed))
        captures = 0
        for _ in range(n):
            res = eng.play_sub_game(cop_policy, heuristic_thief_policy)
            if res.winner == "cop":
                captures += 1
        return captures / n

    trained = QPolicy("cop", config, q_path=cop_q_path)
    trained_rate = run(trained, "trained") if trained.loaded else None
    random_rate = run(random_policy, "random")
    return {"trained_capture_rate": trained_rate, "random_capture_rate": random_rate}
