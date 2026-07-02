"""Trained-vs-baseline and trained-vs-trained evaluation (assignment §8).

Split out of :mod:`agents.train_utils` to keep that module within the
project's per-file line budget.
"""

from __future__ import annotations

import random as _random

from core.engine import (
    GameEngine, heuristic_cop_policy, heuristic_thief_policy, random_policy,
)


def evaluate(config, cop_q_path: str, thief_q_path: str = None,
            n: int = 200, seed: int = 123) -> dict:
    """Trained-vs-baseline and trained-vs-trained capture-rate comparisons.

    Reports the trained cop vs. a random cop (both vs. a heuristic thief), the
    trained thief's survival vs. a heuristic cop, and — the real fairness
    signal for this assignment — the self-play capture rate when both trained
    Q-tables face each other directly.
    """
    from agents.policy import QPolicy  # local: avoids a policy<->turn import cycle

    def capture_rate(cop_policy, thief_policy):
        eng = GameEngine(config, rng=_random.Random(seed))
        captures = 0
        for _ in range(n):
            res = eng.play_sub_game(cop_policy, thief_policy)
            if res.winner == "cop":
                captures += 1
        return captures / n

    trained_cop = QPolicy("cop", config, q_path=cop_q_path)
    trained_thief = QPolicy("thief", config, q_path=thief_q_path) if thief_q_path else None

    trained_rate = capture_rate(trained_cop, heuristic_thief_policy) if trained_cop.loaded else None
    random_rate = capture_rate(random_policy, heuristic_thief_policy)
    thief_ok = trained_thief is not None and trained_thief.loaded
    thief_survival_rate = (1.0 - capture_rate(heuristic_cop_policy, trained_thief)) if thief_ok else None
    self_play_rate = (capture_rate(trained_cop, trained_thief)
                      if trained_cop.loaded and thief_ok else None)
    return {
        "trained_capture_rate": trained_rate,
        "random_capture_rate": random_rate,
        "thief_survival_rate": thief_survival_rate,
        "self_play_capture_rate": self_play_rate,
    }
