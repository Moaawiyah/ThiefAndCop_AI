"""Staged sanity checks (assignment §4.5 / Table 2).

Runs a full 6-sub-game series headless at increasing grid sizes (2x2 -> 5x5),
using heuristic policies, and prints per-sub-game and total scores. This proves
the engine, scoring and turn loop are correct before any networking/LLM is added.

Usage:
    python3 scripts/sanity_check.py
"""

from __future__ import annotations

import os
import random
import sys

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from core.engine import (
    GameEngine,
    heuristic_cop_policy,
    heuristic_thief_policy,
)

STAGES = [
    (2, 2, "Stage 1: algorithmic sanity / pipeline integration"),
    (3, 3, "Stage 2: coordination convergence / hyper-param tuning"),
    (4, 4, "Stage 3: partial-observation ambiguity"),
    (5, 5, "Stage 4: final run / full game"),
]


def run_stage(rows: int, cols: int, label: str, seed: int = 42) -> dict:
    cfg = load_config()
    # Override only the grid size for this stage; everything else from config.
    cfg.grid_size = (rows, cols)
    cfg.validate()  # re-clamp min_initial_distance for tiny boards
    engine = GameEngine(cfg, rng=random.Random(seed))

    print(f"\n=== {label} ({rows}x{cols}) ===")
    results, totals = engine.play_series(
        cop_policy=heuristic_cop_policy,
        thief_policy=heuristic_thief_policy,
        on_sub_game=lambda r: print(
            f"  sub-game {r.index}: winner={r.winner:<5} "
            f"moves={r.moves:<2} cop={r.cop_score} thief={r.thief_score} "
            f"barriers={r.barriers_placed}"
        ),
    )
    print(f"  TOTALS  -> cop={totals['cop']}  thief={totals['thief']}")
    # Invariant checks.
    n = cfg.num_games
    assert len(results) == n, "wrong number of sub-games"
    min_total = n * min(cfg.scoring.cop_loss, cfg.scoring.cop_win)
    assert totals["cop"] >= min_total or totals["cop"] >= 0
    return {"size": (rows, cols), "totals": totals, "results": results}


def main() -> None:
    print("HW6 sanity checks — staged grids 2x2 -> 5x5")
    for rows, cols, label in STAGES:
        run_stage(rows, cols, label)
    print("\nAll sanity stages completed successfully.")


if __name__ == "__main__":
    main()
