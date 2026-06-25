"""Self-play Q-Learning training (assignment §8 / Phase 4).

Trains a cop Q-table and a thief Q-table simultaneously via self-play on the
configured grid, writing:

    artifacts/q_cop.npy
    artifacts/q_thief.npy
    artifacts/learning_curve.csv      (episode, cop_reward, thief_reward, capture_rate, epsilon)
    artifacts/learning_curve.png      (matplotlib plot of the moving averages)

Reward shaping (sparse terminal + small distance shaping so learning is feasible
on the small grids of this assignment):
    * Capture: cop +scoring.cop_win, thief -scoring.cop_win  (terminal)
    * Timeout: cop -scoring.thief_win, thief +scoring.thief_win (terminal)
    * Per step: cop gets a small positive reward for decreasing Chebyshev
      distance to the thief; thief gets the mirror. Plus a tiny time penalty for
      the cop (encourages fast capture) and a tiny survival bonus for the thief.

Usage:
    python3 agents/train.py                 # uses config.qlearning.episodes
    python3 agents/train.py --episodes 2000 # smoke test
    python3 agents/train.py --eval-only     # only run the random-baseline eval

The curve/eval/action helpers live in :mod:`agents.train_utils`.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.config import load_config
from core.engine import GameEngine
from agents.qlearning import QTable
from agents.train_utils import apply_action, evaluate, legal_mask, write_curve


def train(config, episodes: int, seed: int = 0):
    """Run self-play training and return (cop_q, thief_q, history)."""
    rng = np.random.default_rng(seed)
    engine = GameEngine(config)

    ql = config.qlearning
    cop_q = QTable("cop", config.num_cells, config.allow_diagonal,
                   ql.learning_rate, ql.discount_factor, ql.epsilon_start,
                   ql.epsilon_min, ql.epsilon_decay, rng=rng)
    thief_q = QTable("thief", config.num_cells, config.allow_diagonal,
                     ql.learning_rate, ql.discount_factor, ql.epsilon_start,
                     ql.epsilon_min, ql.epsilon_decay, rng=rng)

    history = []  # (episode, cop_reward, thief_reward, captured, epsilon)
    cap_win = config.scoring.cop_win
    surv_win = config.scoring.thief_win

    for ep in range(episodes):
        engine.reset_sub_game()
        s = engine.state
        cop_total = thief_total = 0.0
        captured = False
        # Belief = last-known opponent cell for each agent.
        cop_belief = thief_belief = None

        while s.move_number < config.max_moves:
            s.move_number += 1
            cell = engine.grid.cell_index

            # ---- Thief turn (moves first) ----
            t_self = cell(s.thief)
            t_visible = engine.grid.chebyshev(s.thief, s.cop) <= config.vision_radius
            if t_visible:
                thief_belief = cell(s.cop)
            t_state = thief_q.encode_state(t_self, thief_belief)
            t_mask = legal_mask(thief_q, engine, s.thief, 0)
            t_aidx = thief_q.select_action_index(t_state, t_mask)
            dist_before = engine.grid.chebyshev(s.thief, s.cop)
            apply_action(engine, "thief", thief_q.actions[t_aidx])

            if engine.is_capture():
                # thief stepped onto cop
                r_thief = -cap_win
                r_cop = cap_win
                t_next = thief_q.encode_state(cell(s.thief), thief_belief)
                thief_q.update(t_state, t_aidx, r_thief, t_next, True)
                cop_total += r_cop
                thief_total += r_thief
                captured = True
                break
            dist_after = engine.grid.chebyshev(s.thief, s.cop)
            # Shaping: thief rewarded for increasing distance + survival.
            r_thief = 0.1 * (dist_after - dist_before) + 0.05
            t_next = thief_q.encode_state(cell(s.thief), thief_belief)
            thief_q.update(t_state, t_aidx, r_thief, t_next, False)
            thief_total += r_thief

            # ---- Cop turn ----
            c_self = cell(s.cop)
            c_visible = engine.grid.chebyshev(s.cop, s.thief) <= config.vision_radius
            if c_visible:
                cop_belief = cell(s.thief)
            c_state = cop_q.encode_state(c_self, cop_belief)
            barriers_left = config.max_barriers - s.barriers_placed
            c_mask = legal_mask(cop_q, engine, s.cop, barriers_left)
            c_aidx = cop_q.select_action_index(c_state, c_mask)
            dist_before = engine.grid.chebyshev(s.cop, s.thief)
            apply_action(engine, "cop", cop_q.actions[c_aidx])

            if engine.is_capture():
                r_cop = cap_win
                c_next = cop_q.encode_state(cell(s.cop), cop_belief)
                cop_q.update(c_state, c_aidx, r_cop, c_next, True)
                cop_total += r_cop
                captured = True
                break
            dist_after = engine.grid.chebyshev(s.cop, s.thief)
            # Shaping: cop rewarded for decreasing distance, small time penalty.
            r_cop = 0.1 * (dist_before - dist_after) - 0.05
            c_next = cop_q.encode_state(cell(s.cop), cop_belief)
            cop_q.update(c_state, c_aidx, r_cop, c_next, False)
            cop_total += r_cop

        if not captured:
            # Timeout terminal rewards.
            cop_total += -surv_win
            thief_total += surv_win

        cop_q.decay_epsilon()
        thief_q.decay_epsilon()
        history.append((ep, cop_total, thief_total, int(captured), cop_q.epsilon))

    return cop_q, thief_q, history


def main():
    parser = argparse.ArgumentParser(description="Q-Learning self-play trainer")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override config.qlearning.episodes (smoke tests)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-n", type=int, default=200)
    args = parser.parse_args()

    config = load_config()
    q_dir = config.q_dir_abs()
    cop_path = os.path.join(q_dir, "q_cop.npy")
    thief_path = os.path.join(q_dir, "q_thief.npy")

    if not args.eval_only:
        episodes = args.episodes or config.qlearning.episodes
        print(f"Training self-play Q-Learning: episodes={episodes} grid={config.grid_size}")
        cop_q, thief_q, history = train(config, episodes, seed=args.seed)
        cop_q.save(cop_path)
        thief_q.save(thief_path)
        csv_path, png_path = write_curve(history, q_dir)
        print(f"Saved Q-tables: {cop_path}, {thief_path}")
        print(f"Learning curve: {csv_path}, {png_path}")
        final_cap = np.mean([h[3] for h in history[-max(1, len(history)//10):]])
        print(f"Final 10% capture rate during training: {final_cap:.3f}")

    print("Evaluating trained cop vs random baseline "
          f"({args.eval_n} sub-games each, heuristic thief)...")
    metrics = evaluate(config, cop_path, n=args.eval_n, seed=args.seed + 7)
    print(f"  trained cop capture rate: {metrics['trained_capture_rate']}")
    print(f"  random  cop capture rate: {metrics['random_capture_rate']}")
    if metrics["trained_capture_rate"] is not None:
        if metrics["trained_capture_rate"] >= metrics["random_capture_rate"]:
            print("  RESULT: trained cop >= random baseline. OK")
        else:
            print("  RESULT: trained cop did NOT beat baseline (try more episodes).")


if __name__ == "__main__":
    main()
