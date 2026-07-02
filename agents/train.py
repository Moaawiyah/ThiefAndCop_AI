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
    * Timeout: cop -thief_survive_reward, thief +thief_survive_reward (terminal;
      decoupled from the report's Table-1 thief_win so the survival learning signal
      can match the cop's capture signal)
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
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.config import load_config
from core.engine import GameEngine
from agents.qlearning import QTable
from agents.train_utils import evaluate, write_curve
from agents.turn import act_agent


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
                     ql.epsilon_min, ql.thief_epsilon_decay, rng=rng)

    history = []  # (episode, cop_reward, thief_reward, captured, epsilon)
    cap_win = config.scoring.cop_win
    # Training-only survival terminal, decoupled from the report's Table-1 score:
    # lets the thief's learning signal match the cop's capture signal.
    surv_win = config.qlearning.thief_survive_reward
    win = config.qlearning.loop_window

    for ep in range(episodes):
        engine.reset_sub_game()
        s = engine.state
        cop_total = thief_total = 0.0
        captured = False
        # Belief = last-known opponent cell; history = recent own cells (anti-loop).
        cop_belief = thief_belief = None
        cop_hist: deque = deque(maxlen=win)
        thief_hist: deque = deque(maxlen=win)

        while s.move_number < config.max_moves:
            s.move_number += 1
            last_step = s.move_number >= config.max_moves

            # Thief moves first; its learning is DEFERRED until the cop responds
            # so the thief is penalised terminally when the cop pounces (not only
            # when the thief itself walks onto the cop).
            tt = act_agent(engine, "thief", thief_q, thief_belief, thief_hist, config)
            thief_belief = tt.belief
            if tt.captured:  # thief walked onto the cop -> terminal for the thief
                thief_q.update(tt.state, tt.aidx, tt.reward, tt.next_state, True)
                thief_total += tt.reward
                cop_total += cap_win
                captured = True
                break

            ct = act_agent(engine, "cop", cop_q, cop_belief, cop_hist, config)
            cop_belief = ct.belief
            if ct.captured:  # cop pounced -> the thief's move was fatal
                cop_q.update(ct.state, ct.aidx, ct.reward, ct.next_state, True)
                thief_q.update(tt.state, tt.aidx, -cap_win, tt.next_state, True)
                cop_total += ct.reward
                thief_total += -cap_win
                captured = True
                break

            if last_step:
                # Timeout terminal: thief survives (+surv_win), cop times out (-surv_win).
                thief_q.update(tt.state, tt.aidx, surv_win, tt.next_state, True)
                cop_q.update(ct.state, ct.aidx, -surv_win, ct.next_state, True)
                thief_total += surv_win
                cop_total += -surv_win
            else:
                thief_q.update(tt.state, tt.aidx, tt.reward, tt.next_state, False)
                cop_q.update(ct.state, ct.aidx, ct.reward, ct.next_state, False)
                thief_total += tt.reward
                cop_total += ct.reward

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
    metrics = evaluate(config, cop_path, thief_path, n=args.eval_n, seed=args.seed + 7)
    print(f"  trained cop capture rate:      {metrics['trained_capture_rate']}")
    print(f"  random  cop capture rate:      {metrics['random_capture_rate']}")
    print(f"  trained thief survival rate:   {metrics['thief_survival_rate']}")
    print(f"  self-play (cop vs thief) rate: {metrics['self_play_capture_rate']}")
    if metrics["trained_capture_rate"] is not None:
        if metrics["trained_capture_rate"] >= metrics["random_capture_rate"]:
            print("  RESULT: trained cop >= random baseline. OK")
        else:
            print("  RESULT: trained cop did NOT beat baseline (try more episodes).")


if __name__ == "__main__":
    main()
