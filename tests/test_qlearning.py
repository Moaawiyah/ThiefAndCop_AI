"""Tests for the Q-Learning table: Bellman math, encoding, convergence."""

import numpy as np

from agents.qlearning import QTable, action_set


def test_action_sets():
    cop = action_set("cop", allow_diagonal=True)
    thief = action_set("thief", allow_diagonal=True)
    assert "barrier" in cop and "barrier" not in thief
    assert len(cop) == len(thief) + 1
    cop4 = action_set("cop", allow_diagonal=False)
    assert "NE" not in cop4 and "barrier" in cop4


def test_state_encoding_unique_and_bounded():
    q = QTable("thief", num_cells=25)
    seen = set()
    for self_cell in range(25):
        for opp in list(range(25)) + [None]:
            idx = q.encode_state(self_cell, opp)
            assert 0 <= idx < q.num_states
            seen.add(idx)
    # Every (self, opp-incl-unknown) pair is distinct.
    assert len(seen) == 25 * 26


def test_bellman_update_matches_formula():
    q = QTable("thief", num_cells=4, learning_rate=0.1, discount_factor=0.9)
    s, a, ns = 0, 0, 1
    # Seed next-state values so max is known.
    q.table[ns] = np.array([1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    reward = 5.0
    before = q.table[s, a]
    td = q.update(s, a, reward, ns, done=False)

    best_next = np.max(q.table[ns])
    expected_target = reward + 0.9 * best_next
    expected_td = expected_target - before
    expected_value = before + 0.1 * expected_td
    assert np.isclose(td, expected_td)
    assert np.isclose(q.table[s, a], expected_value)


def test_bellman_done_ignores_future():
    q = QTable("thief", num_cells=4, learning_rate=0.5, discount_factor=0.9)
    q.table[1] = np.ones(q.num_actions) * 100  # should be ignored when done
    q.update(0, 0, reward=10.0, next_state=1, done=True)
    # value <- 0 + 0.5 * (10 + 0 - 0) = 5
    assert np.isclose(q.table[0, 0], 5.0)


def test_epsilon_decays_to_floor():
    q = QTable("cop", num_cells=4, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    for _ in range(100):
        q.decay_epsilon()
    assert np.isclose(q.epsilon, 0.1)


def test_greedy_respects_legal_mask():
    q = QTable("thief", num_cells=4)
    q.epsilon = 0.0
    state = q.encode_state(0, 1)
    # Make action 0 the best, but mask it out -> must pick a legal one.
    q.table[state] = np.zeros(q.num_actions)
    q.table[state, 0] = 10.0
    mask = np.ones(q.num_actions, dtype=bool)
    mask[0] = False
    a = q.greedy_action_index(state, mask)
    assert a != 0


def test_converges_on_tiny_corridor():
    """A 1x3 corridor: a cop at cell 0 chasing a fixed thief at cell 2 should
    learn to move toward it. We verify Q-values for the 'toward' action grow."""
    q = QTable("cop", num_cells=3, allow_diagonal=False,
               learning_rate=0.5, discount_factor=0.9, epsilon=0.0)
    # actions for non-diagonal cop: N,S,E,W,stay,barrier ; on a 1-row board only
    # E/W matter. Drive a deterministic toy MDP: state=(self,opp=2).
    e_idx = q.actions.index("E")
    # Reward +10 when reaching cell 2.
    for _ in range(200):
        # at cell0 -> E -> cell1
        s0 = q.encode_state(0, 2)
        s1 = q.encode_state(1, 2)
        q.update(s0, e_idx, 0.0, s1, False)
        # at cell1 -> E -> cell2 (capture)
        q.update(s1, e_idx, 10.0, q.encode_state(2, 2), True)
    s0 = q.encode_state(0, 2)
    s1 = q.encode_state(1, 2)
    # 'E' should be the greedy action from both states.
    assert q.greedy_action_index(s1) == e_idx
    assert q.table[s0, e_idx] > 0
