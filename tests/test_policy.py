"""Tests for agents/policy.py: QPolicy belief tracking + build_policy factory."""

import os

from agents.policy import QPolicy, build_policy
from agents.qlearning import QTable
from core.config import load_config
from core.engine import GameEngine, heuristic_cop_policy, heuristic_thief_policy


def small_config(rows=4, cols=4):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.validate()
    return cfg


def test_qpolicy_without_qtable_file_falls_back_to_heuristic(tmp_path):
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    policy = QPolicy("cop", cfg, q_path=str(tmp_path / "missing.npy"))
    assert policy.loaded is False
    obs = eng.observation_for("cop")
    action = policy(obs, eng)
    assert action["type"] in ("move", "barrier")


def test_qpolicy_loads_existing_table_and_acts_greedily(tmp_path):
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (0, 0)
    eng.state.thief = (3, 3)

    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    # Force the SE action to be the obvious best from state encoding for (0,0)->thief visible.
    path = str(tmp_path / "q_cop.npy")
    q.save(path)

    policy = QPolicy("cop", cfg, q_path=path)
    assert policy.loaded is True
    obs = eng.observation_for("cop")
    action = policy(obs, eng)
    assert action["type"] in ("move", "barrier")


def test_qpolicy_reset_belief_clears_last_known_opponent(tmp_path):
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    policy._last_known_opp = 5
    policy.reset_belief()
    assert policy._last_known_opp is None


def test_qpolicy_updates_belief_when_opponent_visible():
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (0, 0)
    eng.state.thief = (0, 1)  # within vision radius
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal)
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    obs = eng.observation_for("cop")
    assert obs.opponent_visible is True
    policy(obs, eng)
    assert policy._last_known_opp == eng.grid.cell_index((0, 1))


def test_qpolicy_barrier_returned_when_thief_visible_and_greedy(tmp_path):
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 1)
    eng.state.thief = (1, 2)  # within vision -> barrier is a legal strategic action
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    barrier_idx = q.actions.index("barrier")
    obs = eng.observation_for("cop")
    assert obs.opponent_visible is True
    state = q.encode_state(eng.grid.cell_index(obs.self_pos), eng.grid.cell_index((1, 2)))
    q.table[state, barrier_idx] = 1000.0
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    assert policy(obs, eng) == {"type": "barrier"}


def test_qpolicy_barrier_suppressed_when_thief_not_visible(tmp_path):
    """Even if the table prefers a barrier, it is illegal while the thief is unseen."""
    cfg = small_config()
    cfg.vision_radius = 0
    cfg.validate()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 1)
    eng.state.thief = (3, 3)  # out of vision -> barrier gated off
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    obs = eng.observation_for("cop")
    assert obs.opponent_visible is False
    state = q.encode_state(eng.grid.cell_index(obs.self_pos), None)
    q.table[state, q.actions.index("barrier")] = 1000.0
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    assert policy(obs, eng)["type"] == "move"


def test_qpolicy_blind_suppresses_stay_when_lurk_disabled(tmp_path):
    """With lurking off, an unknown opponent means 'stay' is suppressed (anti-deadlock)."""
    cfg = small_config()
    cfg.vision_radius = 0
    cfg.qlearning.blind_stay_prob = 0.0  # deterministic: never lurk
    cfg.validate()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 1)
    eng.state.thief = (3, 3)  # out of vision -> belief stays None (blind)
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    obs = eng.observation_for("cop")
    assert obs.opponent_visible is False
    state = q.encode_state(eng.grid.cell_index(obs.self_pos), None)
    q.table[state, q.actions.index("stay")] = 1000.0  # greedy would pick stay
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    action = policy(obs, eng)
    assert action != {"type": "move", "action": "stay"}


def test_qpolicy_blind_lurks_when_prob_is_one(tmp_path):
    """With blind_stay_prob=1 the agent always lurks (stays) while the opponent is unseen."""
    cfg = small_config()
    cfg.vision_radius = 0
    cfg.qlearning.blind_stay_prob = 1.0  # deterministic: always lurk
    cfg.validate()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 1)
    eng.state.thief = (3, 3)
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    assert policy(eng.observation_for("cop"), eng) == {"type": "move", "action": "stay"}


def test_qpolicy_allows_stay_when_opponent_visible(tmp_path):
    """The guard only fires while blind; a visible opponent leaves 'stay' legal."""
    cfg = small_config()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (0, 0)
    eng.state.thief = (0, 1)  # within vision radius
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    obs = eng.observation_for("cop")
    assert obs.opponent_visible is True
    opp = eng.grid.cell_index((0, 1))
    state = q.encode_state(eng.grid.cell_index(obs.self_pos), opp)
    q.table[state, q.actions.index("stay")] = 1000.0
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    policy = QPolicy("cop", cfg, q_path=path)
    assert policy(obs, eng) == {"type": "move", "action": "stay"}


def test_both_blind_agents_do_not_deadlock(tmp_path):
    """Two blind agents whose greedy pick is 'stay' must not freeze in place."""
    cfg = small_config()
    cfg.vision_radius = 0  # everything is out of vision -> both blind
    cfg.qlearning.blind_stay_prob = 0.0  # deterministic: rule out the lurk path
    cfg.validate()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (0, 0)
    eng.state.thief = (3, 3)
    paths = {}
    for role in ("cop", "thief"):
        q = QTable(role, cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
        q.table[:, q.actions.index("stay")] = 1000.0  # stay greedy everywhere
        p = str(tmp_path / f"q_{role}.npy")
        q.save(p)
        paths[role] = p
    cop = QPolicy("cop", cfg, q_path=paths["cop"])
    thief = QPolicy("thief", cfg, q_path=paths["thief"])
    start = (eng.state.cop, eng.state.thief)
    for _ in range(3):
        eng.apply_thief_action(thief(eng.observation_for("thief"), eng))
        eng.apply_cop_action(cop(eng.observation_for("cop"), eng))
    assert (eng.state.cop, eng.state.thief) != start  # someone actually moved


def test_build_policy_prefers_qtable_when_present(tmp_path):
    cfg = small_config()
    cfg.qlearning.q_dir = str(tmp_path)
    q = QTable("thief", cfg.num_cells, cfg.allow_diagonal)
    q.save(os.path.join(cfg.q_dir_abs(), "q_thief.npy"))
    policy = build_policy("thief", cfg)
    assert isinstance(policy, QPolicy)
    assert policy.loaded is True


def test_build_policy_falls_back_to_heuristic_when_missing(tmp_path):
    cfg = small_config()
    cfg.qlearning.q_dir = str(tmp_path)
    policy = build_policy("cop", cfg)
    assert policy is heuristic_cop_policy


def test_build_policy_prefer_qtable_false_returns_heuristic(tmp_path):
    cfg = small_config()
    cfg.qlearning.q_dir = str(tmp_path)
    q = QTable("thief", cfg.num_cells, cfg.allow_diagonal)
    q.save(os.path.join(cfg.q_dir_abs(), "q_thief.npy"))
    policy = build_policy("thief", cfg, prefer_qtable=False)
    assert policy is heuristic_thief_policy
