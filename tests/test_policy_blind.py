"""Tests for agents/policy.py: blind-mode conduct (barriers, lurk, anti-deadlock)."""

from agents.policy import QPolicy
from agents.qlearning import QTable
from core.config import load_config
from core.engine import GameEngine


def small_config(rows=4, cols=4):
    cfg = load_config()
    cfg.grid_size = (rows, cols)
    cfg.validate()
    return cfg


def _blind_barrier_policy(tmp_path, requires_visible):
    cfg = small_config()
    cfg.vision_radius = 0
    cfg.qlearning.barrier_requires_visible = requires_visible
    cfg.validate()
    eng = GameEngine(cfg)
    eng.reset_sub_game()
    eng.state.cop = (1, 1)
    eng.state.thief = (3, 3)  # out of vision
    q = QTable("cop", cfg.num_cells, cfg.allow_diagonal, epsilon=0.0)
    obs = eng.observation_for("cop")
    assert obs.opponent_visible is False
    state = q.encode_state(eng.grid.cell_index(obs.self_pos), None)
    q.table[state, q.actions.index("barrier")] = 1000.0
    path = str(tmp_path / "q_cop.npy")
    q.save(path)
    return QPolicy("cop", cfg, q_path=path), obs, eng


def test_qpolicy_barrier_suppressed_when_blind_and_gate_requires_visible(tmp_path):
    """With barrier_requires_visible on, a blind barrier is illegal -> cop moves."""
    policy, obs, eng = _blind_barrier_policy(tmp_path, requires_visible=True)
    assert policy(obs, eng)["type"] == "move"


def test_qpolicy_allows_blind_barrier_when_gate_off(tmp_path):
    """With the gate off, a blind cop may wall off a corridor during search."""
    policy, obs, eng = _blind_barrier_policy(tmp_path, requires_visible=False)
    assert policy(obs, eng) == {"type": "barrier"}


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
