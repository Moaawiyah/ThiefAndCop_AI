"""Built-in reference policies (sanity checks & Q-Learning baselines).

Split out of :mod:`core.engine`. A policy maps an :class:`~core.observation.Observation`
plus the owning :class:`~core.engine.GameEngine` to an action dict, e.g.
``{"type": "move", "action": "NE"}`` or ``{"type": "barrier"}``.

These deterministic helpers are *not* the graded strategy — they exist so the
same engine can back headless sanity checks and self-play baselines.
"""

from __future__ import annotations

from typing import Callable

from .observation import Observation

# A policy maps an Observation + GameEngine -> an action dict.
Action = dict
Policy = Callable[[Observation, "GameEngine"], Action]  # noqa: F821


def random_policy(obs: Observation, engine) -> Action:
    """Pick a uniformly random legal move."""
    moves = engine.grid.legal_moves(obs.self_pos)
    return {"type": "move", "action": engine.rng.choice(moves)}


def heuristic_cop_policy(obs: Observation, engine) -> Action:
    """Greedy pursuit: move to minimise Chebyshev distance to the thief.

    When the thief is not visible, fall back to a random move (the cop has no
    information). This is a deterministic baseline, not the graded strategy.
    """
    if not obs.opponent_visible or obs.opponent_pos is None:
        return random_policy(obs, engine)
    target = obs.opponent_pos
    best_action, best_dist = "stay", engine.grid.chebyshev(obs.self_pos, target)
    for action in engine.grid.legal_moves(obs.self_pos):
        nxt = engine.grid.apply_move(obs.self_pos, action)
        d = engine.grid.chebyshev(nxt, target)
        if d < best_dist:
            best_dist, best_action = d, action
    return {"type": "move", "action": best_action}


def heuristic_thief_policy(obs: Observation, engine) -> Action:
    """Evade: move to maximise Chebyshev distance from a visible cop."""
    if not obs.opponent_visible or obs.opponent_pos is None:
        return random_policy(obs, engine)
    threat = obs.opponent_pos
    best_action, best_dist = "stay", engine.grid.chebyshev(obs.self_pos, threat)
    for action in engine.grid.legal_moves(obs.self_pos):
        nxt = engine.grid.apply_move(obs.self_pos, action)
        d = engine.grid.chebyshev(nxt, threat)
        if d > best_dist:
            best_dist, best_action = d, action
    return {"type": "move", "action": best_action}
