"""Sub-game and series runners for :class:`core.engine.GameEngine`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from .policies import Policy
from .state import Position, SubGameResult

if TYPE_CHECKING:  # pragma: no cover
    from .engine import GameEngine


def play_sub_game(
    engine: "GameEngine",
    cop_policy: Policy,
    thief_policy: Policy,
    index: int = 0,
    on_step: Optional[Callable[["GameEngine"], None]] = None,
) -> SubGameResult:
    """Play one full sub-game and return its scored result."""
    engine.reset_sub_game()
    s = engine.state
    cfg = engine.config
    winner = "thief"
    capture_pos: Optional[Position] = None

    while s.move_number < cfg.max_moves:
        s.move_number += 1
        thief_act = thief_policy(engine.observation_for("thief"), engine)
        engine.apply_thief_action(thief_act)
        if engine.is_capture():
            winner = "cop"
            capture_pos = s.cop
            if on_step:
                on_step(engine)
            break

        cop_act = cop_policy(engine.observation_for("cop"), engine)
        engine.apply_cop_action(cop_act)
        if engine.is_capture():
            winner = "cop"
            capture_pos = s.cop
            if on_step:
                on_step(engine)
            break

        if on_step:
            on_step(engine)

    if winner == "cop":
        cop_score = cfg.scoring.cop_win
        thief_score = cfg.scoring.thief_loss
    else:
        cop_score = cfg.scoring.cop_loss
        thief_score = cfg.scoring.thief_win
    return SubGameResult(
        index=index, winner=winner, moves=s.move_number,
        cop_score=cop_score, thief_score=thief_score,
        capture_pos=capture_pos, barriers_placed=s.barriers_placed,
    )


def play_series(
    engine: "GameEngine",
    cop_policy: Policy,
    thief_policy: Policy,
    on_step: Optional[Callable[["GameEngine"], None]] = None,
    on_sub_game: Optional[Callable[[SubGameResult], None]] = None,
) -> Tuple[List[SubGameResult], dict]:
    """Play ``num_games`` sub-games; return results and totals."""
    results: List[SubGameResult] = []
    for i in range(engine.config.num_games):
        res = play_sub_game(engine, cop_policy, thief_policy, index=i + 1, on_step=on_step)
        results.append(res)
        if on_sub_game:
            on_sub_game(res)
    totals = {
        "cop": sum(r.cop_score for r in results),
        "thief": sum(r.thief_score for r in results),
    }
    return results, totals
