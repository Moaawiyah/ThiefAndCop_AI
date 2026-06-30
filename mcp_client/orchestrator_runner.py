"""The :class:`Orchestrator` — turn-driving / dialogue logic.

Autonomous MCP client that owns the LLM and decision logic, talking to the two
servers only through a :class:`~mcp_client.bus.ToolBus`, and re-exported from
:mod:`orchestrator`. Per turn it exchanges NL messages, picks an action via its
Q/heuristic policy, and submits it to both servers (kept lock-step with a mirror).
"""

from __future__ import annotations

from typing import List

from core.engine import GameEngine
from core.observation import Observation
from agents.policy import build_policy
from llm.glm_client import LLMClient
from mcp_client.bus import ToolBus
from mcp_client.run_logging import setup_run_logger


class Orchestrator:
    """Drives a full 6-sub-game series over a ToolBus."""

    def __init__(self, config, bus: ToolBus, verbose: bool = True):
        self.config = config
        self.bus = bus
        self.token = config.mcp.auth_token
        self.verbose = verbose
        self.logger, self.log_path = setup_run_logger(
            "orchestrator", config.q_dir_abs(), verbose
        )
        self.llm = LLMClient(config.llm)
        # Policies (Q-table if trained, else heuristic) — owned by the client.
        self.cop_policy = build_policy("cop", config)
        self.thief_policy = build_policy("thief", config)
        # A local mirror engine the client uses purely to evaluate policies that
        # need a Grid (legal moves, distances). Kept in lock-step with servers.
        self.mirror = GameEngine(config)

    def log(self, *a):
        self.logger.info(" ".join(str(x) for x in a))

    # ----- low-level helpers ----------------------------------------------
    def _both(self, tool: str, **kwargs) -> dict:
        """Call a mutating tool on BOTH servers; return the cop server's reply."""
        r_cop = self.bus.call("cop", tool, token=self.token, **kwargs)
        self.bus.call("thief", tool, token=self.token, **kwargs)
        return r_cop

    def _obs(self, agent: str) -> Observation:
        """Build the observation from the local mirror (authoritative copy)."""
        return self.mirror.observation_for(agent)

    def _sync_mirror_start(self):
        """Reset the mirror to match the canonical (cop server) start positions."""
        status = self.bus.call("cop", "game_status", token=self.token)
        self.mirror.reset_sub_game()
        self.mirror.state.cop = tuple(status["cop_pos"])
        self.mirror.state.thief = tuple(status["thief_pos"])
        self.mirror.state.move_number = 0
        self.mirror.grid.clear_barriers()

    # ----- turn logic ------------------------------------------------------
    def _exchange_messages(self, agent: str, obs: Observation) -> str:
        """Read opponent message, generate + post this agent's NL message."""
        incoming = self.bus.call(agent, "read_message", token=self.token, reader=agent)
        opp_msg = incoming.get("message", "")
        my_msg = self.llm.generate_message(obs, opp_msg)
        # Post to BOTH mailboxes so either server can serve read_message.
        self._both("send_message", sender=agent, text=my_msg)
        return my_msg

    def _decide_action(self, agent: str, obs: Observation) -> dict:
        policy = self.cop_policy if agent == "cop" else self.thief_policy
        return policy(obs, self.mirror)

    def _apply_action(self, agent: str, action: dict) -> dict:
        if action.get("type") == "barrier":
            # Apply on both servers + mirror.
            self._both("place_barrier", agent=agent)
            self.mirror.apply_cop_action({"type": "barrier"})
            return {"captured": self.mirror.is_capture()}
        act = action.get("action", "stay")
        self._both("submit_move", agent=agent, action=act)
        apply = self.mirror.apply_thief_action if agent == "thief" else self.mirror.apply_cop_action
        apply({"type": "move", "action": act})
        return {"captured": self.mirror.is_capture()}

    def _turn(self, agent: str, move: int) -> bool:
        """Run one agent's full turn; return True if it ended in a capture."""
        obs = self._obs(agent)
        msg = self._exchange_messages(agent, obs)
        action = self._decide_action(agent, obs)
        res = self._apply_action(agent, action)
        label = f"  [{move:>2}] THIEF says:" if agent == "thief" else "       COP   says:"
        self.log(f"{label} \"{msg}\" -> {action}")
        if res["captured"]:
            who = "thief moved onto cop" if agent == "thief" else "cop landed on thief"
            self.log(f"       >> capture! {who} at {self.mirror.state.cop}")
            return True
        return False

    def play_sub_game(self, index: int) -> dict:
        # Reset both servers' sessions and the mirror, with the cop as canonical.
        self._start_sub_game_on_servers()
        self._sync_mirror_start()
        for policy in (self.cop_policy, self.thief_policy):
            if hasattr(policy, "reset_belief"):
                policy.reset_belief()

        cfg = self.config
        winner = "thief"
        move = 0
        self.log(f"\n--- Sub-game {index} (start cop={self.mirror.state.cop} "
                 f"thief={self.mirror.state.thief}) ---")

        while move < cfg.max_moves:
            move += 1
            if self._turn("thief", move):  # thief moves first
                winner = "cop"
                break
            if self._turn("cop", move):
                winner = "cop"
                break
            # Advance move counter on both servers (timeout detection).
            self._both("advance_move_counter")

        if winner == "cop":
            cop_score, thief_score = cfg.scoring.cop_win, cfg.scoring.thief_loss
        else:
            cop_score, thief_score = cfg.scoring.cop_loss, cfg.scoring.thief_win
            self.log(f"  >> thief survived {move} moves -> thief wins")

        return {
            "sub_game": index,
            "winner": winner,
            "moves": move,
            "cop_score": cop_score,
            "thief_score": thief_score,
            "barriers_placed": self.mirror.state.barriers_placed,
        }

    def _start_sub_game_on_servers(self):
        """Start a sub-game on the cop server, then force the thief server to the
        identical canonical start positions via the start_sub_game tool."""
        cop = self.bus.call("cop", "start_sub_game", token=self.token)
        self.bus.call(
            "thief", "start_sub_game", token=self.token,
            cop_row=cop["cop_pos"][0], cop_col=cop["cop_pos"][1],
            thief_row=cop["thief_pos"][0], thief_col=cop["thief_pos"][1],
        )

    def play_series(self) -> dict:
        results: List[dict] = []
        for i in range(1, self.config.num_games + 1):
            results.append(self.play_sub_game(i))
        totals = {
            "cop": sum(r["cop_score"] for r in results),
            "thief": sum(r["thief_score"] for r in results),
        }
        self.log("\n===== SERIES COMPLETE =====")
        for r in results:
            self.log(f"  sub-game {r['sub_game']}: winner={r['winner']:<5} "
                     f"moves={r['moves']:<2} cop={r['cop_score']} thief={r['thief_score']}")
        self.log(f"  TOTALS -> cop={totals['cop']} thief={totals['thief']}")
        self.log(f"  LLM (GLM) active: {self.llm.available} | tokens used: {self.llm.usage}")
        self.log(f"  full run log written to: {self.log_path}")
        return {
            "results": results, "totals": totals,
            "llm_active": self.llm.available, "llm_usage": dict(self.llm.usage),
        }
