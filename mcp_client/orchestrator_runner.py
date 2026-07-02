"""Turn-driving dialogue logic for the MCP client orchestrator."""

from __future__ import annotations

from core.engine import GameEngine
from core.observation import Observation
from agents.policy import build_policy
from llm.glm_client import LLMClient
from gui.series_gif import SeriesGif
from mcp_client.bus import ToolBus
from mcp_client.orchestrator_series import OrchestratorSeriesMixin
from mcp_client.run_logging import MarkdownLog, setup_run_logger


class Orchestrator(OrchestratorSeriesMixin):
    """Drives a full six-sub-game series over a ToolBus."""

    def __init__(self, config, bus: ToolBus, verbose: bool = True,
                 record_gif: bool = False, serve_live: bool = False,
                 host: str = "127.0.0.1", port: int = 8000):
        self.config = config
        self.bus = bus
        self.token = config.mcp.auth_token
        self.verbose = verbose
        self.logger, self.log_path = setup_run_logger(
            "orchestrator", config.q_dir_abs(), verbose
        )
        self.md = MarkdownLog()
        self.gif = SeriesGif(config, record_gif, serve=serve_live, host=host, port=port)
        self.llm = LLMClient(config.llm)
        self.cop_policy = build_policy("cop", config)
        self.thief_policy = build_policy("thief", config)
        self.mirror = GameEngine(config)

    def log(self, *a):
        self.logger.info(" ".join(str(x) for x in a))

    def _both(self, tool: str, **kwargs) -> dict:
        """Call a mutating tool on both servers; return the cop reply."""
        r_cop = self.bus.call("cop", tool, token=self.token, **kwargs)
        self.bus.call("thief", tool, token=self.token, **kwargs)
        return r_cop

    def _obs(self, agent: str) -> Observation:
        """Build the observation from the local mirror."""
        return self.mirror.observation_for(agent)

    def _sync_mirror_start(self):
        """Reset the mirror to match the canonical cop-server start."""
        status = self.bus.call("cop", "game_status", token=self.token)
        self.mirror.reset_sub_game()
        self.mirror.state.cop = tuple(status["cop_pos"])
        self.mirror.state.thief = tuple(status["thief_pos"])
        self.mirror.state.move_number = 0

    def _exchange_messages(self, agent: str, obs: Observation) -> str:
        """Read opponent message, generate and post this agent's message."""
        incoming = self.bus.call(agent, "read_message", token=self.token, reader=agent)
        opp_msg = incoming.get("message", "")
        my_msg = self.llm.generate_message(obs, opp_msg)
        self._both("send_message", sender=agent, text=my_msg)
        return my_msg

    def _decide_action(self, agent: str, obs: Observation) -> dict:
        policy = self.cop_policy if agent == "cop" else self.thief_policy
        return policy(obs, self.mirror)

    def _apply_action(self, agent: str, action: dict) -> dict:
        if action.get("type") == "barrier":
            self._both("place_barrier", agent=agent)
            self.mirror.apply_cop_action({"type": "barrier"})
            return {"captured": self.mirror.is_capture()}
        act = action.get("action", "stay")
        self._both("submit_move", agent=agent, action=act)
        apply = self.mirror.apply_thief_action if agent == "thief" else self.mirror.apply_cop_action
        apply({"type": "move", "action": act})
        return {"captured": self.mirror.is_capture()}

    def _turn(self, agent: str, move: int) -> bool:
        """Run one agent turn; return True if it ended in capture."""
        self.mirror.state.move_number = move
        obs = self._obs(agent)
        msg = self._exchange_messages(agent, obs)
        action = self._decide_action(agent, obs)
        res = self._apply_action(agent, action)
        label = f"  [{move:>2}] THIEF says:" if agent == "thief" else "       COP   says:"
        self.log(f"{label} \"{msg}\" -> {action}")
        self.md.turn(move, agent, msg, action)
        self.gif.turn(agent, msg, self.mirror, self.llm.available)
        if res["captured"]:
            who = "thief moved onto cop" if agent == "thief" else "cop landed on thief"
            self.log(f"       >> capture! {who} at {self.mirror.state.cop}")
            self.md.capture(who, self.mirror.state.cop)
            return True
        return False
