"""Series/sub-game methods for the MCP orchestrator."""

from __future__ import annotations

from typing import List

from mcp_client.run_logging import log_series_summary


class OrchestratorSeriesMixin:
    """Mixin holding the longer orchestration run-loop methods."""

    def play_sub_game(self, index: int) -> dict:
        self._start_sub_game_on_servers()
        self._sync_mirror_start()
        for policy in (self.cop_policy, self.thief_policy):
            if hasattr(policy, "reset_belief"):
                policy.reset_belief()

        cfg = self.config
        winner = "thief"
        move = 0
        self.gif.start_sub_game(index)
        self.log(f"\n--- Sub-game {index} (start cop={self.mirror.state.cop} "
                 f"thief={self.mirror.state.thief}) ---")
        self.md.sub_game_header(index, self.mirror.state.cop, self.mirror.state.thief)

        while move < cfg.max_moves:
            move += 1
            if self._turn("thief", move):
                winner = "cop"
                break
            if self._turn("cop", move):
                winner = "cop"
                break
            self._both("advance_move_counter")

        if winner == "cop":
            cop_score, thief_score = cfg.scoring.cop_win, cfg.scoring.thief_loss
        else:
            cop_score, thief_score = cfg.scoring.cop_loss, cfg.scoring.thief_win
            self.log(f"  >> thief survived {move} moves -> thief wins")
            self.md.thief_survived(move)

        self.gif.add_score(cop_score, thief_score)
        return {
            "sub_game": index,
            "winner": winner,
            "moves": move,
            "cop_score": cop_score,
            "thief_score": thief_score,
            "barriers_placed": self.mirror.state.barriers_placed,
        }

    def _start_sub_game_on_servers(self):
        """Start both servers with identical canonical positions."""
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
        gif_path = self.gif.save()
        self.md.series_complete(results, totals, self.llm.available, dict(self.llm.usage))
        md_path = self.md.write(self.config.q_dir_abs())
        log_series_summary(self.logger, results, totals, self.llm,
                           self.log_path, md_path, gif_path)
        return {
            "results": results,
            "totals": totals,
            "llm_active": self.llm.available,
            "llm_usage": dict(self.llm.usage),
        }
