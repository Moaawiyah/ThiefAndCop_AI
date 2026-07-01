"""Per-run logging: every orchestrator run writes a fresh timestamped file under
artifacts/logs/, in addition to the console — no more manual ``> file.txt``
redirection to keep dialogue evidence.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple


def setup_run_logger(name: str, artifacts_dir: str, verbose: bool = True) -> Tuple[logging.Logger, str]:
    """Build a logger that always writes to a per-run file, and to the console
    only when ``verbose`` (matching the existing --quiet flag). Returns the
    logger plus the path of the file it's writing to."""
    log_dir = os.path.join(artifacts_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{name}_{stamp}.log")

    logger = logging.getLogger(f"hw6.{name}.{stamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger, path


def log_series_summary(logger, results, totals, llm, log_path,
                       md_path="", gif_path="") -> None:
    """Log the end-of-series scoreboard + written-artifact paths (console + file)."""
    logger.info("\n===== SERIES COMPLETE =====")
    for r in results:
        logger.info(f"  sub-game {r['sub_game']}: winner={r['winner']:<5} "
                    f"moves={r['moves']:<2} cop={r['cop_score']} thief={r['thief_score']}")
    logger.info(f"  TOTALS -> cop={totals['cop']} thief={totals['thief']}")
    logger.info(f"  LLM (GLM) active: {llm.available} | tokens used: {llm.usage}")
    logger.info(f"  full run log written to: {log_path}")
    if md_path:
        logger.info(f"  markdown game log written to: {md_path}")
    if gif_path:
        logger.info(f"  animation written to: {gif_path}")


class MarkdownLog:
    """Accumulates one full series (transcript + final results) as Markdown,
    written once to ``full_game_log.md`` when the run completes — replaces the
    old manual ``> full_game_log.txt`` shell redirection."""

    def __init__(self) -> None:
        self.lines: List[str] = ["# Full Game Log"]

    def sub_game_header(self, index: int, cop, thief) -> None:
        self.lines.append(f"\n## Sub-game {index} (start cop={cop} thief={thief})\n")

    def turn(self, move: int, agent: str, msg: str, action: dict) -> None:
        who = "THIEF" if agent == "thief" else "COP"
        self.lines.append(f"- **[{move}] {who}** says: \"{msg}\" -> `{action}`")

    def capture(self, who: str, pos) -> None:
        self.lines.append(f"- **>> capture!** {who} at {pos}")

    def thief_survived(self, moves: int) -> None:
        self.lines.append(f"- **>> thief survived {moves} moves -> thief wins**")

    def series_complete(self, results: List[dict], totals: Dict[str, int],
                         llm_active: bool, llm_usage: dict) -> None:
        self.lines.append("\n## Series Complete\n")
        self.lines.append("| Sub-game | Winner | Moves | Cop Score | Thief Score |")
        self.lines.append("|---|---|---|---|---|")
        for r in results:
            self.lines.append(f"| {r['sub_game']} | {r['winner']} | {r['moves']} | "
                               f"{r['cop_score']} | {r['thief_score']} |")
        self.lines.append(f"\n**Totals:** cop={totals['cop']}, thief={totals['thief']}\n")
        self.lines.append(f"**LLM active:** {llm_active} | tokens used: {llm_usage}")

    def write(self, out_dir: str) -> str:
        path = os.path.join(out_dir, "full_game_log.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")
        return path
