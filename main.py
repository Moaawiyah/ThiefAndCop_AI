"""One-shot pipeline entry point: regenerate every submission artifact.

Runs the four artifact-producing stages in order and writes their outputs under
``artifacts/`` (``config.q_dir``):

  train   -> q_cop.npy, q_thief.npy, learning_curve.csv/.png   (agents/train.py)
  gui     -> gui_*.png headless screenshots                    (gui/visualizer.py)
  play    -> orchestrator_*.log + live LLM dialogue            (orchestrator.py)
  report  -> report_internal.json (schema-valid Internal JSON) (reporting/...)

The ``.env`` file is loaded first so ``GLM_API_KEY`` reaches the LLM client and
the played series really runs over the LLM (``LLM (GLM) active: True``).

    uv run python3 main.py                  # all stages, in-process series
    uv run python3 main.py --networked      # play over the two live MCP servers
    uv run python3 main.py --episodes 2000  # faster training smoke run
    uv run python3 main.py --only play      # run a single stage (repeatable)
    uv run python3 main.py --no-train       # skip the slow training stage
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orchestrator
from agents.train import train
from agents.train_utils import evaluate, write_curve
from core.config import load_config
from gui.visualizer import Visualizer
from reporting.email_report import build_report
from reporting.report_schema import to_json, validate_internal


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from .env so GLM_API_KEY reaches the LLM client."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def stage_train(config, episodes: int) -> None:
    q_dir = config.q_dir_abs()
    cop_path = os.path.join(q_dir, "q_cop.npy")
    thief_path = os.path.join(q_dir, "q_thief.npy")
    print(f"[train] self-play Q-Learning: episodes={episodes} grid={config.grid_size}")
    cop_q, thief_q, history = train(config, episodes, seed=0)
    cop_q.save(cop_path)
    thief_q.save(thief_path)
    csv_path, png_path = write_curve(history, q_dir)
    print(f"[train] wrote {cop_path}, {thief_path}, {csv_path}, {png_path}")
    metrics = evaluate(config, cop_path, n=200, seed=7)
    print(f"[train] trained capture={metrics['trained_capture_rate']} "
          f"random={metrics['random_capture_rate']}")


def stage_gui(config, shots: int) -> None:
    print(f"[gui] rendering {shots} headless screenshots -> {config.q_dir_abs()}")
    Visualizer(config, headless=True, screenshots=shots).run()


def stage_play(networked: bool) -> dict:
    mode = "networked" if networked else "in-process"
    print(f"[play] full series ({mode}); LLM active when GLM_API_KEY is set")
    summary = orchestrator.run(networked=networked, verbose=True)
    print(f"[play] totals={summary['totals']} llm_active={summary['llm_active']} "
          f"tokens={summary['llm_usage']['total_tokens']}")
    return summary


def stage_report(config) -> None:
    report = build_report(config, play=True)
    out = os.path.join(config.q_dir_abs(), "report_internal.json")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(to_json(report))
    print(f"[report] schema valid={validate_internal(report)} -> {out}")


STAGES = ("train", "gui", "play", "report")


def main() -> None:
    parser = argparse.ArgumentParser(description="HW6 one-shot artifact pipeline")
    parser.add_argument("--networked", action="store_true",
                        help="play over the two live MCP servers (default: in-process)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override config.qlearning.episodes for training")
    parser.add_argument("--only", choices=STAGES, action="append",
                        help="run only the named stage(s); repeatable")
    parser.add_argument("--no-train", action="store_true",
                        help="skip the slow training stage")
    args = parser.parse_args()

    load_dotenv()
    config = load_config()
    stages = args.only or [s for s in STAGES
                           if not (args.no_train and s == "train")]

    if "train" in stages:
        stage_train(config, args.episodes or config.qlearning.episodes)
    if "gui" in stages:
        stage_gui(config, shots=5)
    if "play" in stages:
        stage_play(args.networked)
    if "report" in stages:
        stage_report(config)
    print("[done] artifacts written under", config.q_dir_abs())


if __name__ == "__main__":
    main()
