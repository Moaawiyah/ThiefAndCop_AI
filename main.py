"""End-to-end HW6 pipeline: train -> play series -> email report.

Runs the three phases in order, reusing the existing entry points:

  1. **Train** — Q-Learning self-play (:func:`agents.train.train`); writes
     ``artifacts/q_cop.npy`` / ``q_thief.npy`` + learning curves.
  2. **Orchestrate** — the MCP client plays a 6-sub-game series with the freshly
     trained policies (:func:`orchestrator.run`); writes the game log + GIF.
  3. **Report** — build the JSON-only Internal Game Report and email it via the
     Gmail API (:func:`reporting.email_report`). Dry-run by default.

Usage:
    python3 main.py                       # train + orchestrate + email (dry-run)
    python3 main.py --episodes 2000       # quick smoke run
    python3 main.py --send                # actually deliver the email
    python3 main.py --skip-train          # reuse existing Q-tables
    python3 main.py --skip-orchestrate    # no live series (email uses sample data)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import load_config
from agents.train import train
from agents.train_utils import write_curve
from orchestrator import run as run_orchestrator
from reporting.email_report import build_report, send_email
from reporting.report_schema import (
    build_internal_game_report, to_json, validate_internal,
)


def phase_train(config, episodes: int, seed: int) -> None:
    """Train both Q-tables and persist them + the learning curves."""
    q_dir = config.q_dir_abs()
    print(f"[1/3] Training self-play Q-Learning: episodes={episodes} grid={config.grid_size}")
    cop_q, thief_q, history = train(config, episodes, seed=seed)
    cop_q.save(os.path.join(q_dir, "q_cop.npy"))
    thief_q.save(os.path.join(q_dir, "q_thief.npy"))
    write_curve(history, q_dir)
    print(f"      saved Q-tables + learning curves to {q_dir}")


def phase_orchestrate(networked: bool, record_gif: bool, verbose: bool = True) -> dict:
    """Play a full series with the trained policies (writes log + optional GIF).

    ``verbose`` prints each turn's natural-language dialogue to the terminal (the
    live game), matching ``orchestrator.py``'s default; pass ``--quiet`` to mute.
    """
    print(f"[2/3] Playing MCP series ({'networked' if networked else 'in-process'})...")
    summary = run_orchestrator(networked=networked, verbose=verbose,
                               record_gif=record_gif, serve_live=False)
    print(f"      series complete — totals: {summary.get('totals', {})}")
    return summary


def phase_email(config, results, send: bool) -> None:
    """Build the JSON report and either send it or print it (dry-run).

    ``results`` are the sub-game results from the orchestrated series when it
    ran (so the report reflects the real game); otherwise the report falls back
    to sample data — no second series is played.
    """
    print("[3/3] Building Internal Game JSON report...")
    report = (build_internal_game_report(config, results) if results is not None
              else build_report(config, play=False))
    assert validate_internal(report), "report failed schema validation"
    if send:
        resp = send_email(config, report)
        print(f"      sent to {config.report.email_target} — id: {resp.get('id')}")
    else:
        print(to_json(report))
        print(f"      [dry-run] would send to: {config.report.email_target}")


def main() -> None:
    p = argparse.ArgumentParser(description="HW6 end-to-end pipeline (train -> play -> email)")
    p.add_argument("--episodes", type=int, default=None,
                   help="override config.qlearning.episodes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--send", action="store_true", help="actually send the email (Gmail API)")
    p.add_argument("--networked", action="store_true",
                   help="orchestrate over the live FastMCP servers (default: in-process)")
    p.add_argument("--no-gif", action="store_true", help="skip writing artifacts/game_full.gif")
    p.add_argument("--quiet", action="store_true", help="mute the per-turn game dialogue printout")
    p.add_argument("--skip-train", action="store_true", help="reuse existing Q-tables")
    p.add_argument("--skip-orchestrate", action="store_true",
                   help="skip the live series (the report then uses sample data)")
    p.add_argument("--skip-email", action="store_true")
    args = p.parse_args()

    config = load_config()
    if not args.skip_train:
        episodes = args.episodes or config.qlearning.episodes
        phase_train(config, episodes, args.seed)
    summary = None
    if not args.skip_orchestrate:
        summary = phase_orchestrate(args.networked, not args.no_gif, verbose=not args.quiet)
    if not args.skip_email:
        results = summary["results"] if summary else None
        phase_email(config, results, args.send)


if __name__ == "__main__":
    main()
