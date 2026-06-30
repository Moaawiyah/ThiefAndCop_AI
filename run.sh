#!/usr/bin/env bash
# Convenience launcher: loads secrets from .env, then runs the orchestrator.
# Usage:  ./run.sh            (networked series, verbose)
#         ./run.sh --inprocess
#         ./run.sh --quiet
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
exec uv run python3 orchestrator.py "$@"
