"""Per-run logging: every orchestrator run writes a fresh timestamped file under
artifacts/logs/, in addition to the console — no more manual ``> file.txt``
redirection to keep dialogue evidence.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Tuple


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
