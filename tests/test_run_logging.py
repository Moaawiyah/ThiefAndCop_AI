"""Tests for mcp_client/run_logging.py: per-run file + optional console logger."""

import logging
import os

from mcp_client.run_logging import MarkdownLog, setup_run_logger


def test_setup_run_logger_creates_file_and_writes(tmp_path):
    logger, path = setup_run_logger("testrun", str(tmp_path), verbose=False)
    assert os.path.exists(path)
    assert path.startswith(os.path.join(str(tmp_path), "logs"))
    logger.info("hello world")
    for h in logger.handlers:
        h.flush()
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "hello world" in content


def test_setup_run_logger_verbose_adds_console_handler(tmp_path):
    logger, _ = setup_run_logger("testrun2", str(tmp_path), verbose=True)
    kinds = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in kinds or any(
        issubclass(k, logging.StreamHandler) for k in kinds
    )
    assert len(logger.handlers) == 2


def test_setup_run_logger_quiet_has_only_file_handler(tmp_path):
    logger, _ = setup_run_logger("testrun3", str(tmp_path), verbose=False)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.FileHandler)


def test_setup_run_logger_path_uses_name_and_logs_dir(tmp_path):
    logger, path = setup_run_logger("myrun", str(tmp_path), verbose=False)
    assert "myrun_" in os.path.basename(path)
    assert path.endswith(".log")
    assert logger.name.startswith("hw6.myrun.")


def test_markdown_log_writes_transcript_and_results(tmp_path):
    md = MarkdownLog()
    md.sub_game_header(1, (0, 0), (3, 3))
    md.turn(1, "thief", "hi", {"type": "move", "action": "N"})
    md.turn(1, "cop", "hi back", {"type": "move", "action": "S"})
    md.capture("cop landed on thief", (1, 1))
    md.series_complete(
        results=[{"sub_game": 1, "winner": "cop", "moves": 1,
                  "cop_score": 20, "thief_score": 5}],
        totals={"cop": 20, "thief": 5},
        llm_active=False,
        llm_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    path = md.write(str(tmp_path))
    assert path == os.path.join(str(tmp_path), "full_game_log.md")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "# Full Game Log" in content
    assert "Sub-game 1" in content
    assert "THIEF" in content and "COP" in content
    assert "capture!" in content
    assert "Series Complete" in content
    assert "cop=20, thief=5" in content


def test_markdown_log_thief_survived(tmp_path):
    md = MarkdownLog()
    md.thief_survived(12)
    assert "thief survived 12 moves" in md.lines[-1]
