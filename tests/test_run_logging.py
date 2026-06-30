"""Tests for mcp_client/run_logging.py: per-run file + optional console logger."""

import logging
import os

from mcp_client.run_logging import setup_run_logger


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
