"""Unit tests for config.logging_config module."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import get_logger


class TestLogging:
    """Tests for the logging configuration."""

    def test_get_logger_returns_logger(self) -> None:
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self) -> None:
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_root_has_handlers(self) -> None:
        get_logger("init_check")
        root = logging.getLogger()
        assert len(root.handlers) >= 1
