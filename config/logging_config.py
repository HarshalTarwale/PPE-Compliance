"""
Logging configuration for the PPE Compliance Detection project.

Provides a ``get_logger`` factory that returns a consistently formatted
logger bound to the calling module name.  Log files are written to
``<project_root>/logs/`` automatically.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import Settings

_INITIALISED = False


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
) -> None:
    """Initialise the root logger with console and file handlers.

    Safe to call multiple times — subsequent calls are no-ops.

    Args:
        level: Logging level (default ``INFO``).
        log_dir: Directory for the log file.  Defaults to
            ``Settings().paths.logs_dir``.
    """
    global _INITIALISED
    if _INITIALISED:
        return

    if log_dir is None:
        log_dir = Settings().paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)

    # File handler
    file_handler = logging.FileHandler(log_dir / "ppe_compliance.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    _INITIALISED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising the logging system if needed.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
    """
    setup_logging()
    return logging.getLogger(name)
