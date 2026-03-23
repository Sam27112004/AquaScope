"""
Loguru-based logging configuration for Aquascope.

Import :func:`get_logger` everywhere instead of using the stdlib ``logging``
module directly.  The root logger is configured once on first import.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

_CONFIGURED = False


def _configure_logger(log_dir: str | Path | None = None, level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    _loguru_logger.remove()

    # Colourised console output.
    _loguru_logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )

    # Optional file sink.
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            Path(log_dir) / "aquascope_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation="10 MB",
            retention="14 days",
            compression="zip",
        )

    _CONFIGURED = True


_configure_logger()


def get_logger(name: str):
    """Return a Loguru logger bound to *name*.

    Usage::

        logger = get_logger(__name__)
        logger.info("Hello from %s", __name__)
    """
    return _loguru_logger.bind(name=name)
