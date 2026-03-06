"""Lightweight logging helper."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "detectzoo", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a sensible default configuration.

    Calling this multiple times with the same *name* returns the same
    logger instance; the handler is only attached once.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _CONFIGURED = True

    return logger
