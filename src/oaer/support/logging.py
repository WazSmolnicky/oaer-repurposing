"""Logging setup for the pipeline entry points.

Library modules never configure handlers; the console entry points call
``configure_logging`` once and then use ``get_logger``.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO, *, stream: object | None = None) -> None:
    """Attach a single stream handler to the package logger."""

    global _CONFIGURED
    root = logging.getLogger("oaer")
    root.setLevel(level)
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger."""

    return logging.getLogger(name if name.startswith("oaer") else f"oaer.{name}")
