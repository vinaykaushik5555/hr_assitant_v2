from __future__ import annotations

import logging
import sys
from typing import Optional


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure application-wide logging once.

    Subsequent calls are ignored to avoid duplicate handlers.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True
