"""Logging helpers for the research pipeline."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure a simple process-wide logging format if none exists."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level.upper())
        return

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
