"""Logging utilities."""
from __future__ import annotations

import logging
from typing import Optional


def create_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name if name else "muon")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
