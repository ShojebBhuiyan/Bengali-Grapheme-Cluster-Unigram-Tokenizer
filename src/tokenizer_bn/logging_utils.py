"""Structured logging utilities for pipeline steps."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from tokenizer_bn.config import Config, ensure_dirs, load_config

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(step: str, config: Config | None = None) -> logging.Logger:
    """Return a step-scoped logger writing to console and logs/<step>.log."""
    if step in _LOGGERS:
        return _LOGGERS[step]

    cfg = config or load_config()
    ensure_dirs(cfg)

    logger = logging.getLogger(f"tokenizer_bn.{step}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

        log_file = cfg.paths.logs_dir / f"{step}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[step] = logger
    return logger
