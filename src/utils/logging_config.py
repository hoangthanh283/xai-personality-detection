"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_file: str | None = None, level: str = "INFO") -> None:
    """Configure loguru logger."""
    logger.remove()  # Remove default handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=level,
        format=log_format,
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation="50 MB",
            retention="7 days",
            compression="gz",
        )


# Default logging setup
setup_logging()
