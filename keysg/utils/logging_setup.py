"""
Centralized logging setup for the KeySG project.

Goals:
- Use Loguru consistently across modules.
- Default to console-only logging (no files).
- Allow optional file logging via env var KeySG_LOG_FILE if needed.
- Allow log levels to be provided by the Hydra pipeline configuration.
"""

from __future__ import annotations

import os
from loguru import logger


def setup_logging(
    console_level: str = "INFO",
    file_level: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure Loguru sinks in a minimal, consistent way.

    - Remove pre-existing handlers to avoid duplicates when re-running.
    - Add a single console sink, INFO by default.
    - If env var KeySG_LOG_FILE is set to a filepath, also log to that file.
    - Use config-provided levels for console and file sinks.
    """
    try:
        logger.remove()
    except Exception:
        # If loguru hasn't been configured yet
        pass

    console_level = (console_level or "INFO").upper()
    file_level = (file_level or console_level).upper()

    # Console sink
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=console_level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n",
    )

    # Optional file sink controlled by argument or env var for backward compatibility
    log_file = log_file or os.environ.get("KeySG_LOG_FILE")
    if log_file:
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except Exception:
            pass
        logger.add(log_file, level=file_level, rotation="10 MB", retention=3)


__all__ = ["setup_logging", "logger"]
