"""
Centralized logging setup for the KeySG project.

Goals:
- Use Loguru consistently across modules.
- Default to console-only logging (no files).
- Allow optional file logging via env var KeySG_LOG_FILE if needed.
- Allow log levels to be provided by the Hydra pipeline configuration.
- Optionally capture *all* stdout/stderr output (not just loguru) into the log file.
"""

from __future__ import annotations

import io
import os
import sys
from loguru import logger


class StreamTee(io.TextIOBase):
    """A writable text stream that duplicates every write to *both* the
    original stream (e.g. ``sys.__stdout__``) **and** a log file on disk.

    This ensures that ``print()`` calls, third-party library warnings, and
    any other non-loguru output are captured in the log file alongside the
    structured loguru messages.

    Usage::

        tee = StreamTee(sys.stdout, "/path/to/log.txt")
        sys.stdout = tee          # install
        # ... later ...
        sys.stdout = tee.original # restore
    """

    def __init__(self, original: io.TextIOBase, log_path: str) -> None:
        super().__init__()
        self.original = original
        self._log_path = log_path
        # Open the file in append mode so multiple StreamTee instances
        # (stdout + stderr) can share the same file safely.
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)

    # -- io.TextIOBase interface ------------------------------------------

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return self.original.readable()

    def write(self, text: str) -> int:
        # Always forward to the original stream first (keeps terminal output).
        n = self.original.write(text)
        try:
            self._file.write(text)
        except Exception:
            pass  # never let a log-file error break the real stream
        return n

    def flush(self) -> None:
        self.original.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def fileno(self) -> int:
        """Return the fd of the *original* stream so that libraries calling
        ``os.write(stream.fileno(), ...)`` still work (e.g. subprocess)."""
        return self.original.fileno()

    def isatty(self) -> bool:
        return self.original.isatty()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self.original, "encoding", "utf-8") or "utf-8"

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass
        # Do NOT close the original stream.

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def install_stream_capture(log_path: str) -> tuple[StreamTee, StreamTee]:
    """Replace ``sys.stdout`` and ``sys.stderr`` with :class:`StreamTee`
    instances that duplicate all output to *log_path*.

    Returns ``(stdout_tee, stderr_tee)`` so the caller can restore the
    originals later via ``sys.stdout = tee.original``.
    """
    stdout_tee = StreamTee(sys.stdout, log_path)
    stderr_tee = StreamTee(sys.stderr, log_path)
    sys.stdout = stdout_tee  # type: ignore[assignment]
    sys.stderr = stderr_tee  # type: ignore[assignment]
    return stdout_tee, stderr_tee


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
