"""Structured logging configuration with console and file output.

This module provides a centralized logging system with:
- Color-coded console output for development.
- Rotating file handlers to prevent log bloat.
- UTC ISO-8601 timestamps with milliseconds (``Z`` suffix) on both handlers,
  matching the Electron log's ``new Date().toISOString()`` so the two logs
  can be correlated on a single timeline.
- A per-request id (``[be-xxxxxxxx]``) injected into every line via
  ``src.core.request_context`` (set by the request-logging middleware).
- Log level driven by the ``ERUDI_LOG_LEVEL`` environment variable
  (default ``INFO``; invalid values fall back to ``INFO``).

Log Levels (the rules are in docs/logging.md, "Logging rules"):
    - DEBUG: Internal diagnostics (token generation, model loading steps).
    - INFO: Lifecycle transitions (startup, shutdown, model switched) and the
      expected outcomes of user actions, a 4xx the client asked for included.
    - WARNING: The app recovered or degraded on its own (fallback to CPU, a
      skipped file, an answer without its KB excerpts).
    - ERROR: The operation did not happen (5xx, a dead inference child, a
      failed download or ingestion, a background task that died), always with
      the exception attached.
    - CRITICAL: System failures (database corruption, engine crash).

Log Format:
    Console::

        [INFO]  2026-07-02T09:15:32.123Z [be-1f2e3d4c] - erudi - backend/src/main.py:app:l42 - Server started

    File::

        [INFO] 2026-07-02T09:15:32.123Z [be-1f2e3d4c] - erudi - main.py:42 - Server started

File Management:
    - Logs written to ``backend.log`` in the runtime log directory configured
      by ``src.launcher.runtime_paths`` (typically ``backend/logs`` in dev).
      The filename is stable (no date suffix): a date computed at import time
      froze at the process start, so a process crossing midnight kept writing
      to the previous day's file. Rotation preserves history instead.
    - Rotation: 10 MB max file size.
    - Retention: 10 backup files (backend.log.1, backend.log.2, ...).
    - Encoding: UTF-8 for multilingual support.

Example:
    Use the global logger instance::

        from src.core.logging import logger

        logger.debug("Loading model weights...")
        logger.info("Model successfully loaded")
        logger.warning("GPU memory low, reducing batch size")
        logger.error("Failed to load model", exc_info=True)

    Create a custom logger for a specific module::

        from src.core.logging import get_logger

        module_logger = get_logger("erudi.domains.llms")
        module_logger.info("LLM service initialized")

Note:
    The logger is configured as a singleton. Repeated calls to get_logger()
    with the same name will return the existing logger instance. Use
    configure_logger() to force a reconfiguration (e.g. after changing
    ERUDI_LOG_LEVEL in tests).

Warning:
    Never log sensitive data (HF tokens, user prompts, file paths with PII).
    Use structured logging: logger.info("Model loaded", extra={"model_id": 123})
"""

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

from src.core.request_context import request_id_var
from src.launcher import ensure_runtime_paths_initialized

DEFAULT_LOG_LEVEL = logging.INFO
LOG_FILE_NAME = "backend.log"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"
# `%(asctime)s.%(msecs)03dZ` + gmtime converter = UTC ISO-8601 with ms.
FILE_LOG_FORMAT = (
    "[%(levelname)s] %(asctime)s.%(msecs)03dZ [%(request_id)s] - %(name)s"
    " - %(filename)s:%(lineno)d - %(message)s"
)
_CONSOLE_BODY = (
    "%(asctime)s.%(msecs)03dZ [%(request_id)s] - %(name)s"
    " - %(short_path)s:%(funcName)s:l%(lineno)d - %(message)s"
)

MAX_LOG_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 10  # Keep 10 rotated files


# ----------------------------
# Level resolution
# ----------------------------
def resolve_log_level(raw: str | None) -> int:
    """Map an ERUDI_LOG_LEVEL value to a logging level, defaulting to INFO.

    Args:
        raw: Raw environment value ("DEBUG", "info", ...) or None.

    Returns:
        int: A stdlib logging level. Unknown/empty values fall back to INFO.
    """
    if not raw:
        return DEFAULT_LOG_LEVEL
    level = logging.getLevelName(raw.strip().upper())
    return level if isinstance(level, int) else DEFAULT_LOG_LEVEL


# ----------------------------
# Request-id injection
# ----------------------------
class RequestIdFilter(logging.Filter):
    """Inject the current request id into every record (``-`` outside requests).

    Attached to both the logger and its handlers so records logged through
    child loggers (which bypass ancestor logger filters) are still tagged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


_REQUEST_ID_FILTER = RequestIdFilter()


# ----------------------------
# Formatters
# ----------------------------
def utc_formatter(fmt: str) -> logging.Formatter:
    """Build a formatter emitting UTC ISO-8601 timestamps with ms and Z suffix."""
    formatter = logging.Formatter(fmt, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime
    return formatter


class CustomFormatter(logging.Formatter):
    """Custom formatter to add colors for console output and shorten pathname."""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;21m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: blue + "[DEBUG] " + _CONSOLE_BODY + reset,
        logging.INFO: grey + "[INFO]  " + _CONSOLE_BODY + reset,
        logging.WARNING: yellow + "[WARN]  " + _CONSOLE_BODY + reset,
        logging.ERROR: red + "[ERROR] " + _CONSOLE_BODY + reset,
        logging.CRITICAL: bold_red + "[CRIT] " + _CONSOLE_BODY + reset,
    }

    def __init__(self):
        super().__init__(datefmt=LOG_DATEFMT)
        # Pre-build one UTC formatter per level instead of one per record.
        self._formatters = {level: utc_formatter(fmt) for level, fmt in self.FORMATS.items()}

    def format(self, record):
        """Format log record with color codes and shortened paths.

        Args:
            record: LogRecord instance containing log event information.

        Returns:
            str: Formatted log message with ANSI color codes.

        Note:
            The shortened path (from "backend/") is computed into a dedicated
            ``short_path`` attribute — mutating ``record.pathname`` in place
            would leak the truncation to every other handler formatting the
            same record.
        """
        backend_idx = record.pathname.find("backend/")
        record.short_path = record.pathname[backend_idx:] if backend_idx != -1 else record.pathname
        formatter = self._formatters.get(record.levelno) or self._formatters[logging.INFO]
        return formatter.format(record)


# ----------------------------
# Logger setup
# ----------------------------
def configure_logger(name: str = "erudi") -> logging.Logger:
    """(Re)apply handlers, level, formatters and filters from the environment.

    Reads ERUDI_LOG_LEVEL (default INFO) and applies it to the logger and to
    both handlers. Idempotent: handlers previously attached to this logger
    are removed and closed first, so it is safe to call again (e.g. from
    tests after monkeypatching the environment).

    Args:
        name: Logger name to configure. Defaults to "erudi".

    Returns:
        logging.Logger: The configured logger instance.
    """
    runtime_paths = ensure_runtime_paths_initialized()
    log_dir = runtime_paths.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME

    level = resolve_log_level(os.environ.get("ERUDI_LOG_LEVEL"))

    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(level)
    if _REQUEST_ID_FILTER not in logger.filters:
        logger.addFilter(_REQUEST_ID_FILTER)

    # Console handler (for development)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    ch.addFilter(_REQUEST_ID_FILTER)
    logger.addHandler(ch)

    # File handler (rotating)
    fh = RotatingFileHandler(
        filename=log_file,
        maxBytes=MAX_LOG_FILE_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(utc_formatter(FILE_LOG_FORMAT))
    fh.addFilter(_REQUEST_ID_FILTER)
    logger.addHandler(fh)

    return logger


def get_logger(name: str = "erudi") -> logging.Logger:
    """Create or retrieve a configured logger instance.

    If a logger with the given name (or one of its ancestors) already has
    handlers, returns the existing instance to prevent duplicate handler
    registration — child loggers such as "erudi.domains.llms" propagate to
    the "erudi" handlers. Otherwise the logger is configured from scratch
    via configure_logger().

    Args:
        name: Logger name, typically "erudi" or "erudi.<module>".
            Defaults to "erudi".

    Returns:
        logging.Logger: Configured logger instance with console and file handlers.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Avoid duplicate handlers
    return configure_logger(name)


# ----------------------------
# Silence noisy third-party loggers
# ----------------------------
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
# uvicorn's per-request access log is replaced by the request-logging
# middleware (src.core.api.RequestLoggingMiddleware); run.py also passes
# access_log=False to uvicorn.Config.
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ----------------------------
# Global logger instance
# ----------------------------
logger = configure_logger()
