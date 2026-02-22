"""Structured logging for CodeSage.

Provides configurable logging with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- JSON format for production
- Human-readable format for development
- Optional file logging with rotation
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional
from functools import wraps
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json

        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation

        return json.dumps(log_data)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level = f"{color}{level}{reset}"

        message = record.getMessage()
        timestamp = self.formatTime(record, "%H:%M:%S")

        formatted = f"{timestamp} [{level}] {record.name}: {message}"

        # Add duration if present
        if hasattr(record, "duration_ms"):
            formatted += f" ({record.duration_ms:.1f}ms)"

        # Add exception if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


_loggers: dict = {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """Set up logging for CodeSage.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        json_format: Use JSON format (for production)
        use_colors: Use colored output (development)

    Returns:
        Configured root logger
    """
    logger = logging.getLogger("codesage")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = HumanFormatter(use_colors=use_colors)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        try:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            )
            # Always use JSON for file logging
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Args:
        name: Logger name (will be prefixed with 'codesage.')

    Returns:
        Logger instance
    """
    full_name = f"codesage.{name}" if not name.startswith("codesage.") else name

    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)

    return _loggers[full_name]


@contextmanager
def log_operation(logger: logging.Logger, operation: str, level: int = logging.INFO):
    """Context manager for logging operation timing.

    Args:
        logger: Logger to use
        operation: Name of the operation
        level: Log level for completion message

    Example:
        with log_operation(logger, "indexing files"):
            # do work
    """
    start = time.perf_counter()
    logger.log(level, f"Starting: {operation}")

    try:
        yield
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.error(
            f"Failed: {operation}",
            extra={"duration_ms": duration, "operation": operation},
            exc_info=True,
        )
        raise
    else:
        duration = (time.perf_counter() - start) * 1000
        logger.log(
            level,
            f"Completed: {operation}",
            extra={"duration_ms": duration, "operation": operation},
        )


def timed(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator for timing function execution.

    Args:
        logger: Logger to use
        level: Log level for timing message

    Example:
        @timed(logger)
        def my_function():
            # do work
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{func.__name__} completed",
                    extra={"duration_ms": duration, "operation": func.__name__},
                )
            return result
        return wrapper
    return decorator
