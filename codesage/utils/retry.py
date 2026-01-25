"""Retry utilities with exponential backoff.

Provides decorators and utilities for retrying operations
that may fail transiently (network issues, service unavailability, etc.)
"""

import time
import logging
import random
from functools import wraps
from typing import Callable, Type, Tuple, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """Decorator for retry with exponential backoff.

    Implements exponential backoff with optional jitter to prevent
    thundering herd problems when multiple clients retry simultaneously.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter to delay to prevent thundering herd
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback called on each retry (exception, attempt)

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_external_service():
            # May fail transiently
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter (±25% randomization)
                    if jitter:
                        jitter_range = delay * 0.25
                        delay = delay + random.uniform(-jitter_range, jitter_range)
                        delay = max(0.1, delay)  # Ensure minimum delay

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    # Call optional retry callback
                    if on_retry:
                        try:
                            on_retry(e, attempt)
                        except Exception:
                            pass  # Don't let callback errors affect retry logic

                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            raise RetryExhausted(
                f"{func.__name__} failed after {max_retries + 1} attempts",
                last_exception
            )

        return wrapper

    return decorator


def retry_on_connection_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """Convenience decorator for retrying on connection errors.

    Catches common connection-related exceptions.
    """
    # Common connection-related exceptions
    connection_exceptions = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    try:
        import urllib.error
        connection_exceptions = connection_exceptions + (urllib.error.URLError,)
    except ImportError:
        pass

    try:
        import requests
        connection_exceptions = connection_exceptions + (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )
    except ImportError:
        pass

    return retry_with_backoff(
        max_retries=max_retries,
        base_delay=base_delay,
        exceptions=connection_exceptions,
    )


class RetryContext:
    """Context manager for retry logic.

    Useful when you need more control over the retry logic
    than the decorator provides.

    Example:
        with RetryContext(max_retries=3) as retry:
            for attempt in retry:
                try:
                    result = risky_operation()
                    break
                except TransientError as e:
                    retry.record_failure(e)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.attempt = 0
        self.last_exception: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __iter__(self):
        for self.attempt in range(self.max_retries + 1):
            yield self.attempt

    def record_failure(self, exception: Exception) -> None:
        """Record a failure and sleep before next attempt."""
        self.last_exception = exception

        if self.attempt < self.max_retries:
            delay = min(
                self.base_delay * (2 ** self.attempt),
                self.max_delay
            )
            time.sleep(delay)
        else:
            raise RetryExhausted(
                f"Failed after {self.max_retries + 1} attempts",
                exception
            )

    @property
    def is_last_attempt(self) -> bool:
        """Check if this is the last retry attempt."""
        return self.attempt >= self.max_retries
