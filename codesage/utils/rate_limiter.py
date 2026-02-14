"""Rate limiter for LLM API calls.

Implements token bucket algorithm for rate limiting requests
to prevent overwhelming the LLM service.
"""

import time
import threading
from typing import Optional


class RateLimiter:
    """Enhanced token bucket rate limiter for API calls.

    Thread-safe implementation that limits requests per minute
    using the token bucket algorithm with configurable bucket size
    and refill rate for optimal performance and accuracy.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        bucket_size: Optional[int] = None,
        refill_rate: Optional[int] = None,
    ):
        """Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
            bucket_size: Token bucket size (defaults to requests_per_minute)
            refill_rate: Tokens per minute refill rate (defaults to requests_per_minute)
        """
        self.rpm = max(1, requests_per_minute)
        # Use provided values or default to rpm
        self.bucket_size = bucket_size if bucket_size is not None else self.rpm
        self.refill_rate = refill_rate if refill_rate is not None else self.rpm

        # Initialize tokens to bucket size for burst handling
        self.tokens = float(self.bucket_size)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

        # Enhanced metrics
        self.total_requests = 0
        self.total_waits = 0
        self.total_wait_time = 0.0
        self.peak_wait_time = 0.0

    def _refill_tokens(self) -> None:
        """Refill tokens based on configured refill rate."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now

        # Calculate tokens to add based on refill rate
        tokens_to_add = elapsed * (self.refill_rate / 60.0)
        self.tokens = min(self.bucket_size, self.tokens + tokens_to_add)

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a token, blocking if necessary.

        Args:
            timeout: Maximum time to wait for a token (None = wait forever)

        Returns:
            True if token acquired, False if timeout exceeded
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._lock:
            while True:
                self._refill_tokens()

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.total_requests += 1
                    return True

                # Calculate wait time for next token
                wait_time = (1.0 - self.tokens) * (60.0 / self.rpm)

                # Check timeout
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)

                # Track wait metrics
                self.total_waits += 1
                self.total_wait_time += wait_time
                # Track peak wait time for performance monitoring
                if wait_time > self.peak_wait_time:
                    self.peak_wait_time = wait_time

                # Release lock during sleep
                self._lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._lock.acquire()

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if rate limited
        """
        with self._lock:
            self._refill_tokens()

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                self.total_requests += 1
                return True

            return False

    def get_metrics(self) -> dict:
        """Get rate limiter metrics.

        Returns:
            Dictionary with request and wait statistics
        """
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_waits": self.total_waits,
                "total_wait_time_seconds": self.total_wait_time,
                "avg_wait_time_ms": (
                    (self.total_wait_time / self.total_waits * 1000)
                    if self.total_waits > 0
                    else 0.0
                ),
                "peak_wait_time_ms": self.peak_wait_time * 1000,
                "current_tokens": self.tokens,
                "bucket_size": self.bucket_size,
                "refill_rate": self.refill_rate,
                "requests_per_minute": self.rpm,
            }

    def reset(self) -> None:
        """Reset the rate limiter state and metrics."""
        with self._lock:
            self.tokens = float(self.bucket_size)
            self.last_update = time.monotonic()
            self.total_requests = 0
            self.total_waits = 0
            self.total_wait_time = 0.0
            self.peak_wait_time = 0.0
