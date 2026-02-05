"""Resource manager for parallel context retrieval with token budgets.

Provides dedicated executors and token bucket rate limiting for:
- LLM API calls (rate-limited)
- Embedding generation (CPU/GPU bound)
- Graph traversal (I/O bound)
- File operations (I/O bound)
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from codesage.utils.logging import get_logger

logger = get_logger("resource_manager")

T = TypeVar("T")


class ResourceType(str, Enum):
    """Types of resources managed."""

    LLM = "llm"  # LLM API calls - rate limited
    EMBEDDING = "embedding"  # Embedding generation - compute bound
    GRAPH = "graph"  # Graph traversal - I/O bound
    FILE_IO = "file_io"  # File operations - I/O bound
    SECURITY = "security"  # Security scanning - CPU bound


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Implements the token bucket algorithm for smooth rate limiting.
    Tokens regenerate at a fixed rate up to a maximum capacity.
    """

    capacity: int  # Maximum tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire.
            timeout: Maximum time to wait (None = non-blocking).

        Returns:
            True if tokens acquired, False if timeout or would block.
        """
        deadline = time.monotonic() + timeout if timeout else None

        with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if deadline is None:
                    return False

                # Wait for tokens to regenerate
                wait_time = (tokens - self.tokens) / self.refill_rate
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    return False

                # Release lock while waiting
                self._lock.release()
                try:
                    time.sleep(min(wait_time, remaining))
                finally:
                    self._lock.acquire()

    async def acquire_async(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Async version of acquire."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.acquire, tokens, timeout)


@dataclass
class ResourceConfig:
    """Configuration for a resource type."""

    max_workers: int = 4
    token_capacity: int = 10
    tokens_per_second: float = 5.0
    use_process_pool: bool = False  # Use ProcessPoolExecutor for CPU-bound


@dataclass
class ResourceStats:
    """Statistics for resource usage."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_wait_time: float = 0.0
    avg_execution_time: float = 0.0

    def record_success(self, execution_time: float) -> None:
        self.total_requests += 1
        self.successful_requests += 1
        # Running average
        n = self.successful_requests
        self.avg_execution_time = (
            (self.avg_execution_time * (n - 1) + execution_time) / n
        )

    def record_failure(self) -> None:
        self.total_requests += 1
        self.failed_requests += 1

    def record_rate_limit(self, wait_time: float) -> None:
        self.rate_limited_requests += 1
        self.total_wait_time += wait_time


class ResourceManager:
    """Manages dedicated executors and token buckets for parallel operations.

    Provides:
    - Dedicated thread/process pools per resource type
    - Token bucket rate limiting for API calls
    - Resource usage statistics
    - Graceful shutdown

    Example:
        ```python
        manager = ResourceManager()

        # Run LLM call with rate limiting
        result = await manager.run(
            ResourceType.LLM,
            llm.chat,
            messages,
            tokens=1,  # Cost of this operation
        )

        # Run embeddings in parallel (CPU-bound)
        embeddings = await manager.run_batch(
            ResourceType.EMBEDDING,
            embed_fn,
            texts,
            batch_size=10,
        )
        ```
    """

    # Default configurations per resource type
    DEFAULT_CONFIGS: Dict[ResourceType, ResourceConfig] = {
        ResourceType.LLM: ResourceConfig(
            max_workers=2,  # Limited concurrent LLM calls
            token_capacity=5,  # Burst capacity
            tokens_per_second=2.0,  # 2 requests/second
            use_process_pool=False,
        ),
        ResourceType.EMBEDDING: ResourceConfig(
            max_workers=4,
            token_capacity=20,
            tokens_per_second=10.0,
            use_process_pool=False,  # Embeddings use external service
        ),
        ResourceType.GRAPH: ResourceConfig(
            max_workers=8,  # High concurrency for I/O
            token_capacity=100,
            tokens_per_second=50.0,
            use_process_pool=False,
        ),
        ResourceType.FILE_IO: ResourceConfig(
            max_workers=8,
            token_capacity=50,
            tokens_per_second=25.0,
            use_process_pool=False,
        ),
        ResourceType.SECURITY: ResourceConfig(
            max_workers=4,
            token_capacity=10,
            tokens_per_second=5.0,
            use_process_pool=True,  # CPU-bound scanning
        ),
    }

    def __init__(
        self,
        configs: Optional[Dict[ResourceType, ResourceConfig]] = None,
    ) -> None:
        """Initialize the resource manager.

        Args:
            configs: Optional custom configurations per resource type.
        """
        self._configs = {**self.DEFAULT_CONFIGS, **(configs or {})}
        self._executors: Dict[ResourceType, ThreadPoolExecutor | ProcessPoolExecutor] = {}
        self._buckets: Dict[ResourceType, TokenBucket] = {}
        self._stats: Dict[ResourceType, ResourceStats] = {}
        self._shutdown = False
        self._lock = threading.Lock()

        self._init_resources()

    def _init_resources(self) -> None:
        """Initialize executors and token buckets."""
        for resource_type, config in self._configs.items():
            # Create executor
            if config.use_process_pool:
                self._executors[resource_type] = ProcessPoolExecutor(
                    max_workers=config.max_workers
                )
            else:
                self._executors[resource_type] = ThreadPoolExecutor(
                    max_workers=config.max_workers,
                    thread_name_prefix=f"codesage-{resource_type.value}",
                )

            # Create token bucket
            self._buckets[resource_type] = TokenBucket(
                capacity=config.token_capacity,
                refill_rate=config.tokens_per_second,
            )

            # Initialize stats
            self._stats[resource_type] = ResourceStats()

    async def run(
        self,
        resource_type: ResourceType,
        fn: Callable[..., T],
        *args: Any,
        tokens: int = 1,
        timeout: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> T:
        """Run a function with resource management.

        Args:
            resource_type: Type of resource to use.
            fn: Function to execute.
            *args: Positional arguments for fn.
            tokens: Number of tokens to consume (cost of operation).
            timeout: Maximum time to wait for resources.
            **kwargs: Keyword arguments for fn.

        Returns:
            Result of fn(*args, **kwargs).

        Raises:
            TimeoutError: If resources not available within timeout.
            Exception: Any exception from fn.
        """
        if self._shutdown:
            raise RuntimeError("ResourceManager has been shut down")

        bucket = self._buckets[resource_type]
        executor = self._executors[resource_type]
        stats = self._stats[resource_type]

        # Acquire tokens (rate limiting)
        wait_start = time.monotonic()
        acquired = await bucket.acquire_async(tokens, timeout)

        if not acquired:
            stats.record_rate_limit(time.monotonic() - wait_start)
            raise TimeoutError(f"Rate limit exceeded for {resource_type.value}")

        wait_time = time.monotonic() - wait_start
        if wait_time > 0.01:  # Only record significant waits
            stats.record_rate_limit(wait_time)

        # Execute in appropriate pool
        loop = asyncio.get_event_loop()
        exec_start = time.monotonic()

        try:
            result = await loop.run_in_executor(
                executor,
                lambda: fn(*args, **kwargs),
            )
            stats.record_success(time.monotonic() - exec_start)
            return result
        except Exception:
            stats.record_failure()
            raise

    async def run_batch(
        self,
        resource_type: ResourceType,
        fn: Callable[[T], Any],
        items: List[T],
        tokens_per_item: int = 1,
        batch_size: int = 10,
        timeout: Optional[float] = 60.0,
    ) -> List[Any]:
        """Run a function over multiple items with batching.

        Args:
            resource_type: Type of resource to use.
            fn: Function to apply to each item.
            items: List of items to process.
            tokens_per_item: Tokens consumed per item.
            batch_size: Maximum concurrent operations.
            timeout: Timeout per batch.

        Returns:
            List of results in same order as items.
        """
        results: List[Any] = [None] * len(items)
        errors: List[Exception] = []

        # Process in batches
        for batch_start in range(0, len(items), batch_size):
            batch_end = min(batch_start + batch_size, len(items))
            batch_items = items[batch_start:batch_end]

            tasks = [
                self.run(resource_type, fn, item, tokens=tokens_per_item, timeout=timeout)
                for item in batch_items
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                idx = batch_start + i
                if isinstance(result, Exception):
                    errors.append(result)
                    results[idx] = None
                else:
                    results[idx] = result

        if errors:
            logger.warning(f"Batch processing had {len(errors)} errors")

        return results

    def get_stats(self, resource_type: Optional[ResourceType] = None) -> Dict[str, Any]:
        """Get resource usage statistics.

        Args:
            resource_type: Specific resource or None for all.

        Returns:
            Dictionary of statistics.
        """
        if resource_type:
            stats = self._stats[resource_type]
            return {
                "resource": resource_type.value,
                "total_requests": stats.total_requests,
                "successful": stats.successful_requests,
                "failed": stats.failed_requests,
                "rate_limited": stats.rate_limited_requests,
                "avg_execution_time_ms": round(stats.avg_execution_time * 1000, 2),
                "total_wait_time_ms": round(stats.total_wait_time * 1000, 2),
            }

        return {
            rt.value: self.get_stats(rt)
            for rt in ResourceType
        }

    def get_bucket_status(self, resource_type: ResourceType) -> Dict[str, Any]:
        """Get token bucket status for a resource.

        Args:
            resource_type: Resource to check.

        Returns:
            Bucket status dictionary.
        """
        bucket = self._buckets[resource_type]
        config = self._configs[resource_type]

        with bucket._lock:
            bucket._refill()
            return {
                "available_tokens": round(bucket.tokens, 2),
                "capacity": config.token_capacity,
                "refill_rate": config.tokens_per_second,
                "utilization": round(1 - bucket.tokens / config.token_capacity, 2),
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all executors.

        Args:
            wait: Whether to wait for pending tasks.
        """
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

        for executor in self._executors.values():
            executor.shutdown(wait=wait)

        logger.info("ResourceManager shut down")

    def __enter__(self) -> "ResourceManager":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()


# Global instance for convenience
_global_manager: Optional[ResourceManager] = None
_manager_lock = threading.Lock()


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance.

    Creates one if it doesn't exist.
    """
    global _global_manager

    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                _global_manager = ResourceManager()

    return _global_manager


def shutdown_resource_manager() -> None:
    """Shutdown the global resource manager."""
    global _global_manager

    with _manager_lock:
        if _global_manager is not None:
            _global_manager.shutdown()
            _global_manager = None
