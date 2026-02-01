"""Tests for production critical fixes (retry, timeouts, permissions)."""

import pytest
import time
import tempfile
import os
import stat
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from codesage.utils.retry import (
    retry_with_backoff,
    RetryExhausted,
    RetryContext,
    retry_on_connection_error,
)
from codesage.utils.config import LLMConfig


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test that failures trigger retries."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exhausted(self):
        """Test that max retries are respected."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_specific_exceptions_only(self):
        """Test that only specified exceptions trigger retries."""
        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            exceptions=(ConnectionError,),
        )
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not a connection error")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1  # No retries for ValueError

    def test_exponential_backoff_timing(self):
        """Test that backoff delay increases exponentially."""
        delays = []
        last_time = [time.monotonic()]

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.05,
            exponential_base=2.0,
            jitter=False,
        )
        def track_timing():
            now = time.monotonic()
            if last_time[0]:
                delays.append(now - last_time[0])
            last_time[0] = now
            if len(delays) < 3:
                raise ConnectionError("Fail")
            return "done"

        track_timing()

        # First retry should be ~0.05s, second ~0.1s
        assert len(delays) >= 2
        # Allow some tolerance for timing
        assert delays[1] > delays[0] * 1.5  # Should be exponentially larger

    def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        retry_info = []

        def on_retry(exception, attempt):
            retry_info.append((str(exception), attempt))

        @retry_with_backoff(
            max_retries=2,
            base_delay=0.01,
            on_retry=on_retry,
        )
        def fails_twice():
            if len(retry_info) < 2:
                raise ConnectionError("Fail")
            return "success"

        fails_twice()

        assert len(retry_info) == 2
        assert retry_info[0][1] == 0  # First retry
        assert retry_info[1][1] == 1  # Second retry


class TestRetryContext:
    """Tests for RetryContext context manager."""

    def test_successful_first_attempt(self):
        """Test successful operation on first attempt."""
        with RetryContext(max_retries=3) as retry:
            for attempt in retry:
                result = "success"
                break

        assert result == "success"
        assert retry.attempt == 0

    def test_retry_until_success(self):
        """Test retrying until success."""
        attempt_count = 0

        with RetryContext(max_retries=3, base_delay=0.01) as retry:
            for attempt in retry:
                attempt_count += 1
                try:
                    if attempt < 2:
                        raise ValueError("Not yet")
                    result = "success"
                    break
                except ValueError as e:
                    retry.record_failure(e)

        assert result == "success"
        assert attempt_count == 3

    def test_is_last_attempt(self):
        """Test is_last_attempt property."""
        with RetryContext(max_retries=2) as retry:
            attempts = []
            for attempt in retry:
                attempts.append(retry.is_last_attempt)
                if not retry.is_last_attempt:
                    try:
                        raise ValueError("Fail")
                    except ValueError as e:
                        retry.record_failure(e)
                else:
                    break

        assert attempts == [False, False, True]


class TestLLMConfigTimeouts:
    """Tests for LLMConfig timeout settings."""

    def test_default_timeout_values(self):
        """Test that default timeout values are set."""
        config = LLMConfig()
        assert config.request_timeout == 30.0
        assert config.connect_timeout == 5.0
        assert config.max_retries == 3

    def test_custom_timeout_values(self):
        """Test custom timeout configuration."""
        config = LLMConfig(
            request_timeout=60.0,
            connect_timeout=10.0,
            max_retries=5,
        )
        assert config.request_timeout == 60.0
        assert config.connect_timeout == 10.0
        assert config.max_retries == 5

    def test_invalid_timeout_validation(self):
        """Test that invalid timeouts are rejected."""
        config = LLMConfig(request_timeout=0)
        with pytest.raises(ValueError, match="request_timeout must be positive"):
            config.validate()

        config = LLMConfig(connect_timeout=-1)
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            config.validate()

        config = LLMConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            config.validate()

    def test_zero_retries_allowed(self):
        """Test that zero retries is valid (no retries)."""
        config = LLMConfig(max_retries=0)
        config.validate()  # Should not raise


class TestDatabasePermissions:
    """Tests for database file permission hardening."""

    def test_database_file_permissions(self, tmp_path):
        """Test that database file has restricted permissions."""
        from codesage.storage.database import Database

        db_path = tmp_path / "test.db"
        db = Database(db_path)

        # Check file exists and has restricted permissions
        assert db_path.exists()

        # Get file permissions
        file_stat = os.stat(db_path)
        mode = file_stat.st_mode

        # Should be owner read/write only (0o600)
        # Note: On some systems, umask may affect this
        owner_perms = mode & 0o700
        assert owner_perms == 0o600, f"Expected 0o600, got {oct(owner_perms)}"

        # Group and others should have no permissions
        group_other_perms = mode & 0o077
        assert group_other_perms == 0o000, f"Group/other should have no perms: {oct(mode)}"

        db.close()

    def test_database_wal_mode(self, tmp_path):
        """Test that WAL mode is enabled for performance."""
        from codesage.storage.database import Database

        db_path = tmp_path / "test_wal.db"
        db = Database(db_path)

        # Check WAL mode is enabled
        cursor = db.conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

        db.close()


class TestGracefulShutdown:
    """Tests for graceful shutdown infrastructure."""

    def test_register_cleanup_handler(self):
        """Test registering cleanup handlers."""
        from codesage.cli.utils.signals import register_cleanup, unregister_cleanup, _cleanup_handlers

        cleanup_called = []

        def my_cleanup():
            cleanup_called.append(True)

        # Store original handlers
        original_handlers = _cleanup_handlers.copy()

        try:
            register_cleanup(my_cleanup)
            assert my_cleanup in _cleanup_handlers

            # Duplicate registration should not add twice
            register_cleanup(my_cleanup)
            assert _cleanup_handlers.count(my_cleanup) == 1

            unregister_cleanup(my_cleanup)
            assert my_cleanup not in _cleanup_handlers
        finally:
            # Restore original handlers
            _cleanup_handlers.clear()
            _cleanup_handlers.extend(original_handlers)

    def test_unregister_nonexistent_handler(self):
        """Test that unregistering non-existent handler doesn't fail."""
        from codesage.cli.utils.signals import unregister_cleanup

        def nonexistent():
            pass

        # Should not raise
        unregister_cleanup(nonexistent)


class TestEmbeddingRetry:
    """Tests for embedding service retry logic."""

    def test_embedding_error_class(self):
        """Test that EmbeddingError is properly defined."""
        from codesage.llm.embeddings import EmbeddingError

        error = EmbeddingError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


class TestLLMProviderErrors:
    """Tests for LLM provider error classes."""

    def test_error_hierarchy(self):
        """Test LLM error class hierarchy."""
        from codesage.llm.provider import (
            LLMProviderError,
            LLMConnectionError,
            LLMTimeoutError,
        )

        # Test hierarchy
        assert issubclass(LLMConnectionError, LLMProviderError)
        assert issubclass(LLMTimeoutError, LLMProviderError)

        # Test instantiation
        conn_err = LLMConnectionError("Connection failed")
        assert "Connection failed" in str(conn_err)

        timeout_err = LLMTimeoutError("Request timed out")
        assert "timed out" in str(timeout_err)


class TestDatabaseError:
    """Tests for database error class."""

    def test_database_error_class(self):
        """Test that DatabaseError is properly defined."""
        from codesage.storage.database import DatabaseError

        error = DatabaseError("Database operation failed")
        assert str(error) == "Database operation failed"
        assert isinstance(error, Exception)
