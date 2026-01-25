"""Tests for observability features (logging and health checks)."""

import pytest
import logging
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from codesage.utils.logging import (
    setup_logging,
    get_logger,
    log_operation,
    JSONFormatter,
    HumanFormatter,
)
from codesage.utils.health import (
    HealthStatus,
    check_ollama,
    check_database,
    check_disk_space,
)


class TestLogging:
    """Tests for structured logging."""

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger."""
        logger = setup_logging(level="DEBUG")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "codesage"

    def test_get_logger_with_name(self):
        """Test getting a named logger."""
        logger = get_logger("test.module")
        assert logger.name == "codesage.test.module"

    def test_get_logger_already_prefixed(self):
        """Test that already prefixed names aren't double-prefixed."""
        logger = get_logger("codesage.another")
        assert logger.name == "codesage.another"

    def test_json_formatter_output(self):
        """Test JSON formatter produces valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_human_formatter_output(self):
        """Test human formatter produces readable output."""
        formatter = HumanFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)

        assert "INFO" in output
        assert "Test message" in output

    def test_log_operation_context_manager(self):
        """Test log_operation times operations."""
        logger = get_logger("test")

        with patch.object(logger, 'log') as mock_log:
            with log_operation(logger, "test operation"):
                pass

            # Should log start and complete
            assert mock_log.call_count == 2


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_healthy_when_all_ok(self):
        """Test is_healthy when all components are ok."""
        status = HealthStatus(
            ollama_available=True,
            database_accessible=True,
            vector_store_accessible=True,
        )
        assert status.is_healthy

    def test_unhealthy_when_ollama_down(self):
        """Test is_healthy when Ollama is down."""
        status = HealthStatus(
            ollama_available=False,
            database_accessible=True,
            vector_store_accessible=True,
        )
        assert not status.is_healthy

    def test_to_dict(self):
        """Test conversion to dictionary."""
        status = HealthStatus(
            ollama_available=True,
            database_accessible=True,
            vector_store_accessible=True,
            disk_space_ok=True,
            ollama_latency_ms=50.0,
        )
        data = status.to_dict()

        assert data["healthy"] is True
        assert data["components"]["ollama"]["status"] == "ok"
        assert data["components"]["ollama"]["latency_ms"] == 50.0


class TestOllamaCheck:
    """Tests for Ollama health check."""

    def test_ollama_connection_error(self):
        """Test handling of connection error."""
        ok, error, latency = check_ollama("http://localhost:99999", timeout=1)
        assert not ok
        assert error is not None
        assert "Cannot connect" in error


class TestDatabaseCheck:
    """Tests for database health check."""

    def test_database_not_found(self, tmp_path):
        """Test handling of missing database."""
        db_path = tmp_path / "nonexistent.db"
        ok, error, size = check_database(db_path)
        assert not ok
        assert "not found" in error


class TestDiskSpaceCheck:
    """Tests for disk space check."""

    def test_disk_space_ok(self, tmp_path):
        """Test disk space check passes with normal space."""
        ok, warning = check_disk_space(tmp_path, min_free_gb=0.001)
        assert ok
        assert warning is None

    def test_disk_space_low(self, tmp_path):
        """Test disk space check warns with very high requirement."""
        ok, warning = check_disk_space(tmp_path, min_free_gb=99999)
        assert not ok
        assert "Low disk space" in warning
