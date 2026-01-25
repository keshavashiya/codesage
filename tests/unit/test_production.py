"""Tests for production features (rate limiting, configuration validation)."""

import pytest
import time
import threading

from codesage.utils.rate_limiter import RateLimiter
from codesage.utils.config import LLMConfig


class TestRateLimiter:
    """Tests for rate limiter."""

    def test_basic_acquire(self):
        """Test basic token acquisition."""
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.try_acquire()
        assert limiter.total_requests == 1

    def test_multiple_acquires(self):
        """Test multiple acquisitions within limit."""
        limiter = RateLimiter(requests_per_minute=100)
        for _ in range(10):
            assert limiter.try_acquire()
        assert limiter.total_requests == 10

    def test_rate_limiting_kicks_in(self):
        """Test that rate limiting works."""
        limiter = RateLimiter(requests_per_minute=3)

        # Exhaust tokens
        for _ in range(3):
            limiter.try_acquire()

        # Next one should fail
        assert not limiter.try_acquire()

    def test_tokens_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(requests_per_minute=600)  # 10 per second

        # Use some tokens
        for _ in range(5):
            limiter.try_acquire()

        # Wait for refill (should get ~1.5 tokens in 0.15 seconds at 10/s)
        time.sleep(0.15)

        # Should have some tokens back
        assert limiter.try_acquire()

    def test_acquire_with_timeout(self):
        """Test acquire with timeout."""
        limiter = RateLimiter(requests_per_minute=60)

        # Exhaust tokens
        for _ in range(60):
            limiter.try_acquire()

        # Acquire with short timeout should return quickly
        start = time.monotonic()
        result = limiter.acquire(timeout=0.1)
        elapsed = time.monotonic() - start

        assert elapsed < 0.2  # Should timeout quickly

    def test_metrics(self):
        """Test metrics tracking."""
        limiter = RateLimiter(requests_per_minute=10)

        for _ in range(5):
            limiter.try_acquire()

        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 5
        assert metrics["requests_per_minute"] == 10

    def test_reset(self):
        """Test reset functionality."""
        limiter = RateLimiter(requests_per_minute=10)

        for _ in range(10):
            limiter.try_acquire()

        limiter.reset()

        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["current_tokens"] == 10

    def test_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        limiter = RateLimiter(requests_per_minute=1000)
        successful = []

        def worker():
            for _ in range(10):
                if limiter.try_acquire():
                    successful.append(1)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total should match
        assert limiter.total_requests == len(successful)


class TestConfigurationValidation:
    """Tests for configuration validation."""

    def test_llm_config_validate_ollama(self):
        """Test validation passes for Ollama without API key."""
        config = LLMConfig(provider="ollama")
        config.validate()  # Should not raise

    def test_llm_config_validate_openai_no_key(self, monkeypatch):
        """Test validation fails for OpenAI without API key."""
        monkeypatch.delenv("CODESAGE_API_KEY", raising=False)
        config = LLMConfig(provider="openai", api_key=None)
        with pytest.raises(ValueError, match="requires CODESAGE_API_KEY"):
            config.validate()

    def test_llm_config_validate_openai_with_key(self, monkeypatch):
        """Test validation passes for OpenAI with API key."""
        monkeypatch.setenv("CODESAGE_API_KEY", "test-key")
        config = LLMConfig(provider="openai")
        config.validate()  # Should not raise
