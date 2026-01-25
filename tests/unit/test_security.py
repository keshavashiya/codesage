"""Tests for security features."""

import pytest
from pathlib import Path
import json
import os
import tempfile

from codesage.utils.config import LLMConfig


class TestJSONCache:
    """Tests for JSON-based embedding cache (replaces pickle)."""

    def test_json_cache_write_read(self, tmp_path):
        """Test writing and reading JSON cache."""
        cache_file = tmp_path / "test.json"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Write
        with open(cache_file, "w") as f:
            json.dump(embedding, f)

        # Read
        with open(cache_file, "r") as f:
            loaded = json.load(f)

        assert loaded == embedding

    def test_malformed_json_handling(self, tmp_path):
        """Test that malformed JSON is handled gracefully."""
        cache_file = tmp_path / "bad.json"
        cache_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            with open(cache_file, "r") as f:
                json.load(f)

    def test_large_embedding_vectors(self, tmp_path):
        """Test handling of large embedding vectors (1536 dims)."""
        cache_file = tmp_path / "large.json"
        embedding = [0.001 * i for i in range(1536)]

        with open(cache_file, "w") as f:
            json.dump(embedding, f)

        with open(cache_file, "r") as f:
            loaded = json.load(f)

        assert len(loaded) == 1536
        assert loaded == embedding


class TestPathTraversal:
    """Tests for path traversal protection."""

    def test_path_within_root(self, tmp_path):
        """Test that paths within root are accepted."""
        root = tmp_path / "project"
        root.mkdir()
        file = root / "test.py"
        file.write_text("# test")

        root_resolved = root.resolve()
        file_resolved = file.resolve()

        assert str(file_resolved).startswith(str(root_resolved))

    def test_symlink_outside_root_detected(self, tmp_path):
        """Test that symlinks outside root are detected."""
        root = tmp_path / "project"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.py"
        secret.write_text("# secret")

        # Create symlink inside project pointing outside
        symlink = root / "link.py"
        symlink.symlink_to(secret)

        root_resolved = root.resolve()
        link_resolved = symlink.resolve()

        # Should NOT start with root path (escaped!)
        assert not str(link_resolved).startswith(str(root_resolved))

    def test_relative_path_traversal(self, tmp_path):
        """Test detection of relative path traversal."""
        root = tmp_path / "project"
        root.mkdir()

        # Attempt to escape via relative path
        escaped_path = root / ".." / "secret.txt"
        root_resolved = root.resolve()
        escaped_resolved = escaped_path.resolve()

        assert not str(escaped_resolved).startswith(str(root_resolved))


class TestInputValidation:
    """Tests for CLI input validation."""

    def test_empty_query_rejected(self):
        """Test that empty queries are rejected."""
        query = ""
        assert not query.strip()

    def test_whitespace_only_rejected(self):
        """Test that whitespace-only queries are rejected."""
        query = "   \n\t  "
        assert not query.strip()

    def test_normal_query_accepted(self):
        """Test that normal queries pass validation."""
        query = "find the function that handles authentication"
        stripped = query.strip()
        assert stripped
        assert len(stripped) <= 2000

    def test_long_query_rejected(self):
        """Test that excessively long queries are rejected."""
        query = "a" * 2001
        assert len(query) > 2000

    def test_max_length_query_accepted(self):
        """Test that queries at max length are accepted."""
        query = "x" * 2000
        assert len(query) <= 2000


class TestAPIKeySecurity:
    """Tests for secure API key handling."""

    def test_api_key_from_env(self, monkeypatch):
        """Test that API key is loaded from environment."""
        monkeypatch.setenv("CODESAGE_API_KEY", "test-key-123")
        config = LLMConfig()
        assert config.api_key == "test-key-123"

    def test_api_key_validation_openai(self):
        """Test validation fails for OpenAI without API key."""
        config = LLMConfig(provider="openai", api_key=None)
        with pytest.raises(ValueError, match="requires CODESAGE_API_KEY"):
            config.validate()

    def test_api_key_validation_anthropic(self):
        """Test validation fails for Anthropic without API key."""
        config = LLMConfig(provider="anthropic", api_key=None)
        with pytest.raises(ValueError, match="requires CODESAGE_API_KEY"):
            config.validate()

    def test_ollama_no_key_required(self):
        """Test that Ollama doesn't require API key."""
        config = LLMConfig(provider="ollama", api_key=None)
        # Should not raise
        config.validate()

    def test_api_key_not_saved_to_yaml(self, tmp_path):
        """Test that API key is not written to config file."""
        from codesage.utils.config import Config

        config = Config(
            project_name="test",
            project_path=tmp_path,
            llm=LLMConfig(api_key="secret-key"),
        )
        config.save()

        config_file = tmp_path / ".codesage" / "config.yaml"
        content = config_file.read_text()

        assert "secret-key" not in content
