"""Tests for configuration system."""

import pytest
from pathlib import Path
import yaml

from codesage.utils.config import Config, LLMConfig, StorageConfig, initialize_project


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_default_values(self):
        """Test default LLM config values."""
        config = LLMConfig()

        assert config.provider == "ollama"
        assert config.model == "qwen2.5-coder:7b"
        assert config.embedding_model == "qwen3-embedding"
        assert config.temperature == 0.3

    def test_custom_values(self):
        """Test custom LLM config values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7


class TestConfig:
    """Tests for main configuration."""

    def test_config_creation(self, tmp_path):
        """Test creating a new config."""
        config = Config(
            project_name="test-project",
            project_path=tmp_path,
        )

        assert config.project_name == "test-project"
        assert config.project_path == tmp_path
        assert config.language == "python"
        assert "venv" in config.exclude_dirs

    def test_config_save_load(self, tmp_path):
        """Test saving and loading config."""
        # Create and save
        config = Config(
            project_name="test-project",
            project_path=tmp_path,
            llm=LLMConfig(model="codellama:7b"),
        )
        config.save()

        # Check file exists
        config_file = tmp_path / ".codesage" / "config.yaml"
        assert config_file.exists()

        # Load and verify
        loaded = Config.load(tmp_path)
        assert loaded.project_name == "test-project"
        assert loaded.llm.model == "codellama:7b"

    def test_codesage_dir_property(self, tmp_path):
        """Test codesage_dir property."""
        config = Config(
            project_name="test",
            project_path=tmp_path,
        )

        assert config.codesage_dir == tmp_path / ".codesage"

    def test_cache_dir_property(self, tmp_path):
        """Test cache_dir property."""
        config = Config(
            project_name="test",
            project_path=tmp_path,
        )

        assert config.cache_dir == tmp_path / ".codesage" / "cache"


class TestInitializeProject:
    """Tests for project initialization."""

    def test_initialize_creates_directory(self, tmp_path):
        """Test that initialization creates .codesage directory."""
        config = initialize_project(tmp_path)

        assert (tmp_path / ".codesage").exists()
        assert (tmp_path / ".codesage" / "cache").exists()
        assert (tmp_path / ".codesage" / "config.yaml").exists()

    def test_initialize_with_custom_model(self, tmp_path):
        """Test initialization with custom model."""
        config = initialize_project(
            tmp_path,
            model="codellama:13b",
            embedding_model="mxbai-embed-large",
        )

        assert config.llm.model == "codellama:13b"
        assert config.llm.embedding_model == "mxbai-embed-large"

    def test_config_yaml_content(self, tmp_path):
        """Test the content of saved config.yaml."""
        initialize_project(tmp_path)

        config_file = tmp_path / ".codesage" / "config.yaml"
        with open(config_file) as f:
            data = yaml.safe_load(f)

        assert data["project_name"] == tmp_path.name
        assert data["llm"]["provider"] == "ollama"
        assert data["llm"]["model"] == "qwen2.5-coder:7b"
