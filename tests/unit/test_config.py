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
        assert config.embedding_model == "nomic-embed-text"
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


class TestEnhancedOllamaConfig:
    """Tests for enhanced Ollama configuration."""

    def test_all_enhanced_parameters(self):
        """Test that all enhanced parameters can be set."""
        config = LLMConfig(
            context_window=16384,
            num_predict=256,
            top_k=50,
            top_p=0.95,
            repeat_penalty=1.2,
            repeat_last_n=128,
            token_bucket_size=2000,
            token_bucket_refill_rate=120,
            chunk_size=1024,
            chunk_overlap=100,
            stop_sequences=["\n\n", "```", "###", "---"],
            model_parameters={"num_thread": 8, "num_gpu": 2, "low_vram": True},
        )

        assert config.context_window == 16384
        assert config.num_predict == 256
        assert config.top_k == 50
        assert config.top_p == 0.95
        assert config.repeat_penalty == 1.2
        assert config.repeat_last_n == 128
        assert config.token_bucket_size == 2000
        assert config.token_bucket_refill_rate == 120
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.stop_sequences == ["\n\n", "```", "###", "---"]
        assert config.model_parameters == {
            "num_thread": 8,
            "num_gpu": 2,
            "low_vram": True,
        }

    def test_default_enhanced_parameters(self):
        """Test that enhanced parameters have sensible defaults."""
        config = LLMConfig()

        assert config.context_window == 32768
        assert config.num_predict == 128
        assert config.top_k == 40
        assert config.top_p == 0.9
        assert config.repeat_penalty == 1.1
        assert config.repeat_last_n == 64
        assert config.token_bucket_size == 1000
        assert config.token_bucket_refill_rate == 60
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.stop_sequences == ["\n\n", "```"]
        assert config.model_parameters == {}

    def test_parameter_validation_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            config = LLMConfig(temperature=3.0)
            config.validate_ollama_config()

        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            config = LLMConfig(temperature=-0.1)
            config.validate_ollama_config()

    def test_parameter_validation_top_p(self):
        """Test top_p validation."""
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            config = LLMConfig(top_p=1.5)
            config.validate_ollama_config()

        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            config = LLMConfig(top_p=-0.1)
            config.validate_ollama_config()

    def test_parameter_validation_top_k(self):
        """Test top_k validation."""
        with pytest.raises(ValueError, match="top_k must be non-negative"):
            config = LLMConfig(top_k=-1)
            config.validate_ollama_config()

    def test_parameter_validation_chunk_size(self):
        """Test chunk_size validation."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config = LLMConfig(chunk_size=0)
            config.validate_ollama_config()

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config = LLMConfig(chunk_size=-100)
            config.validate_ollama_config()

    def test_parameter_validation_chunk_overlap(self):
        """Test chunk_overlap validation."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            config = LLMConfig(chunk_overlap=-1)
            config.validate_ollama_config()

        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            config = LLMConfig(chunk_size=100, chunk_overlap=150)
            config.validate_ollama_config()

    def test_parameter_validation_token_bucket(self):
        """Test token bucket validation."""
        with pytest.raises(ValueError, match="token_bucket_size must be positive"):
            config = LLMConfig(token_bucket_size=0)
            config.validate_ollama_config()

        with pytest.raises(
            ValueError, match="token_bucket_refill_rate must be positive"
        ):
            config = LLMConfig(token_bucket_refill_rate=0)
            config.validate_ollama_config()

    def test_parameter_validation_context_window(self):
        """Test context_window validation."""
        with pytest.raises(ValueError, match="context_window must be positive"):
            config = LLMConfig(context_window=0)
            config.validate_ollama_config()

    def test_parameter_validation_num_predict(self):
        """Test num_predict validation."""
        with pytest.raises(ValueError, match="num_predict must be non-negative"):
            config = LLMConfig(num_predict=-1)
            config.validate_ollama_config()

    def test_backward_compatibility_old_config(self):
        """Test backward compatibility with old config without enhanced fields."""
        old_config_data = {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "embedding_model": "qwen3-embedding",
            "base_url": "http://localhost:11434",
            "temperature": 0.3,
            "max_tokens": 500,
            "request_timeout": 30.0,
            "connect_timeout": 5.0,
            "max_retries": 3,
        }

        config = LLMConfig(**old_config_data)

        # Old fields should be set
        assert config.model == "qwen2.5-coder:7b"
        assert config.temperature == 0.3

        # New fields should have defaults
        assert config.context_window == 32768
        assert config.chunk_size == 512
        assert config.token_bucket_size == 1000

        # Validation should pass
        config.validate()

    def test_enhanced_config_save_load(self, tmp_path):
        """Test saving and loading config with enhanced parameters."""
        config = Config(
            project_name="test-enhanced",
            project_path=tmp_path,
            llm=LLMConfig(
                model="qwen2.5-coder:14b",
                context_window=16384,
                num_predict=256,
                top_k=50,
                top_p=0.95,
                repeat_penalty=1.2,
                repeat_last_n=128,
                token_bucket_size=2000,
                token_bucket_refill_rate=120,
                chunk_size=1024,
                chunk_overlap=100,
                stop_sequences=["\n\n", "```", "###", "---"],
                model_parameters={"num_thread": 8, "num_gpu": 2},
            ),
        )

        config.save()
        loaded = Config.load(tmp_path)

        # Verify all enhanced parameters
        assert loaded.llm.model == "qwen2.5-coder:14b"
        assert loaded.llm.context_window == 16384
        assert loaded.llm.num_predict == 256
        assert loaded.llm.top_k == 50
        assert loaded.llm.top_p == 0.95
        assert loaded.llm.repeat_penalty == 1.2
        assert loaded.llm.repeat_last_n == 128
        assert loaded.llm.token_bucket_size == 2000
        assert loaded.llm.token_bucket_refill_rate == 120
        assert loaded.llm.chunk_size == 1024
        assert loaded.llm.chunk_overlap == 100
        assert loaded.llm.stop_sequences == ["\n\n", "```", "###", "---"]
        assert loaded.llm.model_parameters == {"num_thread": 8, "num_gpu": 2}

    def test_non_ollama_provider_skip_validation(self):
        """Test that non-Ollama providers skip Ollama-specific validation."""
        # This should not raise even with invalid Ollama parameters
        # because validation is provider-specific
        config = LLMConfig(
            provider="openai",
            api_key="test-key",
            top_p=1.5,  # Invalid for Ollama but provider skips validation
        )

        # Only validates API key for non-Ollama
        config.validate()

    def test_validation_skipped_for_non_ollama(self):
        """Test that validate_ollama_config is skipped for non-Ollama providers."""
        config = LLMConfig(
            provider="openai",
            api_key="test-key",
            temperature=3.0,  # Invalid for Ollama
            top_p=1.5,  # Invalid for Ollama
            chunk_size=-100,  # Invalid
        )

        # These would fail for Ollama but should pass for OpenAI
        # because validate_ollama_config returns early
        config.validate_ollama_config()
        # No exception raised
