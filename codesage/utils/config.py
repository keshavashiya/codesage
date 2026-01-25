"""Configuration management for CodeSage."""

import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
import yaml
import json


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "ollama"  # ollama, openai, anthropic
    model: str = "qwen2.5-coder:7b"
    embedding_model: str = "mxbai-embed-large"
    base_url: Optional[str] = "http://localhost:11434"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("CODESAGE_API_KEY"))
    temperature: float = 0.3
    max_tokens: int = 500

    # Production hardening: timeout and retry settings
    request_timeout: float = 30.0  # Timeout for LLM requests (seconds)
    connect_timeout: float = 5.0   # Timeout for initial connection (seconds)
    max_retries: int = 3           # Maximum retry attempts for transient failures

    def validate(self) -> None:
        """Validate LLM configuration."""
        if self.provider in ("openai", "anthropic") and not self.api_key:
            raise ValueError(
                f"{self.provider} provider requires CODESAGE_API_KEY environment variable"
            )

        # Validate timeout values
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass
class StorageConfig:
    """Storage configuration."""

    vector_store: str = "chromadb"  # chromadb, faiss, pinecone
    db_path: Optional[Path] = None
    chroma_path: Optional[Path] = None


@dataclass
class Config:
    """CodeSage configuration."""

    project_name: str
    project_path: Path
    language: str = "python"
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    exclude_dirs: List[str] = field(default_factory=lambda: [
        "venv", "env", ".venv", ".env",
        "node_modules",
        ".git",
        "__pycache__", ".pytest_cache", ".mypy_cache",
        "build", "dist", "*.egg-info",
        ".codesage",
        ".tox", ".nox",
    ])
    include_extensions: List[str] = field(default_factory=lambda: [
        ".py",
    ])

    def __post_init__(self):
        """Post-initialization processing."""
        self.project_path = Path(self.project_path).resolve()

        if self.storage.db_path is None:
            self.storage.db_path = self.project_path / ".codesage" / "codesage.db"

        if self.storage.chroma_path is None:
            self.storage.chroma_path = self.project_path / ".codesage" / "chromadb"

    @property
    def codesage_dir(self) -> Path:
        """Get .codesage directory path."""
        return self.project_path / ".codesage"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return self.codesage_dir / "cache"

    @classmethod
    def load(cls, project_path: Path) -> "Config":
        """Load configuration from .codesage/config.yaml or config.json."""
        project_path = Path(project_path).resolve()
        config_yaml = project_path / ".codesage" / "config.yaml"
        config_json = project_path / ".codesage" / "config.json"

        if config_yaml.exists():
            with open(config_yaml) as f:
                data = yaml.safe_load(f) or {}
        elif config_json.exists():
            with open(config_json) as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(
                f"Config not found in {project_path}. Run 'codesage init' first."
            )

        # Build nested configs
        llm_data = data.pop("llm", {})
        storage_data = data.pop("storage", {})

        return cls(
            project_path=project_path,
            llm=LLMConfig(**llm_data),
            storage=StorageConfig(**storage_data),
            **data
        )

    def save(self) -> None:
        """Save configuration to .codesage/config.yaml."""
        config_dir = self.codesage_dir
        config_dir.mkdir(parents=True, exist_ok=True)

        # Also create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        config_file = config_dir / "config.yaml"

        data = {
            "project_name": self.project_name,
            "language": self.language,
            "exclude_dirs": self.exclude_dirs,
            "include_extensions": self.include_extensions,
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "embedding_model": self.llm.embedding_model,
                "base_url": self.llm.base_url,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                # Production settings
                "request_timeout": self.llm.request_timeout,
                "connect_timeout": self.llm.connect_timeout,
                "max_retries": self.llm.max_retries,
            },
            "storage": {
                "vector_store": self.storage.vector_store,
            }
        }

        # Don't save api_key to file
        with open(config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def initialize_project(
    project_path: Path,
    model: str = "qwen2.5-coder:7b",
    embedding_model: str = "mxbai-embed-large",
) -> Config:
    """Initialize CodeSage in a project directory."""
    project_path = Path(project_path).resolve()

    # Create .codesage directory
    codesage_dir = project_path / ".codesage"
    codesage_dir.mkdir(parents=True, exist_ok=True)

    # Create cache directory
    cache_dir = codesage_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Create config
    config = Config(
        project_name=project_path.name,
        project_path=project_path,
        llm=LLMConfig(
            model=model,
            embedding_model=embedding_model,
        ),
    )

    config.save()

    return config
