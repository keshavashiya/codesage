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

    # Backend selection
    vector_backend: str = "lancedb"  # LanceDB is the default and only vector store
    use_graph: bool = True           # Enable KuzuDB graph store

    # Paths (auto-set in Config.__post_init__)
    db_path: Optional[Path] = None       # SQLite
    lance_path: Optional[Path] = None    # LanceDB
    kuzu_path: Optional[Path] = None     # KuzuDB


@dataclass
class SecurityConfig:
    """Security scanning configuration."""

    enabled: bool = True
    severity_threshold: str = "medium"  # low, medium, high, critical
    block_on_critical: bool = True
    custom_patterns: List[str] = field(default_factory=list)
    ignore_rules: List[str] = field(default_factory=list)  # Rule IDs to skip


@dataclass
class HooksConfig:
    """Git hooks configuration."""

    pre_commit_enabled: bool = True
    run_security_scan: bool = True
    run_review: bool = False
    severity_threshold: str = "medium"


@dataclass
class MemoryConfig:
    """Developer memory configuration."""

    enabled: bool = True  # Enable memory learning
    global_dir: Optional[Path] = None  # Global memory directory (default: ~/.codesage/developer)
    learn_on_index: bool = True  # Learn patterns during indexing
    min_pattern_confidence: float = 0.5  # Minimum confidence for patterns
    min_pattern_occurrences: int = 2  # Minimum occurrences to store pattern


@dataclass
class Config:
    """CodeSage configuration."""

    project_name: str
    project_path: Path
    languages: List[str] = field(default_factory=lambda: ["python"])
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    exclude_dirs: List[str] = field(default_factory=lambda: [
        # Version control
        ".git", ".svn", ".hg",
        # Python
        "venv", "env", ".venv", ".env",
        "__pycache__", ".pytest_cache", ".mypy_cache",
        "build", "dist", "*.egg-info",
        ".tox", ".nox",
        # JavaScript/TypeScript
        "node_modules", ".next", ".nuxt",
        # Go
        "vendor",
        # Rust
        "target",
        # IDE/Tools
        ".codesage", ".idea", ".vscode",
    ])
    include_extensions: List[str] = field(default_factory=lambda: [
        # Python
        ".py",
        # JavaScript
        ".js", ".jsx", ".mjs", ".cjs",
        # TypeScript
        ".ts", ".tsx", ".mts", ".cts",
        # Go
        ".go",
        # Rust
        ".rs",
    ])

    def __post_init__(self):
        """Post-initialization processing."""
        self.project_path = Path(self.project_path).resolve()

        # Backward compatibility: convert 'language' string to 'languages' list
        if hasattr(self, 'language') and isinstance(getattr(self, 'language', None), str):
            lang = getattr(self, 'language')
            if lang and lang not in self.languages:
                self.languages = [lang] + [l for l in self.languages if l != lang]

        # Set default storage paths
        if self.storage.db_path is None:
            self.storage.db_path = self.project_path / ".codesage" / "codesage.db"

        if self.storage.lance_path is None:
            self.storage.lance_path = self.project_path / ".codesage" / "lancedb"

        if self.storage.kuzu_path is None:
            self.storage.kuzu_path = self.project_path / ".codesage" / "kuzudb"

    @property
    def language(self) -> str:
        """Get primary language (backward compatibility)."""
        return self.languages[0] if self.languages else "python"

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

        # Backward compatibility: convert old 'language' to 'languages'
        if "language" in data and "languages" not in data:
            data["languages"] = [data.pop("language")]
        elif "language" in data:
            data.pop("language")  # Remove old field if both present

        # Build nested configs
        llm_data = data.pop("llm", {})
        storage_data = data.pop("storage", {})
        security_data = data.pop("security", {})
        hooks_data = data.pop("hooks", {})
        memory_data = data.pop("memory", {})

        return cls(
            project_path=project_path,
            llm=LLMConfig(**llm_data),
            storage=StorageConfig(**storage_data),
            security=SecurityConfig(**security_data),
            hooks=HooksConfig(**hooks_data),
            memory=MemoryConfig(**memory_data),
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
            "languages": self.languages,
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
                "vector_backend": self.storage.vector_backend,
                "use_graph": self.storage.use_graph,
            },
            "security": {
                "enabled": self.security.enabled,
                "severity_threshold": self.security.severity_threshold,
                "block_on_critical": self.security.block_on_critical,
                "custom_patterns": self.security.custom_patterns,
                "ignore_rules": self.security.ignore_rules,
            },
            "hooks": {
                "pre_commit_enabled": self.hooks.pre_commit_enabled,
                "run_security_scan": self.hooks.run_security_scan,
                "run_review": self.hooks.run_review,
                "severity_threshold": self.hooks.severity_threshold,
            },
            "memory": {
                "enabled": self.memory.enabled,
                "learn_on_index": self.memory.learn_on_index,
                "min_pattern_confidence": self.memory.min_pattern_confidence,
                "min_pattern_occurrences": self.memory.min_pattern_occurrences,
            },
        }

        # Don't save api_key to file
        with open(config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def initialize_project(
    project_path: Path,
    model: str = "qwen2.5-coder:7b",
    embedding_model: str = "mxbai-embed-large",
    auto_detect: bool = True,
) -> Config:
    """Initialize CodeSage in a project directory.

    Args:
        project_path: Path to the project root
        model: Ollama model for analysis
        embedding_model: Model for embeddings
        auto_detect: Whether to auto-detect languages

    Returns:
        Initialized Config object
    """
    project_path = Path(project_path).resolve()

    # Create .codesage directory
    codesage_dir = project_path / ".codesage"
    codesage_dir.mkdir(parents=True, exist_ok=True)

    # Create cache directory
    cache_dir = codesage_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Auto-detect languages if enabled
    languages = ["python"]  # Default
    detected_info = []

    if auto_detect:
        try:
            from codesage.utils.language_detector import detect_languages
            detected_info = detect_languages(project_path)
            if detected_info:
                languages = [lang.name for lang in detected_info]
        except Exception:
            pass  # Fall back to default

    # Create config
    config = Config(
        project_name=project_path.name,
        project_path=project_path,
        languages=languages,
        llm=LLMConfig(
            model=model,
            embedding_model=embedding_model,
        ),
    )

    config.save()

    return config
