"""Configuration management for CodeSage."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import yaml
import json


@dataclass
class LLMConfig:
    """Enhanced LLM provider configuration with full Ollama support."""

    provider: str = "ollama"  # ollama, openai, anthropic
    model: str = "qwen2.5-coder:7b"
    embedding_model: str = "nomic-embed-text"
    base_url: Optional[str] = "http://localhost:11434"
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("CODESAGE_API_KEY")
    )
    temperature: float = 0.3
    max_tokens: int = 500

    # Production hardening: timeout and retry settings
    request_timeout: float = 30.0  # Timeout for LLM requests (seconds)
    connect_timeout: float = 5.0  # Timeout for initial connection (seconds)
    max_retries: int = 3  # Maximum retry attempts for transient failures

    # Advanced model parameters (Ollama-specific)
    context_window: int = 32768  # Maximum context window size
    num_predict: int = 128  # Maximum tokens to predict
    top_k: int = 40  # Top-k sampling parameter (0-100)
    top_p: float = 0.9  # Top-p sampling parameter (0.0-1.0)
    repeat_penalty: float = 1.1  # Repetition penalty (1.0 = disabled)
    repeat_last_n: int = 64  # Number of tokens to consider for repetition

    # Token management
    token_bucket_size: int = 1000  # Token bucket size for rate limiting
    token_bucket_refill_rate: int = 60  # Tokens per minute refill rate

    # Text processing
    chunk_size: int = 512  # Text chunking size for embeddings
    chunk_overlap: int = 50  # Overlap between chunks (tokens)

    # Custom sequences - empty by default to avoid truncation
    # Can be configured for specific use cases (e.g., ["```"] for code blocks)
    stop_sequences: List[str] = field(default_factory=list)

    # Model-specific parameters (passed directly to Ollama)
    model_parameters: Dict[str, Any] = field(default_factory=dict)

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

        # Validate Ollama-specific parameters
        self.validate_ollama_config()

    def validate_ollama_config(self) -> None:
        """Validate Ollama-specific configuration parameters."""
        if self.provider != "ollama":
            return

        # Validate parameter ranges
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")

        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        if self.token_bucket_size <= 0:
            raise ValueError("token_bucket_size must be positive")

        if self.token_bucket_refill_rate <= 0:
            raise ValueError("token_bucket_refill_rate must be positive")

        if self.context_window <= 0:
            raise ValueError("context_window must be positive")

        if self.num_predict < 0:
            raise ValueError("num_predict must be non-negative")

        if self.repeat_penalty < 0:
            raise ValueError("repeat_penalty must be non-negative")

        if self.repeat_last_n < 0:
            raise ValueError("repeat_last_n must be non-negative")


@dataclass
class StorageConfig:
    """Storage configuration."""

    # Backend selection
    vector_backend: str = "lancedb"  # LanceDB is the default and only vector store
    use_graph: bool = True  # Enable KuzuDB graph store

    # Paths (auto-set in Config.__post_init__)
    db_path: Optional[Path] = None  # SQLite
    lance_path: Optional[Path] = None  # LanceDB
    kuzu_path: Optional[Path] = None  # KuzuDB


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
    global_dir: Optional[Path] = (
        None  # Global memory directory (default: ~/.codesage/developer)
    )
    learn_on_index: bool = True  # Learn patterns during indexing
    min_pattern_confidence: float = 0.5  # Minimum confidence for patterns
    min_pattern_occurrences: int = 2  # Minimum occurrences to store pattern


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""

    embedding_batch_size: int = 200  # Files per embedding batch
    max_elements_per_batch: int = 200  # Cap to avoid OOM
    embedding_cache_size: int = 1000  # In-memory LRU size for query embeddings
    cache_enabled: bool = True  # Toggle disk + memory embedding caches


@dataclass
class FeaturesConfig:
    """Feature flags for experimental capabilities."""

    # Core features
    embeddings: bool = True  # Local embeddings for semantic search
    memory: bool = True  # Developer memory and pattern learning
    llm_explanations: bool = True  # LLM-powered explanations
    graph_storage: bool = True  # Graph-based code relationships

    # Advanced features
    context_provider_mode: bool = False
    graph_enriched_search: bool = False
    code_smell_detection: bool = False
    docs_generation: bool = False
    cross_project_recommendations: bool = False


@dataclass
class DocsConfig:
    """Documentation generation configuration."""

    output_dir: str = "docs"
    format: str = "markdown"


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
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    docs: DocsConfig = field(default_factory=DocsConfig)
    exclude_dirs: List[str] = field(
        default_factory=lambda: [
            # Version control
            ".git",
            ".svn",
            ".hg",
            # Python
            "venv",
            "env",
            ".venv",
            ".env",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            "*.egg-info",
            ".tox",
            ".nox",
            # JavaScript/TypeScript
            "node_modules",
            ".next",
            ".nuxt",
            # Go
            "vendor",
            # Rust
            "target",
            # IDE/Tools
            ".codesage",
            ".idea",
            ".vscode",
        ]
    )
    include_extensions: List[str] = field(
        default_factory=lambda: [
            # Python
            ".py",
            # JavaScript
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",
            # TypeScript
            ".ts",
            ".tsx",
            ".mts",
            ".cts",
            # Go
            ".go",
            # Rust
            ".rs",
        ]
    )

    def __post_init__(self):
        """Post-initialization processing."""
        self.project_path = Path(self.project_path).resolve()

        # Backward compatibility: convert 'language' string to 'languages' list
        if hasattr(self, "language") and isinstance(
            getattr(self, "language", None), str
        ):
            lang = getattr(self, "language")
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
        performance_data = data.pop("performance", {})
        features_data = data.pop("features", {})
        docs_data = data.pop("docs", {})

        return cls(
            project_path=project_path,
            llm=LLMConfig(**llm_data),
            storage=StorageConfig(**storage_data),
            security=SecurityConfig(**security_data),
            hooks=HooksConfig(**hooks_data),
            memory=MemoryConfig(**memory_data),
            performance=PerformanceConfig(**performance_data),
            features=FeaturesConfig(**features_data),
            docs=DocsConfig(**docs_data),
            **data,
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
                # Advanced Ollama parameters
                "context_window": self.llm.context_window,
                "num_predict": self.llm.num_predict,
                "top_k": self.llm.top_k,
                "top_p": self.llm.top_p,
                "repeat_penalty": self.llm.repeat_penalty,
                "repeat_last_n": self.llm.repeat_last_n,
                # Token management
                "token_bucket_size": self.llm.token_bucket_size,
                "token_bucket_refill_rate": self.llm.token_bucket_refill_rate,
                # Text processing
                "chunk_size": self.llm.chunk_size,
                "chunk_overlap": self.llm.chunk_overlap,
                # Custom sequences
                "stop_sequences": self.llm.stop_sequences,
                # Model-specific parameters
                "model_parameters": self.llm.model_parameters,
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
            "performance": {
                "embedding_batch_size": self.performance.embedding_batch_size,
                "max_elements_per_batch": self.performance.max_elements_per_batch,
                "embedding_cache_size": self.performance.embedding_cache_size,
                "cache_enabled": self.performance.cache_enabled,
            },
            "features": {
                "context_provider_mode": self.features.context_provider_mode,
                "graph_enriched_search": self.features.graph_enriched_search,
                "code_smell_detection": self.features.code_smell_detection,
                "docs_generation": self.features.docs_generation,
                "cross_project_recommendations": self.features.cross_project_recommendations,
            },
            "docs": {
                "output_dir": self.docs.output_dir,
                "format": self.docs.format,
            },
        }

        # Don't save api_key to file
        with open(config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def initialize_project(
    project_path: Path,
    model: str = "qwen2.5-coder:7b",
    embedding_model: str = "nomic-embed-text",
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
