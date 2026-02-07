"""LangChain embeddings for code vectorization."""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path
import hashlib
import json
import re
from collections import OrderedDict

from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

from codesage.utils.config import Config, LLMConfig, PerformanceConfig
from codesage.utils.retry import retry_with_backoff
from codesage.utils.logging import get_logger

logger = get_logger("llm.embeddings")

# Embedding models that require an instruction prefix to avoid NaN on
# multi-line or code inputs.  Maps model-name prefix → task instruction.
_INSTRUCTION_MODELS = {
    "qwen3-embedding": "Represent this text for retrieval: ",
}


def _needs_instruction_prefix(model_name: str) -> Optional[str]:
    """Return the instruction prefix for models that need one, else None."""
    for prefix, instruction in _INSTRUCTION_MODELS.items():
        if model_name.startswith(prefix):
            return instruction
    return None


class _InstructionEmbeddings(Embeddings):
    """Wraps an Embeddings instance to prepend an instruction prefix.

    Some embedding models (e.g. qwen3-embedding) produce NaN vectors when
    given raw multi-line code without an instruction prefix.  This wrapper
    transparently prepends the required prefix.
    """

    def __init__(self, delegate: Embeddings, instruction: str) -> None:
        self._delegate = delegate
        self._instruction = instruction

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [self._instruction + t for t in texts]
        return self._delegate.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return self._delegate.embed_query(self._instruction + text)


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class EmbeddingModelError(EmbeddingError):
    """Non-retryable model error (e.g. NaN output, invalid input)."""
    pass


# Patterns that indicate a deterministic model failure (never succeeds on retry)
_NON_RETRYABLE_PATTERNS = re.compile(
    r"NaN|unsupported value|invalid input|model not found",
    re.IGNORECASE,
)


class EmbeddingService:
    """Embedding service with LangChain and caching.

    Uses LangChain's embedding interfaces for generating
    vector representations of code elements.

    Features:
        - Automatic retry with exponential backoff
        - File-based embedding cache
        - Configurable timeouts
    """

    # Max characters to embed
    # qwen3-embedding: 32K tokens (~96000 chars) — default model
    # mxbai-embed-large: 512 tokens (~1500 chars)
    # nomic-embed-text: 8192 tokens (~24000 chars)
    # 8000 chars covers most functions/classes without being wasteful
    MAX_CHARS = 8000

    def __init__(
        self,
        config: LLMConfig,
        cache_dir: Path,
        performance: Optional[PerformanceConfig] = None,
    ):
        """Initialize the embedding service.

        Args:
            config: LLM configuration
            cache_dir: Directory for embedding cache
            performance: Optional performance configuration
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.performance = performance or PerformanceConfig()
        self._cache_enabled = bool(self.performance.cache_enabled)

        self._embedder: Embeddings = self._init_embedder()
        self._memory_cache: Optional[_EmbeddingLRU] = None
        if self._cache_enabled and self.performance.embedding_cache_size > 0:
            self._memory_cache = _EmbeddingLRU(self.performance.embedding_cache_size)

        # Create retry decorator based on config.
        # EmbeddingModelError is excluded — it's a deterministic failure (NaN etc.)
        # that will never succeed on retry.
        self._retry = retry_with_backoff(
            max_retries=config.max_retries,
            base_delay=1.0,
            max_delay=30.0,
            exceptions=(ConnectionError, TimeoutError, OSError, EmbeddingError),
            exclude_exceptions=(EmbeddingModelError,),
            on_retry=lambda e, attempt: logger.warning(
                f"Embedding call failed, retrying (attempt {attempt + 1}): {e}"
            ),
        )

    def _init_embedder(self) -> Embeddings:
        """Initialize the LangChain embeddings based on provider."""
        # Note: OllamaEmbeddings doesn't directly support timeout param,
        # but the underlying httpx client respects system timeouts
        if self.config.provider == "ollama":
            base = OllamaEmbeddings(
                model=self.config.embedding_model,
                base_url=self.config.base_url,
            )
        elif self.config.provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model=self.config.embedding_model,
                    api_key=self.config.api_key,
                    timeout=self.config.request_timeout,
                    max_retries=0,  # We handle retries ourselves
                )
            except ImportError:
                raise ImportError(
                    "OpenAI embeddings require langchain-openai. "
                    "Install with: pipx inject pycodesage 'pycodesage[openai]' (or pip install 'pycodesage[openai]')"
                )
        else:
            # Default to Ollama for other providers
            base = OllamaEmbeddings(
                model=self.config.embedding_model,
                base_url=self.config.base_url or "http://localhost:11434",
            )

        # Wrap with instruction prefix for models that need it (e.g. qwen3-embedding)
        instruction = _needs_instruction_prefix(self.config.embedding_model)
        if instruction:
            logger.info(f"Using instruction prefix for {self.config.embedding_model}")
            return _InstructionEmbeddings(base, instruction)
        return base

    @property
    def embedder(self) -> Embeddings:
        """Get the underlying LangChain embedder."""
        return self._embedder

    def get_dimension(self) -> int:
        """Get the embedding vector dimension by probing the model.

        Embeds a short test string and caches the result.

        Returns:
            Integer dimension of the embedding vectors.
        """
        if not hasattr(self, "_vector_dim"):
            try:
                vec = self._embedder.embed_query("dimension probe")
                self._vector_dim = len(vec)
                logger.info(f"Detected embedding dimension: {self._vector_dim}")
            except Exception as e:
                logger.warning(f"Failed to probe embedding dimension, defaulting to 4096: {e}")
                self._vector_dim = 4096
        return self._vector_dim

    @staticmethod
    def _sanitize(text: str) -> str:
        """Sanitize text to prevent model errors (NaN, encoding issues).

        Removes NUL bytes, control characters, and normalizes whitespace
        that can cause embedding models to produce NaN output.

        Args:
            text: Raw text to sanitize

        Returns:
            Cleaned text safe for embedding
        """
        # Remove NUL bytes and other control characters (keep \n, \t, \r)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Collapse runs of blank lines (>3 consecutive newlines → 2)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        # Ensure non-empty (models can NaN on empty string)
        if not text:
            text = "empty"
        return text

    def _truncate(self, text: str) -> str:
        """Sanitize and truncate text to fit within embedding model context.

        Args:
            text: Text to truncate

        Returns:
            Cleaned, truncated text
        """
        text = self._sanitize(text)
        if len(text) <= self.MAX_CHARS:
            return text
        # Truncate and add indicator
        return text[:self.MAX_CHARS - 20] + "\n... [truncated]"

    def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails after retries
        """
        # Truncate if needed
        text = self._truncate(text)

        use_cache = use_cache and self._cache_enabled

        if use_cache:
            cache_key = self._hash(text)
            if self._memory_cache:
                cached_mem = self._memory_cache.get(cache_key)
                if cached_mem is not None:
                    return cached_mem
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                if self._memory_cache:
                    self._memory_cache.set(cache_key, cached)
                return cached

        # Generate embedding with retry (skips retry for deterministic model errors)
        @self._retry
        def _generate_embedding():
            try:
                return self._embedder.embed_query(text)
            except Exception as e:
                error_str = str(e)
                logger.error(f"Embedding generation failed: {e}")
                # Don't retry deterministic model errors (NaN, invalid input)
                if _NON_RETRYABLE_PATTERNS.search(error_str):
                    raise EmbeddingModelError(
                        f"Model error (not retryable): {e}"
                    ) from e
                raise EmbeddingError(f"Failed to generate embedding: {e}") from e

        try:
            embedding = _generate_embedding()
        except EmbeddingModelError:
            raise  # Don't wrap model errors further
        except Exception as e:
            logger.error(f"Embedding failed after retries: {e}")
            raise EmbeddingError(f"Embedding failed: {e}") from e

        if use_cache:
            self._save_to_cache(cache_key, embedding)
            if self._memory_cache:
                self._memory_cache.set(cache_key, embedding)

        return embedding

    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache

        Returns:
            List of embedding vectors
        """
        # Truncate all texts first
        texts = [self._truncate(text) for text in texts]

        use_cache = use_cache and self._cache_enabled

        if len(texts) == 1:
            # Route through single-embed path for query caching
            return [self.embed(texts[0], use_cache=use_cache)]

        if not use_cache:
            @self._retry
            def _embed_all():
                try:
                    return self._embedder.embed_documents(texts)
                except Exception as e:
                    error_str = str(e)
                    if _NON_RETRYABLE_PATTERNS.search(error_str):
                        raise EmbeddingModelError(
                            f"Model error (not retryable): {e}"
                        ) from e
                    raise EmbeddingError(f"Batch embedding failed: {e}") from e
            return _embed_all()

        # Check cache for each text
        embeddings: List[List[float] | None] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cache_key = self._hash(text)
            cached = self._get_from_cache(cache_key)

            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # type: ignore
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts with retry
        if uncached_texts:
            @self._retry
            def _embed_uncached():
                try:
                    return self._embedder.embed_documents(uncached_texts)
                except Exception as e:
                    error_str = str(e)
                    if _NON_RETRYABLE_PATTERNS.search(error_str):
                        raise EmbeddingModelError(
                            f"Model error (not retryable): {e}"
                        ) from e
                    raise EmbeddingError(f"Batch embedding failed: {e}") from e

            new_embeddings = _embed_uncached()

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                embeddings[idx] = emb
                self._save_to_cache(self._hash(text), emb)

        return embeddings  # type: ignore

    def _hash(self, text: str) -> str:
        """Generate cache key from text."""
        # Include model in hash for cache invalidation on model change
        key = f"{self.config.embedding_model}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_to_cache(self, key: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        cache_file = self.cache_dir / f"{key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(embedding, f)
        except IOError:
            pass  # Cache failures are not critical

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception:
                pass


def create_embedding_service(config: Config) -> EmbeddingService:
    """Factory function to create an embedding service from config.

    Args:
        config: Full CodeSage config

    Returns:
        Configured embedding service
    """
    return EmbeddingService(config.llm, config.cache_dir, config.performance)


class _EmbeddingLRU:
    """Simple LRU cache for embeddings."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._data: "OrderedDict[str, List[float]]" = OrderedDict()

    def get(self, key: str) -> Optional[List[float]]:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: str, value: List[float]) -> None:
        if self.max_size <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)
