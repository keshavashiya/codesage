"""LangChain embeddings for code vectorization."""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path
import hashlib
import json

from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

from codesage.utils.config import Config, LLMConfig
from codesage.utils.retry import retry_with_backoff
from codesage.utils.logging import get_logger

logger = get_logger("llm.embeddings")


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


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
    # mxbai-embed-large: 512 tokens (~1500 chars)
    # nomic-embed-text: 8192 tokens (~24000 chars)
    # We use a conservative limit that works for most models
    MAX_CHARS = 1500

    def __init__(self, config: LLMConfig, cache_dir: Path):
        """Initialize the embedding service.

        Args:
            config: LLM configuration
            cache_dir: Directory for embedding cache
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._embedder: Embeddings = self._init_embedder()

        # Create retry decorator based on config
        self._retry = retry_with_backoff(
            max_retries=config.max_retries,
            base_delay=1.0,
            max_delay=30.0,
            exceptions=(ConnectionError, TimeoutError, OSError, EmbeddingError),
            on_retry=lambda e, attempt: logger.warning(
                f"Embedding call failed, retrying (attempt {attempt + 1}): {e}"
            ),
        )

    def _init_embedder(self) -> Embeddings:
        """Initialize the LangChain embeddings based on provider."""
        # Note: OllamaEmbeddings doesn't directly support timeout param,
        # but the underlying httpx client respects system timeouts
        if self.config.provider == "ollama":
            return OllamaEmbeddings(
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
            return OllamaEmbeddings(
                model=self.config.embedding_model,
                base_url=self.config.base_url or "http://localhost:11434",
            )

    @property
    def embedder(self) -> Embeddings:
        """Get the underlying LangChain embedder."""
        return self._embedder

    def _truncate(self, text: str) -> str:
        """Truncate text to fit within embedding model context.

        Args:
            text: Text to truncate

        Returns:
            Truncated text
        """
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

        if use_cache:
            cache_key = self._hash(text)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

        # Generate embedding with retry
        @self._retry
        def _generate_embedding():
            try:
                return self._embedder.embed_query(text)
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                raise EmbeddingError(f"Failed to generate embedding: {e}") from e

        try:
            embedding = _generate_embedding()
        except Exception as e:
            logger.error(f"Embedding failed after retries: {e}")
            raise EmbeddingError(f"Embedding failed: {e}") from e

        if use_cache:
            self._save_to_cache(cache_key, embedding)

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

        if not use_cache:
            @self._retry
            def _embed_all():
                try:
                    return self._embedder.embed_documents(texts)
                except Exception as e:
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
    return EmbeddingService(config.llm, config.cache_dir)
