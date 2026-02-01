"""LanceDB-based pattern store for semantic pattern search.

Stores pattern embeddings for similarity search across learned patterns.
Follows the LanceVectorStore pattern from codesage.storage.lance_store.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from codesage.utils.logging import get_logger

from .models import LearnedPattern, PatternCategory

logger = get_logger("memory.pattern_store")

# Type alias for embedding function
EmbeddingFunction = Callable[[List[str]], List[List[float]]]


class PatternStore:
    """LanceDB store for semantic pattern search.

    Stores pattern embeddings to enable:
        - Semantic search over patterns
        - Finding similar patterns
        - Pattern clustering and analysis

    Attributes:
        TABLE_NAME: Name of the LanceDB table.
        VECTOR_DIM: Dimension of embedding vectors.
    """

    TABLE_NAME = "patterns"
    VECTOR_DIM = 1024  # mxbai-embed-large dimension

    def __init__(
        self,
        persist_dir: Union[str, Path],
        embedding_fn: Optional[EmbeddingFunction] = None,
        vector_dim: int = 1024,
    ) -> None:
        """Initialize the pattern store.

        Args:
            persist_dir: Directory for LanceDB persistence.
            embedding_fn: Function to generate embeddings from text.
            vector_dim: Dimension of embedding vectors.
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "LanceDB is required for the pattern store. "
                "Install with: pipx inject pycodesage lancedb (or pip install lancedb)"
            )

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_fn = embedding_fn
        self.vector_dim = vector_dim

        # Connect to LanceDB
        self._db = lancedb.connect(str(self.persist_dir))
        self._table = self._get_or_create_table()

    def _get_or_create_table(self):
        """Get existing table or create a new one.

        Returns:
            LanceDB table instance.
        """
        import pyarrow as pa

        if self.TABLE_NAME in self._db.table_names():
            return self._db.open_table(self.TABLE_NAME)

        # Define schema for patterns
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("category", pa.string()),
                pa.field("description", pa.string()),
                pa.field("pattern_text", pa.string()),
                pa.field("document", pa.string()),  # Combined text for embedding
                pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),
                pa.field("occurrence_count", pa.int32()),
                pa.field("confidence_score", pa.float32()),
            ]
        )

        return self._db.create_table(self.TABLE_NAME, schema=schema)

    def set_embedding_fn(self, embedding_fn: EmbeddingFunction) -> None:
        """Set the embedding function.

        Args:
            embedding_fn: Function to generate embeddings.
        """
        self._embedding_fn = embedding_fn

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If embedding function is not set.
        """
        if self._embedding_fn is None:
            raise ValueError("Embedding function not set. Call set_embedding_fn first.")

        # Support both callable functions and Langchain Embeddings objects
        if callable(self._embedding_fn):
            # If it's a direct callable
            return self._embedding_fn(texts)
        elif hasattr(self._embedding_fn, 'embed_documents'):
            # If it's a Langchain Embeddings object
            return self._embedding_fn.embed_documents(texts)
        else:
            raise TypeError(
                f"Embedding function must be callable or have embed_documents method, "
                f"got {type(self._embedding_fn)}"
            )

    def _get_embedding_text(self, pattern: LearnedPattern) -> str:
        """Get text representation for embedding.

        Args:
            pattern: Pattern to convert.

        Returns:
            Text suitable for embedding.
        """
        parts = [
            f"Pattern: {pattern.name}",
            f"Category: {pattern.category.value}",
            f"Description: {pattern.description}",
            f"Pattern: {pattern.pattern_text}",
        ]
        if pattern.examples:
            parts.append(f"Examples: {'; '.join(pattern.examples[:3])}")

        return "\n".join(parts)

    def add_pattern(self, pattern: LearnedPattern) -> None:
        """Add a pattern to the store.

        Args:
            pattern: Pattern to add.
        """
        self.add_patterns([pattern])

    def add_patterns(self, patterns: List[LearnedPattern]) -> None:
        """Add multiple patterns to the store.

        Args:
            patterns: Patterns to add.
        """
        if not patterns:
            return

        # Prepare documents for embedding
        documents = [self._get_embedding_text(p) for p in patterns]

        # Generate embeddings if function is available
        if self._embedding_fn is not None:
            embeddings = self._embed(documents)
        else:
            # Use zero vectors as placeholder
            embeddings = [[0.0] * self.vector_dim for _ in patterns]
            logger.warning("No embedding function set, using zero vectors")

        # Prepare data for insertion
        data = []
        for pattern, doc, embedding in zip(patterns, documents, embeddings):
            data.append(
                {
                    "id": pattern.id,
                    "name": pattern.name,
                    "category": pattern.category.value,
                    "description": pattern.description,
                    "pattern_text": pattern.pattern_text,
                    "document": doc,
                    "vector": embedding,
                    "occurrence_count": pattern.occurrence_count,
                    "confidence_score": pattern.confidence_score,
                }
            )

        # Add to table
        self._table.add(data)
        logger.debug(f"Added {len(data)} patterns to LanceDB")

    def search(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[PatternCategory] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar patterns.

        Args:
            query: Search query text.
            n_results: Number of results to return.
            category: Optional category filter.
            min_confidence: Minimum confidence score.

        Returns:
            List of matching patterns with similarity scores.
        """
        if self._embedding_fn is None:
            logger.warning("No embedding function set, cannot search")
            return []

        # Generate query embedding
        query_embedding = self._embed([query])[0]

        # Build search query
        search = self._table.search(query_embedding).limit(n_results)

        # Apply filters
        filters = []
        if category:
            filters.append(f"category = '{category.value}'")
        if min_confidence > 0:
            filters.append(f"confidence_score >= {min_confidence}")

        if filters:
            search = search.where(" AND ".join(filters))

        # Execute search
        try:
            results = search.to_pandas()
        except Exception as e:
            logger.warning(f"LanceDB search failed: {e}")
            return []

        if results.empty:
            return []

        # Convert to standard format
        output = []
        for _, row in results.iterrows():
            distance = row.get("_distance", 0)
            similarity = 1 / (1 + distance)

            output.append(
                {
                    "id": row.get("id", ""),
                    "name": row.get("name", ""),
                    "category": row.get("category", ""),
                    "description": row.get("description", ""),
                    "pattern_text": row.get("pattern_text", ""),
                    "occurrence_count": row.get("occurrence_count", 0),
                    "confidence_score": row.get("confidence_score", 0.0),
                    "distance": distance,
                    "similarity": similarity,
                }
            )

        return output

    def find_similar(
        self,
        pattern_id: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find patterns similar to a given pattern.

        Args:
            pattern_id: ID of the reference pattern.
            n_results: Number of similar patterns to return.

        Returns:
            List of similar patterns with similarity scores.
        """
        # Get the pattern's embedding
        try:
            results = self._table.search().where(f"id = '{pattern_id}'").limit(1).to_pandas()
        except Exception as e:
            logger.warning(f"Failed to get pattern {pattern_id}: {e}")
            return []

        if results.empty:
            return []

        # Get the vector from the result
        vector = results.iloc[0].get("vector")
        if vector is None:
            return []

        # Search for similar patterns (excluding the original)
        search = self._table.search(vector).limit(n_results + 1)

        try:
            similar = search.to_pandas()
        except Exception as e:
            logger.warning(f"Similar pattern search failed: {e}")
            return []

        # Filter out the original pattern and convert
        output = []
        for _, row in similar.iterrows():
            if row.get("id") == pattern_id:
                continue

            distance = row.get("_distance", 0)
            similarity = 1 / (1 + distance)

            output.append(
                {
                    "id": row.get("id", ""),
                    "name": row.get("name", ""),
                    "category": row.get("category", ""),
                    "description": row.get("description", ""),
                    "similarity": similarity,
                }
            )

            if len(output) >= n_results:
                break

        return output

    def get_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a pattern by ID.

        Args:
            pattern_id: Pattern ID.

        Returns:
            Pattern data or None if not found.
        """
        try:
            results = self._table.search().where(f"id = '{pattern_id}'").limit(1).to_pandas()
        except Exception as e:
            logger.warning(f"Failed to get pattern {pattern_id}: {e}")
            return None

        if results.empty:
            return None

        row = results.iloc[0]
        return {
            "id": row.get("id", ""),
            "name": row.get("name", ""),
            "category": row.get("category", ""),
            "description": row.get("description", ""),
            "pattern_text": row.get("pattern_text", ""),
            "occurrence_count": row.get("occurrence_count", 0),
            "confidence_score": row.get("confidence_score", 0.0),
        }

    def delete(self, pattern_ids: List[str]) -> None:
        """Delete patterns by ID.

        Args:
            pattern_ids: List of pattern IDs to delete.
        """
        if not pattern_ids:
            return

        id_list = ", ".join([f"'{id}'" for id in pattern_ids])
        self._table.delete(f"id IN ({id_list})")
        logger.debug(f"Deleted {len(pattern_ids)} patterns from LanceDB")

    def update_pattern(self, pattern: LearnedPattern) -> None:
        """Update a pattern (delete and re-add).

        Args:
            pattern: Updated pattern.
        """
        self.delete([pattern.id])
        self.add_pattern(pattern)

    def count(self) -> int:
        """Get total count of patterns.

        Returns:
            Number of patterns in the store.
        """
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    def clear(self) -> None:
        """Clear all patterns from the store."""
        if self.TABLE_NAME in self._db.table_names():
            self._db.drop_table(self.TABLE_NAME)
        self._table = self._get_or_create_table()
        logger.info("Cleared all patterns from PatternStore")

    def get_metrics(self) -> Dict[str, Any]:
        """Get storage metrics.

        Returns:
            Dictionary with storage statistics.
        """
        return {
            "backend": "lancedb",
            "table_name": self.TABLE_NAME,
            "pattern_count": self.count(),
            "persist_dir": str(self.persist_dir),
            "vector_dim": self.vector_dim,
            "has_embedding_fn": self._embedding_fn is not None,
        }
