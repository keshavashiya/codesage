"""LanceDB vector store for efficient code search.

LanceDB provides 10x better memory efficiency and query performance
compared to ChromaDB, making it ideal for local-first code intelligence.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from codesage.storage.vector_base import VectorStoreBase
from codesage.utils.logging import get_logger

logger = get_logger("storage.lance")

# Type alias for embedding function
EmbeddingFunction = Callable[[List[str]], List[List[float]]]


class LanceVectorStore(VectorStoreBase):
    """LanceDB vector store for semantic code search.

    Advantages over ChromaDB:
        - 10x lower memory usage (50MB vs 500MB for 100K vectors)
        - 10x faster queries (~20ms vs ~200ms)
        - 10x smaller disk footprint
        - Native hybrid search support
        - Apache Arrow format for efficiency
        - Built-in versioning support

    Attributes:
        TABLE_NAME: Name of the LanceDB table.
        VECTOR_DIM: Dimension of embedding vectors.
    """

    TABLE_NAME = "code_elements"

    def __init__(
        self,
        persist_dir: Union[str, Path],
        embedding_fn: EmbeddingFunction,
        vector_dim: int,
    ) -> None:
        """Initialize the LanceDB vector store.

        Args:
            persist_dir: Directory for LanceDB persistence.
            embedding_fn: Function to generate embeddings from text.
            vector_dim: Dimension of embedding vectors.
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "LanceDB is required for this vector store. "
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
            table = self._db.open_table(self.TABLE_NAME)
            # Check for dimension mismatch (e.g. embedding model changed)
            try:
                vec_field = table.schema.field("vector")
                existing_dim = vec_field.type.list_size
                if existing_dim != self.vector_dim:
                    logger.warning(
                        f"Vector dimension mismatch: table has {existing_dim}, "
                        f"expected {self.vector_dim}. Recreating table."
                    )
                    self._db.drop_table(self.TABLE_NAME)
                    # Force fresh table creation below
                else:
                    return table
            except Exception as e:
                logger.warning(f"Error checking table schema, recreating: {e}")
                try:
                    self._db.drop_table(self.TABLE_NAME)
                except Exception:
                    pass

        # Define schema for code elements
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("document", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),
                pa.field("file", pa.string()),
                pa.field("type", pa.string()),
                pa.field("name", pa.string()),
                pa.field("language", pa.string()),
                pa.field("line_start", pa.int32()),
                pa.field("line_end", pa.int32()),
            ]
        )

        # Create empty table with schema
        try:
            return self._db.create_table(self.TABLE_NAME, schema=schema)
        except Exception as e:
            # If creation fails, try to drop and recreate
            logger.error(f"Failed to create table: {e}")
            try:
                self._db.drop_table(self.TABLE_NAME)
            except Exception:
                pass
            return self._db.create_table(self.TABLE_NAME, schema=schema)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self._embedding_fn(texts)

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            ids: List of unique IDs.
            documents: List of text documents.
            metadatas: Optional list of metadata dicts.
        """
        if not ids:
            return

        # Truncate documents
        truncated_docs = [self._truncate(doc) for doc in documents]

        # Generate embeddings
        embeddings = self._embed(truncated_docs)

        # Prepare data for insertion
        data = []
        for i, (doc_id, doc, embedding) in enumerate(
            zip(ids, truncated_docs, embeddings)
        ):
            metadata = metadatas[i] if metadatas else {}
            data.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "vector": embedding,
                    "file": metadata.get("file", ""),
                    "type": metadata.get("type", ""),
                    "name": metadata.get("name", ""),
                    "language": metadata.get("language", ""),
                    "line_start": metadata.get("line_start", 0),
                    "line_end": metadata.get("line_end", 0),
                }
            )

        # Add to table
        self._table.add(data)
        logger.debug(f"Added {len(data)} documents to LanceDB")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query for similar documents.

        Args:
            query_text: Text to search for.
            n_results: Number of results to return.
            where: Optional filter dict (e.g., {"file": "path/to/file.py"}).

        Returns:
            List of results with id, document, metadata, similarity.
        """
        # Truncate query
        query_text = self._truncate(query_text)

        # Generate query embedding
        query_embedding = self._embed([query_text])[0]

        # Build search query
        search = self._table.search(query_embedding).limit(n_results)

        # Apply filters if provided
        if where:
            filter_conditions = []
            for key, value in where.items():
                if isinstance(value, str):
                    filter_conditions.append(f"{key} = '{value}'")
                else:
                    filter_conditions.append(f"{key} = {value}")
            if filter_conditions:
                search = search.where(" AND ".join(filter_conditions))

        # Execute search
        try:
            results = search.to_pandas()
        except Exception as e:
            logger.warning(f"LanceDB query failed: {e}")
            return []

        if results.empty:
            return []

        # Convert to standard format
        output = []
        for _, row in results.iterrows():
            # LanceDB returns _distance (L2 distance), convert to similarity
            distance = row.get("_distance", 0)
            # Convert L2 distance to cosine similarity approximation
            similarity = 1 / (1 + distance)

            output.append(
                {
                    "id": row.get("id", ""),
                    "document": row.get("document", ""),
                    "metadata": {
                        "id": row.get("id", ""),
                        "file": row.get("file", ""),
                        "type": row.get("type", ""),
                        "name": row.get("name", ""),
                        "language": row.get("language", ""),
                        "line_start": row.get("line_start", 0),
                        "line_end": row.get("line_end", 0),
                    },
                    "distance": distance,
                    "similarity": similarity,
                }
            )

        return output

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of IDs to delete.
        """
        if not ids:
            return

        # Build delete condition
        id_list = ", ".join([f"'{id}'" for id in ids])
        self._table.delete(f"id IN ({id_list})")
        logger.debug(f"Deleted {len(ids)} documents from LanceDB")

    def delete_by_file(self, file_path: Path) -> None:
        """Delete all documents from a specific file.

        Args:
            file_path: Path to file.
        """
        self._table.delete(f"file = '{str(file_path)}'")
        logger.debug(f"Deleted documents for file: {file_path}")

    def count(self) -> int:
        """Get total count of documents.

        Returns:
            Number of documents in the store.
        """
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    def clear(self) -> None:
        """Clear all documents from the store."""
        # Drop and recreate table
        if self.TABLE_NAME in self._db.table_names():
            self._db.drop_table(self.TABLE_NAME)
        self._table = self._get_or_create_table()
        logger.info("Cleared all documents from LanceDB")

    def get_metrics(self) -> Dict[str, Any]:
        """Get storage metrics.

        Returns:
            Dictionary with storage statistics.
        """
        return {
            "backend": "lancedb",
            "table_name": self.TABLE_NAME,
            "document_count": self.count(),
            "persist_dir": str(self.persist_dir),
            "vector_dim": self.vector_dim,
        }


def create_lance_embedding_fn(embedder) -> EmbeddingFunction:
    """Create an embedding function from a LangChain embedder.

    Args:
        embedder: LangChain Embeddings instance.

    Returns:
        Callable that generates embeddings from text.
    """

    def embed_fn(texts: List[str]) -> List[List[float]]:
        if hasattr(embedder, "embed_batch"):
            return embedder.embed_batch(texts)
        if hasattr(embedder, "embed_documents"):
            return embedder.embed_documents(texts)
        if callable(embedder):
            return embedder(texts)
        raise TypeError(
            "Embedding function must be callable or provide embed_batch/embed_documents"
        )

    return embed_fn
