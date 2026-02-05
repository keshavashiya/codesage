"""Unified storage manager for all backends.

Provides a single entry point for SQLite, LanceDB, and KuzuDB,
ensuring consistent storage across all codesage commands.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from codesage.storage.database import Database
from codesage.storage.kuzu_store import CodeNode, CodeRelationship
from codesage.models.code_element import CodeElement
from codesage.utils.logging import get_logger

if TYPE_CHECKING:
    from codesage.utils.config import Config

logger = get_logger("storage.manager")


class StorageManager:
    """Unified storage manager for all backends.

    Provides a consistent API for storing and querying code elements across:
    - SQLite: Metadata, file hashes, element data
    - LanceDB: Vector embeddings for semantic search
    - KuzuDB: Graph relationships (calls, imports, inheritance)

    Example:
        >>> manager = StorageManager(config, embedding_fn)
        >>> manager.add_elements(elements)
        >>> results = manager.query_with_context("authentication function")
    """

    def __init__(
        self,
        config: "Config",
        embedding_fn=None,
    ) -> None:
        """Initialize storage manager with all backends.

        Args:
            config: CodeSage configuration
            embedding_fn: Embedding function for vector store (required for LanceDB)
        """
        self.config = config
        self._embedding_fn = embedding_fn

        # Initialize SQLite database
        self.db = Database(config.storage.db_path)

        # Initialize vector store (LanceDB)
        self._vector_store = None

        # Initialize graph store (KuzuDB)
        self._graph_store = None
        self._use_graph = config.storage.use_graph

        logger.info(
            f"StorageManager initialized: "
            f"vector=lancedb, graph={self._use_graph}"
        )

    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is not None:
            return self._vector_store

        from codesage.storage.lance_store import LanceVectorStore, create_lance_embedding_fn

        if self._embedding_fn is None:
            raise ValueError("embedding_fn required for LanceDB")

        # Wrap embedding function to support EmbeddingService, LangChain, or callables
        embed_fn = create_lance_embedding_fn(self._embedding_fn)

        self._vector_store = LanceVectorStore(
            persist_dir=self.config.storage.lance_path,
            embedding_fn=embed_fn,
        )
        logger.debug("Initialized LanceDB vector store")

        return self._vector_store

    @property
    def graph_store(self):
        """Lazy load graph store."""
        if not self._use_graph:
            return None

        if self._graph_store is not None:
            return self._graph_store

        try:
            from codesage.storage.kuzu_store import KuzuGraphStore

            self._graph_store = KuzuGraphStore(
                persist_dir=self.config.storage.kuzu_path
            )
            logger.debug("Initialized KuzuDB graph store")
        except ImportError:
            logger.warning("KuzuDB not available, graph features disabled")
            self._use_graph = False
            return None

        return self._graph_store

    def add_elements(self, elements: List[CodeElement]) -> None:
        """Add code elements to all storage backends.

        Args:
            elements: List of code elements to store
        """
        if not elements:
            return

        # Store in SQLite
        self.db.store_elements(elements)

        # Store in vector store
        if self.vector_store:
            self.vector_store.add_elements(elements)

        # Store in graph store
        if self.graph_store:
            self._add_to_graph(elements)

        logger.debug(f"Added {len(elements)} elements to storage")

    def _add_to_graph(self, elements: List[CodeElement]) -> None:
        """Add code elements as nodes in the graph store.

        Args:
            elements: List of code elements
        """
        if not self.graph_store:
            return

        from codesage.storage.kuzu_store import CodeNode

        nodes = []
        for element in elements:
            node = CodeNode(
                id=element.id,
                name=element.name or "",
                node_type=element.type,
                file=str(element.file),
                line_start=element.line_start,
                line_end=element.line_end,
                language=element.language,
            )
            nodes.append(node)

        self.graph_store.add_nodes(nodes)

    def add_relationships(self, relationships: List[CodeRelationship]) -> None:
        """Add relationships to the graph store.

        Args:
            relationships: List of code relationships
        """
        if self.graph_store and relationships:
            self.graph_store.add_relationships(relationships)
            logger.debug(f"Added {len(relationships)} relationships to graph")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query vector store for similar code.

        Args:
            query_text: Search query
            n_results: Number of results
            where: Optional metadata filter

        Returns:
            List of matching results
        """
        if not self.vector_store:
            return []

        return self.vector_store.query(
            query_text=query_text,
            n_results=n_results,
            where=where,
        )

    def query_with_context(
        self,
        query_text: str,
        n_results: int = 5,
        include_graph: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query with full context including graph relationships.

        Args:
            query_text: Search query
            n_results: Number of results
            include_graph: Whether to include graph context

        Returns:
            List of results enriched with graph context
        """
        results = self.query(query_text, n_results)

        if include_graph and self.graph_store:
            for result in results:
                metadata = result.get("metadata", {})
                node_id = metadata.get("id", "")

                if node_id:
                    # Add caller/callee information
                    result["callers"] = self.graph_store.get_callers(node_id)
                    result["callees"] = self.graph_store.get_callees(node_id)

                    # For classes, add inheritance info
                    if metadata.get("type") == "class":
                        result["superclasses"] = self.graph_store.get_superclasses(node_id)
                        result["subclasses"] = self.graph_store.get_subclasses(node_id)

                    # Additional graph enrichment if enabled
                    if getattr(self.config.features, "graph_enriched_search", False):
                        result["dependencies"] = self.graph_store.get_dependencies(node_id)
                        result["dependents"] = self.graph_store.get_dependents(node_id)
                        result["impact_score"] = self._calculate_impact_score(node_id)

        return results

    def _calculate_impact_score(self, node_id: str) -> float:
        """Calculate a simple impact score based on caller counts."""
        if not self.graph_store:
            return 0.0

        direct_callers = self.graph_store.get_callers(node_id)
        transitive = self.graph_store.get_transitive_callers(node_id, max_depth=2)
        score = (len(direct_callers) + len(transitive)) / 50.0
        return min(1.0, score)

    def delete_by_file(self, file_path: Path) -> None:
        """Delete all data for a specific file.

        Args:
            file_path: Path to file
        """
        self.db.delete_elements_for_file(file_path)

        if self.vector_store:
            self.vector_store.delete_by_file(file_path)

        if self.graph_store:
            self.graph_store.delete_by_file(file_path)

    def clear(self) -> None:
        """Clear all stored data."""
        self.db.clear()

        if self.vector_store:
            self.vector_store.clear()

        if self.graph_store:
            self.graph_store.clear()

        logger.info("Cleared all storage backends")

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from all storage backends.

        Returns:
            Dictionary with storage statistics
        """
        metrics = {
            "sqlite": {
                "db_path": str(self.config.storage.db_path),
            },
        }

        # Get SQLite stats
        try:
            stats = self.db.get_stats()
            metrics["sqlite"]["total_elements"] = stats.get("total_elements", 0)
            metrics["sqlite"]["total_files"] = stats.get("total_files", 0)
        except Exception:
            pass

        if self.vector_store:
            metrics["vector"] = self.vector_store.get_metrics()

        if self.graph_store:
            metrics["graph"] = self.graph_store.get_metrics()

        return metrics

    def count(self) -> Dict[str, int]:
        """Get document/node counts from all backends.

        Returns:
            Dictionary with counts per backend
        """
        counts = {}

        if self.vector_store:
            counts["vectors"] = self.vector_store.count()

        if self.graph_store:
            counts["nodes"] = self.graph_store.count_nodes()
            counts["relationships"] = sum(
                self.graph_store.count_relationships().values()
            )

        return counts
