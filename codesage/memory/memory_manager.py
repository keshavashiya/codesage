"""Unified memory manager for developer memory system.

Provides a single interface to access all memory storage backends:
- SQLite (PreferenceStore): Preferences, metrics, interactions
- LanceDB (PatternStore): Pattern embeddings for semantic search
- KuzuDB (MemoryGraph): Pattern relationships and graph queries

Follows the StorageManager pattern from codesage.storage.manager.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from codesage.utils.logging import get_logger

from .memory_graph import MemoryGraph
from .models import (
    CodeStructure,
    DeveloperPreference,
    InteractionRecord,
    LearnedPattern,
    PatternCategory,
    ProjectInfo,
    StructureType,
)
from .pattern_store import PatternStore
from .preference_store import PreferenceStore

logger = get_logger("memory.manager")

# Type alias for embedding function
EmbeddingFunction = Callable[[List[str]], List[List[float]]]


class MemoryManager:
    """Unified access to developer memory storage.

    Provides a consistent API for storing and querying developer patterns,
    preferences, and project information across multiple backends:
        - SQLite: Preferences, pattern metadata, interactions
        - LanceDB: Pattern embeddings for semantic search
        - KuzuDB: Pattern relationships and graph queries

    Attributes:
        GLOBAL_DIR: Default path for global developer memory.
    """

    GLOBAL_DIR = Path.home() / ".codesage" / "developer"

    def __init__(
        self,
        global_dir: Optional[Path] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        vector_dim: int = 0,
    ) -> None:
        """Initialize the memory manager.

        Args:
            global_dir: Directory for global memory storage.
                       Defaults to ~/.codesage/developer/
            embedding_fn: Optional function for generating embeddings.
            vector_dim: Embedding vector dimension (0 = use store default).
        """
        self.global_dir = Path(global_dir) if global_dir else self.GLOBAL_DIR
        self.global_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_fn = embedding_fn
        self._vector_dim = vector_dim

        # Lazy-loaded backends
        self._preference_store: Optional[PreferenceStore] = None
        self._pattern_store: Optional[PatternStore] = None
        self._memory_graph: Optional[MemoryGraph] = None

        logger.debug(f"MemoryManager initialized at {self.global_dir}")

    # ==================== Lazy Properties ====================

    @property
    def preference_store(self) -> PreferenceStore:
        """Get the preference store (SQLite).

        Lazy loads the store on first access.

        Returns:
            PreferenceStore instance.
        """
        if self._preference_store is None:
            db_path = self.global_dir / "profile.db"
            self._preference_store = PreferenceStore(db_path)
            logger.debug("Initialized PreferenceStore")
        return self._preference_store

    @property
    def pattern_store(self) -> PatternStore:
        """Get the pattern store (LanceDB).

        Lazy loads the store on first access.

        Returns:
            PatternStore instance.
        """
        if self._pattern_store is None:
            lance_path = self.global_dir / "patterns.lance"
            kwargs = {
                "persist_dir": lance_path,
                "embedding_fn": self._embedding_fn,
            }
            if self._vector_dim > 0:
                kwargs["vector_dim"] = self._vector_dim
            self._pattern_store = PatternStore(**kwargs)
            logger.debug("Initialized PatternStore")
        return self._pattern_store

    @property
    def memory_graph(self) -> MemoryGraph:
        """Get the memory graph (KuzuDB).

        Lazy loads the graph on first access.

        Returns:
            MemoryGraph instance.
        """
        if self._memory_graph is None:
            kuzu_path = self.global_dir / "memory.kuzu"
            self._memory_graph = MemoryGraph(kuzu_path)
            logger.debug("Initialized MemoryGraph")
        return self._memory_graph

    def set_embedding_fn(self, embedding_fn: EmbeddingFunction) -> None:
        """Set the embedding function for pattern search.

        Args:
            embedding_fn: Function to generate embeddings.
        """
        self._embedding_fn = embedding_fn
        if self._pattern_store is not None:
            self._pattern_store.set_embedding_fn(embedding_fn)

    # ==================== Unified Pattern Operations ====================

    def add_pattern(
        self,
        pattern: LearnedPattern,
        project_name: Optional[str] = None,
    ) -> None:
        """Add a pattern to all storage backends.

        Stores the pattern in:
            1. SQLite: Metadata and metrics
            2. LanceDB: Embedding for semantic search
            3. KuzuDB: Graph node for relationships

        Args:
            pattern: Pattern to add.
            project_name: Optional project name to link to.
        """
        # Store in SQLite (metadata)
        self.preference_store.add_pattern(pattern)

        # Store in LanceDB (embedding)
        try:
            self.pattern_store.add_pattern(pattern)
        except Exception as e:
            logger.warning(f"Failed to add pattern to LanceDB: {e}")

        # Store in KuzuDB (graph)
        try:
            self.memory_graph.add_pattern_node(pattern)
        except Exception as e:
            logger.debug(f"Failed to add pattern to KuzuDB: {e}")

        # Link to project if provided
        if project_name:
            self.preference_store.link_pattern_to_project(pattern.id, project_name)

            # Get project ID and link in graph
            project = self.preference_store.get_project(project_name)
            if project:
                try:
                    self.memory_graph.link_pattern_to_project(pattern.id, project.id)
                except Exception as e:
                    logger.debug(f"Failed to link pattern to project in graph: {e}")

        logger.debug(f"Added pattern {pattern.name} to all stores")

    def get_pattern(self, pattern_id: str) -> Optional[LearnedPattern]:
        """Get a pattern by ID with enriched data.

        Retrieves pattern from SQLite and enriches with:
            - Co-occurring patterns from graph
            - Source projects from graph

        Args:
            pattern_id: Pattern ID.

        Returns:
            LearnedPattern with enriched data, or None if not found.
        """
        pattern = self.preference_store.get_pattern(pattern_id)
        if pattern is None:
            return None

        # Enrich with graph data
        try:
            cooccurring = self.memory_graph.get_cooccurring_patterns(pattern_id)
            pattern.co_occurring_patterns = [p["id"] for p in cooccurring]

            projects = self.memory_graph.get_pattern_projects(pattern_id)
            pattern.source_projects = [p["name"] for p in projects]
        except Exception as e:
            logger.debug(f"Failed to enrich pattern with graph data: {e}")

        return pattern

    def record_pattern_cooccurrence(
        self,
        pattern1_id: str,
        pattern2_id: str,
        correlation: float = 0.5,
    ) -> None:
        """Record that two patterns appear together.

        Args:
            pattern1_id: First pattern ID.
            pattern2_id: Second pattern ID.
            correlation: Correlation strength (0.0-1.0).
        """
        try:
            self.memory_graph.add_cooccurrence(pattern1_id, pattern2_id, correlation)
            # Add reverse relationship too
            self.memory_graph.add_cooccurrence(pattern2_id, pattern1_id, correlation)
            logger.debug(f"Recorded co-occurrence: {pattern1_id} <-> {pattern2_id}")
        except Exception as e:
            logger.debug(f"Failed to record co-occurrence: {e}")

    def find_similar_patterns(
        self,
        query: str,
        limit: int = 5,
        category: Optional[PatternCategory] = None,
    ) -> List[Dict[str, Any]]:
        """Find patterns similar to a query.

        Uses semantic search over pattern embeddings.

        Args:
            query: Search query.
            limit: Maximum results.
            category: Optional category filter.

        Returns:
            List of matching patterns with similarity scores.
        """
        results = self.pattern_store.search(
            query=query,
            n_results=limit,
            category=category,
        )

        # Enrich results with graph relationships
        for result in results:
            try:
                cooccurring = self.memory_graph.get_cooccurring_patterns(
                    result["id"], min_correlation=0.5
                )
                result["co_occurring_patterns"] = [
                    {"id": p["id"], "name": p["name"], "correlation": p["correlation"]}
                    for p in cooccurring[:3]
                ]
            except Exception:
                result["co_occurring_patterns"] = []

        return results

    def get_pattern_context(self, pattern_id: str) -> Dict[str, Any]:
        """Get full context for a pattern.

        Combines data from all stores:
            - Pattern details from SQLite
            - Similar patterns from LanceDB
            - Co-occurring patterns from KuzuDB
            - Source projects from KuzuDB

        Args:
            pattern_id: Pattern ID.

        Returns:
            Dictionary with full pattern context.
        """
        context = {
            "pattern": None,
            "similar_patterns": [],
            "co_occurring_patterns": [],
            "source_projects": [],
        }

        # Get pattern from SQLite
        pattern = self.preference_store.get_pattern(pattern_id)
        if pattern is None:
            return context

        context["pattern"] = pattern.to_dict()

        # Get similar patterns from LanceDB
        try:
            context["similar_patterns"] = self.pattern_store.find_similar(
                pattern_id, n_results=5
            )
        except Exception as e:
            logger.debug(f"Failed to get similar patterns: {e}")

        # Get co-occurring patterns from KuzuDB
        try:
            context["co_occurring_patterns"] = self.memory_graph.get_cooccurring_patterns(
                pattern_id
            )
        except Exception as e:
            logger.debug(f"Failed to get co-occurring patterns: {e}")

        # Get source projects from KuzuDB
        try:
            context["source_projects"] = self.memory_graph.get_pattern_projects(pattern_id)
        except Exception as e:
            logger.debug(f"Failed to get source projects: {e}")

        return context

    # ==================== Unified Project Operations ====================

    def add_project(self, project: ProjectInfo) -> None:
        """Add a project to all storage backends.

        Args:
            project: Project to add.
        """
        # Store in SQLite
        self.preference_store.add_project(project)

        # Store in KuzuDB graph
        try:
            self.memory_graph.add_project_node(project)
        except Exception as e:
            logger.debug(f"Failed to add project to graph: {e}")

        logger.debug(f"Added project {project.name}")

    def get_project(self, name: str) -> Optional[ProjectInfo]:
        """Get a project by name.

        Args:
            name: Project name.

        Returns:
            ProjectInfo or None.
        """
        return self.preference_store.get_project(name)

    def find_similar_projects(
        self,
        project_name: str,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Find projects similar to a given project.

        Args:
            project_name: Reference project name.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of similar projects with similarity scores.
        """
        try:
            return self.memory_graph.find_similar_projects(project_name, min_similarity)
        except Exception as e:
            logger.debug(f"Failed to find similar projects: {e}")
            return []

    # ==================== Unified Structure Operations ====================

    def add_structure(self, structure: CodeStructure) -> None:
        """Add a code structure preference.

        Args:
            structure: Structure to add.
        """
        try:
            self.memory_graph.add_structure_node(structure)
            logger.debug(f"Added structure {structure.name}")
        except Exception as e:
            logger.debug(f"Failed to add structure: {e}")

    def get_preferred_structures(
        self,
        structure_type: Optional[StructureType] = None,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Get preferred code structures.

        Args:
            structure_type: Optional type filter.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of preferred structures.
        """
        try:
            return self.memory_graph.get_preferred_structures(
                structure_type, min_confidence
            )
        except Exception as e:
            logger.debug(f"Failed to get preferred structures: {e}")
            return []

    # ==================== Preference Operations ====================

    def set_preference(
        self,
        key: str,
        value: Any,
        category: str = "general",
        description: str = "",
    ) -> None:
        """Set a developer preference.

        Args:
            key: Preference key.
            value: Preference value.
            category: Preference category.
            description: Optional description.
        """
        pref = DeveloperPreference(
            key=key,
            value=value,
            category=category,
            description=description,
        )
        self.preference_store.set_preference(pref)

    def get_preference(self, key: str) -> Optional[Any]:
        """Get a preference value.

        Args:
            key: Preference key.

        Returns:
            Preference value or None.
        """
        pref = self.preference_store.get_preference(key)
        return pref.value if pref else None

    def get_all_preferences(
        self,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get all preferences as a dictionary.

        Args:
            category: Optional category filter.

        Returns:
            Dictionary of key-value pairs.
        """
        prefs = self.preference_store.get_all_preferences(category)
        return {p.key: p.value for p in prefs}

    # ==================== Interaction Operations ====================

    def record_interaction(
        self,
        interaction_type: str,
        project_name: str,
        query: str = "",
        response: str = "",
        accepted: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a developer interaction.

        Args:
            interaction_type: Type of interaction.
            project_name: Project name.
            query: Query or input.
            response: Response or output.
            accepted: Whether accepted (for suggestions).
            metadata: Additional metadata.
        """
        interaction = InteractionRecord.create(
            interaction_type=interaction_type,
            project_name=project_name,
            query=query,
            response=response,
            accepted=accepted,
            metadata=metadata or {},
        )
        self.preference_store.add_interaction(interaction)

    def get_interaction_stats(self) -> Dict[str, Any]:
        """Get interaction statistics.

        Returns:
            Dictionary with stats.
        """
        return self.preference_store.get_interaction_stats()

    # ==================== Graph Analysis ====================

    def get_developer_style_graph(self) -> Dict[str, Any]:
        """Export developer's full style as a graph.

        Returns:
            Graph structure for visualization.
        """
        try:
            return self.memory_graph.export_style_graph()
        except Exception as e:
            logger.debug(f"Failed to export style graph: {e}")
            return {"patterns": [], "projects": [], "structures": [], "relationships": []}

    def get_cross_project_patterns(self, min_projects: int = 2) -> List[Dict[str, Any]]:
        """Get patterns that appear across multiple projects.

        Args:
            min_projects: Minimum number of projects.

        Returns:
            List of cross-project patterns.
        """
        try:
            return self.memory_graph.get_cross_project_patterns(min_projects)
        except Exception as e:
            logger.debug(f"Failed to get cross-project patterns: {e}")
            return []

    # ==================== Metrics ====================

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics from all stores.

        Returns:
            Dictionary with metrics from all backends.
        """
        metrics = {
            "global_dir": str(self.global_dir),
            "has_embedding_fn": self._embedding_fn is not None,
        }

        # Get metrics from each store
        try:
            metrics["preference_store"] = self.preference_store.get_metrics()
        except Exception as e:
            metrics["preference_store"] = {"error": str(e)}

        try:
            metrics["pattern_store"] = self.pattern_store.get_metrics()
        except Exception as e:
            metrics["pattern_store"] = {"error": str(e)}

        try:
            metrics["memory_graph"] = self.memory_graph.get_metrics()
        except Exception as e:
            metrics["memory_graph"] = {"error": str(e)}

        return metrics

    def clear(self) -> None:
        """Clear all data from all stores."""
        try:
            self.preference_store.clear()
        except Exception as e:
            logger.debug(f"Failed to clear preference store: {e}")

        try:
            self.pattern_store.clear()
        except Exception as e:
            logger.debug(f"Failed to clear pattern store: {e}")

        try:
            self.memory_graph.clear()
        except Exception as e:
            logger.debug(f"Failed to clear memory graph: {e}")

        logger.info("Cleared all data from MemoryManager")

    def close(self) -> None:
        """Close all connections."""
        if self._preference_store is not None:
            self._preference_store.close()
            self._preference_store = None

        # LanceDB and KuzuDB don't have explicit close methods
        self._pattern_store = None
        self._memory_graph = None

        logger.debug("MemoryManager connections closed")
