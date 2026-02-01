"""KuzuDB graph store for pattern relationships.

Stores and queries relationships between patterns, projects, and code structures.
Follows the KuzuGraphStore pattern from codesage.storage.kuzu_store.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from codesage.utils.logging import get_logger

from .models import (
    CodeStructure,
    LearnedPattern,
    PatternCategory,
    ProjectInfo,
    RelationshipType,
    StructureType,
)

logger = get_logger("memory.memory_graph")


class MemoryGraph:
    """KuzuDB graph store for pattern relationships.

    Stores and queries relationships between:
        - Patterns (CO_OCCURS, SIMILAR_TO, EVOLVES_TO)
        - Patterns and Projects (LEARNED_FROM)
        - Patterns and Structures (PREFERS)
        - Projects (SIMILAR_TO)

    This enables rich queries like:
        - Find patterns that co-occur with a given pattern
        - Find projects with similar coding styles
        - Track pattern evolution over time
    """

    def __init__(self, persist_dir: Union[str, Path]) -> None:
        """Initialize the memory graph store.

        Args:
            persist_dir: Directory for KuzuDB persistence.
        """
        try:
            import kuzu
        except ImportError:
            raise ImportError(
                "KuzuDB is required for the memory graph. "
                "Install with: pipx inject pycodesage kuzu (or pip install kuzu)"
            )

        self.persist_dir = Path(persist_dir)
        self.persist_dir.parent.mkdir(parents=True, exist_ok=True)

        self._db = kuzu.Database(str(self.persist_dir))
        self._conn = kuzu.Connection(self._db)

        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize graph schema with node and relationship tables."""
        from .schemas import KUZU_ALL_SCHEMAS

        try:
            for schema_sql in KUZU_ALL_SCHEMAS:
                self._conn.execute(schema_sql)
            logger.debug("MemoryGraph schema initialized")
        except Exception as e:
            logger.debug(f"Schema init note: {e}")

    # ==================== Pattern Node Operations ====================

    def add_pattern_node(self, pattern: LearnedPattern) -> None:
        """Add or update a pattern node.

        Args:
            pattern: Pattern to add.
        """
        try:
            first_seen = pattern.first_seen or datetime.now()
            last_seen = pattern.last_seen or datetime.now()

            self._conn.execute(
                """
                MERGE (p:PatternNode {id: $id})
                SET p.name = $name,
                    p.category = $category,
                    p.description = $description,
                    p.pattern_text = $pattern_text,
                    p.occurrence_count = $occurrence_count,
                    p.confidence = $confidence,
                    p.first_seen = $first_seen,
                    p.last_seen = $last_seen
                """,
                {
                    "id": pattern.id,
                    "name": pattern.name,
                    "category": pattern.category.value,
                    "description": pattern.description,
                    "pattern_text": pattern.pattern_text,
                    "occurrence_count": pattern.occurrence_count,
                    "confidence": pattern.confidence_score,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to add pattern node {pattern.id}: {e}")

    def get_pattern_node(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a pattern node by ID.

        Args:
            pattern_id: Pattern ID.

        Returns:
            Pattern data or None if not found.
        """
        result = self._conn.execute(
            """
            MATCH (p:PatternNode {id: $id})
            RETURN p.id AS id, p.name AS name, p.category AS category,
                   p.description AS description, p.pattern_text AS pattern_text,
                   p.occurrence_count AS occurrence_count, p.confidence AS confidence
            """,
            {"id": pattern_id},
        )
        rows = self._result_to_list(result)
        return rows[0] if rows else None

    def delete_pattern_node(self, pattern_id: str) -> None:
        """Delete a pattern node and its relationships.

        Args:
            pattern_id: Pattern ID to delete.
        """
        try:
            # Delete relationships first
            for rel_type in ["CO_OCCURS", "LEARNED_FROM", "PREFERS", "EVOLVES_TO"]:
                self._conn.execute(
                    f"MATCH (p:PatternNode {{id: $id}})-[r:{rel_type}]->() DELETE r",
                    {"id": pattern_id},
                )
                self._conn.execute(
                    f"MATCH ()-[r:{rel_type}]->(p:PatternNode {{id: $id}}) DELETE r",
                    {"id": pattern_id},
                )

            # Delete node
            self._conn.execute(
                "MATCH (p:PatternNode {id: $id}) DELETE p",
                {"id": pattern_id},
            )
        except Exception as e:
            logger.warning(f"Failed to delete pattern node {pattern_id}: {e}")

    # ==================== Project Node Operations ====================

    def add_project_node(self, project: ProjectInfo) -> None:
        """Add or update a project node.

        Args:
            project: Project info to add.
        """
        try:
            first_indexed = project.first_indexed or datetime.now()
            last_indexed = project.last_indexed or datetime.now()

            self._conn.execute(
                """
                MERGE (p:ProjectNode {id: $id})
                SET p.name = $name,
                    p.path = $path,
                    p.language = $language,
                    p.total_files = $total_files,
                    p.total_elements = $total_elements,
                    p.first_indexed = $first_indexed,
                    p.last_indexed = $last_indexed
                """,
                {
                    "id": project.id,
                    "name": project.name,
                    "path": str(project.path),
                    "language": project.language,
                    "total_files": project.total_files,
                    "total_elements": project.total_elements,
                    "first_indexed": first_indexed,
                    "last_indexed": last_indexed,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to add project node {project.name}: {e}")

    def get_project_node(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get a project node by name.

        Args:
            project_name: Project name.

        Returns:
            Project data or None if not found.
        """
        result = self._conn.execute(
            """
            MATCH (p:ProjectNode {name: $name})
            RETURN p.id AS id, p.name AS name, p.path AS path,
                   p.language AS language, p.total_files AS total_files,
                   p.total_elements AS total_elements
            """,
            {"name": project_name},
        )
        rows = self._result_to_list(result)
        return rows[0] if rows else None

    # ==================== Code Structure Operations ====================

    def add_structure_node(self, structure: CodeStructure) -> None:
        """Add or update a code structure node.

        Args:
            structure: Code structure to add.
        """
        try:
            self._conn.execute(
                """
                MERGE (s:CodeStructureNode {id: $id})
                SET s.structure_type = $structure_type,
                    s.name = $name,
                    s.description = $description,
                    s.example_code = $example_code,
                    s.occurrence_count = $occurrence_count,
                    s.confidence = $confidence
                """,
                {
                    "id": structure.id,
                    "structure_type": structure.structure_type.value,
                    "name": structure.name,
                    "description": structure.description,
                    "example_code": structure.example_code,
                    "occurrence_count": structure.occurrence_count,
                    "confidence": structure.confidence,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to add structure node {structure.id}: {e}")

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
            List of structure data.
        """
        if structure_type:
            result = self._conn.execute(
                """
                MATCH (s:CodeStructureNode)
                WHERE s.structure_type = $type AND s.confidence >= $min_confidence
                RETURN s.id AS id, s.name AS name, s.structure_type AS type,
                       s.description AS description, s.occurrence_count AS occurrence_count,
                       s.confidence AS confidence
                ORDER BY s.confidence DESC
                """,
                {"type": structure_type.value, "min_confidence": min_confidence},
            )
        else:
            result = self._conn.execute(
                """
                MATCH (s:CodeStructureNode)
                WHERE s.confidence >= $min_confidence
                RETURN s.id AS id, s.name AS name, s.structure_type AS type,
                       s.description AS description, s.occurrence_count AS occurrence_count,
                       s.confidence AS confidence
                ORDER BY s.confidence DESC
                """,
                {"min_confidence": min_confidence},
            )

        return self._result_to_list(result)

    # ==================== Relationship Operations ====================

    def add_cooccurrence(
        self,
        pattern1_id: str,
        pattern2_id: str,
        correlation: float = 0.5,
    ) -> None:
        """Record that two patterns co-occur.

        Args:
            pattern1_id: First pattern ID.
            pattern2_id: Second pattern ID.
            correlation: Correlation strength (0.0-1.0).
        """
        try:
            # Check if relationship exists
            result = self._conn.execute(
                """
                MATCH (p1:PatternNode {id: $id1})-[r:CO_OCCURS]->(p2:PatternNode {id: $id2})
                RETURN r.count AS count
                """,
                {"id1": pattern1_id, "id2": pattern2_id},
            )
            rows = self._result_to_list(result)

            if rows:
                # Update existing
                self._conn.execute(
                    """
                    MATCH (p1:PatternNode {id: $id1})-[r:CO_OCCURS]->(p2:PatternNode {id: $id2})
                    SET r.count = r.count + 1, r.correlation = $correlation
                    """,
                    {"id1": pattern1_id, "id2": pattern2_id, "correlation": correlation},
                )
            else:
                # Create new
                self._conn.execute(
                    """
                    MATCH (p1:PatternNode {id: $id1}), (p2:PatternNode {id: $id2})
                    CREATE (p1)-[:CO_OCCURS {count: 1, correlation: $correlation}]->(p2)
                    """,
                    {"id1": pattern1_id, "id2": pattern2_id, "correlation": correlation},
                )
        except Exception as e:
            logger.warning(f"Failed to add co-occurrence {pattern1_id}->{pattern2_id}: {e}")

    def get_cooccurring_patterns(
        self,
        pattern_id: str,
        min_correlation: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Get patterns that co-occur with a given pattern.

        Args:
            pattern_id: Pattern ID.
            min_correlation: Minimum correlation threshold.

        Returns:
            List of co-occurring patterns with correlation.
        """
        result = self._conn.execute(
            """
            MATCH (p1:PatternNode {id: $id})-[r:CO_OCCURS]->(p2:PatternNode)
            WHERE r.correlation >= $min_correlation
            RETURN p2.id AS id, p2.name AS name, p2.category AS category,
                   r.count AS count, r.correlation AS correlation
            ORDER BY r.correlation DESC
            """,
            {"id": pattern_id, "min_correlation": min_correlation},
        )
        return self._result_to_list(result)

    def link_pattern_to_project(
        self,
        pattern_id: str,
        project_id: str,
        occurrences: int = 1,
    ) -> None:
        """Create or update LEARNED_FROM relationship.

        Args:
            pattern_id: Pattern ID.
            project_id: Project ID.
            occurrences: Number of occurrences.
        """
        try:
            # Check if relationship exists
            result = self._conn.execute(
                """
                MATCH (p:PatternNode {id: $pattern_id})-[r:LEARNED_FROM]->(proj:ProjectNode {id: $project_id})
                RETURN r.occurrences AS occurrences
                """,
                {"pattern_id": pattern_id, "project_id": project_id},
            )
            rows = self._result_to_list(result)

            if rows:
                # Update existing
                self._conn.execute(
                    """
                    MATCH (p:PatternNode {id: $pattern_id})-[r:LEARNED_FROM]->(proj:ProjectNode {id: $project_id})
                    SET r.occurrences = r.occurrences + $occurrences
                    """,
                    {"pattern_id": pattern_id, "project_id": project_id, "occurrences": occurrences},
                )
            else:
                # Create new
                self._conn.execute(
                    """
                    MATCH (p:PatternNode {id: $pattern_id}), (proj:ProjectNode {id: $project_id})
                    CREATE (p)-[:LEARNED_FROM {first_seen: $now, occurrences: $occurrences}]->(proj)
                    """,
                    {
                        "pattern_id": pattern_id,
                        "project_id": project_id,
                        "occurrences": occurrences,
                        "now": datetime.now(),
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to link pattern {pattern_id} to project {project_id}: {e}")

    def get_pattern_projects(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Get projects where a pattern was learned.

        Args:
            pattern_id: Pattern ID.

        Returns:
            List of project info with occurrence counts.
        """
        result = self._conn.execute(
            """
            MATCH (p:PatternNode {id: $id})-[r:LEARNED_FROM]->(proj:ProjectNode)
            RETURN proj.id AS id, proj.name AS name, proj.path AS path,
                   r.occurrences AS occurrences, r.first_seen AS first_seen
            ORDER BY r.occurrences DESC
            """,
            {"id": pattern_id},
        )
        return self._result_to_list(result)

    def add_project_similarity(
        self,
        project1_id: str,
        project2_id: str,
        similarity: float,
    ) -> None:
        """Record similarity between projects.

        Args:
            project1_id: First project ID.
            project2_id: Second project ID.
            similarity: Similarity score (0.0-1.0).
        """
        try:
            self._conn.execute(
                """
                MATCH (p1:ProjectNode {id: $id1}), (p2:ProjectNode {id: $id2})
                MERGE (p1)-[r:SIMILAR_TO]->(p2)
                SET r.similarity = $similarity
                """,
                {"id1": project1_id, "id2": project2_id, "similarity": similarity},
            )
        except Exception as e:
            logger.warning(f"Failed to add project similarity: {e}")

    def find_similar_projects(
        self,
        project_name: str,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Find projects similar to a given project.

        Args:
            project_name: Project name.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of similar projects with similarity scores.
        """
        result = self._conn.execute(
            """
            MATCH (p1:ProjectNode {name: $name})-[r:SIMILAR_TO]->(p2:ProjectNode)
            WHERE r.similarity >= $min_similarity
            RETURN p2.id AS id, p2.name AS name, p2.path AS path,
                   r.similarity AS similarity
            ORDER BY r.similarity DESC
            """,
            {"name": project_name, "min_similarity": min_similarity},
        )
        return self._result_to_list(result)

    def add_pattern_evolution(
        self,
        old_pattern_id: str,
        new_pattern_id: str,
        reason: str = "",
    ) -> None:
        """Record pattern evolution.

        Args:
            old_pattern_id: Original pattern ID.
            new_pattern_id: Evolved pattern ID.
            reason: Reason for evolution.
        """
        try:
            self._conn.execute(
                """
                MATCH (p1:PatternNode {id: $old_id}), (p2:PatternNode {id: $new_id})
                CREATE (p1)-[:EVOLVES_TO {date: $now, reason: $reason}]->(p2)
                """,
                {
                    "old_id": old_pattern_id,
                    "new_id": new_pattern_id,
                    "now": datetime.now(),
                    "reason": reason,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to add pattern evolution: {e}")

    def add_structure_preference(
        self,
        pattern_id: str,
        structure_id: str,
        confidence: float = 0.5,
    ) -> None:
        """Record that a pattern prefers a code structure.

        Args:
            pattern_id: Pattern ID.
            structure_id: Structure ID.
            confidence: Preference confidence.
        """
        try:
            # Check if relationship exists
            result = self._conn.execute(
                """
                MATCH (p:PatternNode {id: $pattern_id})-[r:PREFERS]->(s:CodeStructureNode {id: $structure_id})
                RETURN r.usage_count AS usage_count
                """,
                {"pattern_id": pattern_id, "structure_id": structure_id},
            )
            rows = self._result_to_list(result)

            if rows:
                self._conn.execute(
                    """
                    MATCH (p:PatternNode {id: $pattern_id})-[r:PREFERS]->(s:CodeStructureNode {id: $structure_id})
                    SET r.usage_count = r.usage_count + 1, r.confidence = $confidence
                    """,
                    {"pattern_id": pattern_id, "structure_id": structure_id, "confidence": confidence},
                )
            else:
                self._conn.execute(
                    """
                    MATCH (p:PatternNode {id: $pattern_id}), (s:CodeStructureNode {id: $structure_id})
                    CREATE (p)-[:PREFERS {confidence: $confidence, usage_count: 1}]->(s)
                    """,
                    {"pattern_id": pattern_id, "structure_id": structure_id, "confidence": confidence},
                )
        except Exception as e:
            logger.warning(f"Failed to add structure preference: {e}")

    # ==================== Graph Analysis ====================

    def get_pattern_network(
        self,
        pattern_id: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """Get network of related patterns.

        Args:
            pattern_id: Root pattern ID.
            depth: Maximum traversal depth.

        Returns:
            Network structure with nodes and edges.
        """
        # Get the root pattern
        root = self.get_pattern_node(pattern_id)
        if not root:
            return {"nodes": [], "edges": []}

        nodes = [root]
        edges = []
        visited = {pattern_id}

        # BFS to get connected patterns
        current_level = [pattern_id]
        for _ in range(depth):
            next_level = []
            for pid in current_level:
                cooccurring = self.get_cooccurring_patterns(pid, min_correlation=0.3)
                for p in cooccurring:
                    if p["id"] not in visited:
                        visited.add(p["id"])
                        next_level.append(p["id"])
                        nodes.append(p)
                    edges.append({
                        "source": pid,
                        "target": p["id"],
                        "type": "CO_OCCURS",
                        "weight": p.get("correlation", 0.5),
                    })
            current_level = next_level

        return {"nodes": nodes, "edges": edges}

    def get_cross_project_patterns(self, min_projects: int = 2) -> List[Dict[str, Any]]:
        """Get patterns that appear across multiple projects.

        Args:
            min_projects: Minimum number of projects.

        Returns:
            List of patterns with project counts.
        """
        result = self._conn.execute(
            """
            MATCH (p:PatternNode)-[r:LEARNED_FROM]->(proj:ProjectNode)
            WITH p, COUNT(proj) AS project_count, COLLECT(proj.name) AS projects
            WHERE project_count >= $min_projects
            RETURN p.id AS id, p.name AS name, p.category AS category,
                   p.confidence AS confidence, project_count, projects
            ORDER BY project_count DESC
            """,
            {"min_projects": min_projects},
        )
        return self._result_to_list(result)

    def export_style_graph(self) -> Dict[str, Any]:
        """Export full style graph for visualization.

        Returns:
            Graph structure with all nodes and relationships.
        """
        graph = {
            "patterns": [],
            "projects": [],
            "structures": [],
            "relationships": [],
        }

        # Get all patterns
        result = self._conn.execute("""
            MATCH (p:PatternNode)
            RETURN p.id AS id, p.name AS name, p.category AS category,
                   p.occurrence_count AS occurrence_count, p.confidence AS confidence
        """)
        graph["patterns"] = self._result_to_list(result)

        # Get all projects
        result = self._conn.execute("""
            MATCH (p:ProjectNode)
            RETURN p.id AS id, p.name AS name, p.path AS path, p.language AS language
        """)
        graph["projects"] = self._result_to_list(result)

        # Get all structures
        result = self._conn.execute("""
            MATCH (s:CodeStructureNode)
            RETURN s.id AS id, s.name AS name, s.structure_type AS type, s.confidence AS confidence
        """)
        graph["structures"] = self._result_to_list(result)

        # Get CO_OCCURS relationships
        result = self._conn.execute("""
            MATCH (p1:PatternNode)-[r:CO_OCCURS]->(p2:PatternNode)
            RETURN p1.id AS source, p2.id AS target, 'CO_OCCURS' AS type, r.correlation AS weight
        """)
        graph["relationships"].extend(self._result_to_list(result))

        # Get LEARNED_FROM relationships
        result = self._conn.execute("""
            MATCH (p:PatternNode)-[r:LEARNED_FROM]->(proj:ProjectNode)
            RETURN p.id AS source, proj.id AS target, 'LEARNED_FROM' AS type, r.occurrences AS weight
        """)
        graph["relationships"].extend(self._result_to_list(result))

        # Get SIMILAR_TO relationships
        result = self._conn.execute("""
            MATCH (p1:ProjectNode)-[r:SIMILAR_TO]->(p2:ProjectNode)
            RETURN p1.id AS source, p2.id AS target, 'SIMILAR_TO' AS type, r.similarity AS weight
        """)
        graph["relationships"].extend(self._result_to_list(result))

        return graph

    # ==================== Utilities ====================

    def _result_to_list(self, result) -> List[Dict[str, Any]]:
        """Convert Kuzu query result to list of dicts.

        Args:
            result: Kuzu query result.

        Returns:
            List of dictionaries.
        """
        rows = []
        column_names = result.get_column_names()

        while result.has_next():
            row = result.get_next()
            rows.append(dict(zip(column_names, row)))

        return rows

    def count_nodes(self) -> Dict[str, int]:
        """Get count of nodes by type.

        Returns:
            Dictionary with node type counts.
        """
        counts = {}

        for node_type in ["PatternNode", "ProjectNode", "CodeStructureNode"]:
            try:
                result = self._conn.execute(f"MATCH (n:{node_type}) RETURN COUNT(n) AS count")
                if result.has_next():
                    counts[node_type] = result.get_next()[0]
                else:
                    counts[node_type] = 0
            except Exception:
                counts[node_type] = 0

        return counts

    def count_relationships(self) -> Dict[str, int]:
        """Get count of relationships by type.

        Returns:
            Dictionary with relationship type counts.
        """
        counts = {}

        for rel_type in ["CO_OCCURS", "LEARNED_FROM", "SIMILAR_TO", "PREFERS", "EVOLVES_TO"]:
            try:
                result = self._conn.execute(
                    f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) AS count"
                )
                if result.has_next():
                    counts[rel_type] = result.get_next()[0]
                else:
                    counts[rel_type] = 0
            except Exception:
                counts[rel_type] = 0

        return counts

    def clear(self) -> None:
        """Clear all nodes and relationships."""
        try:
            # Delete all relationships
            for rel_type in ["CO_OCCURS", "LEARNED_FROM", "SIMILAR_TO", "PREFERS", "EVOLVES_TO"]:
                self._conn.execute(f"MATCH ()-[r:{rel_type}]->() DELETE r")

            # Delete all nodes
            for node_type in ["PatternNode", "ProjectNode", "CodeStructureNode"]:
                self._conn.execute(f"MATCH (n:{node_type}) DELETE n")

            logger.info("Cleared all nodes and relationships from MemoryGraph")
        except Exception as e:
            logger.warning(f"Error clearing graph: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get graph metrics.

        Returns:
            Dictionary with graph statistics.
        """
        return {
            "backend": "kuzudb",
            "persist_dir": str(self.persist_dir),
            "node_counts": self.count_nodes(),
            "relationship_counts": self.count_relationships(),
        }
