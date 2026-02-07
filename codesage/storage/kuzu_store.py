"""KuzuDB graph store for code relationships.

KuzuDB provides efficient graph queries for understanding code structure:
- What functions call this function?
- What classes does this class inherit from?
- What modules does this file import?
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from codesage.utils.logging import get_logger

logger = get_logger("storage.kuzu")


@dataclass
class CodeNode:
    """Represents a node in the code graph.

    Attributes:
        id: Unique identifier.
        name: Name of the code element.
        node_type: Type of node (function, class, file, module).
        file: Source file path.
        line_start: Starting line number.
        line_end: Ending line number.
        language: Programming language.
    """

    id: str
    name: str
    node_type: str  # "function", "class", "method", "file", "module"
    file: str
    line_start: int = 0
    line_end: int = 0
    language: str = "python"


@dataclass
class CodeRelationship:
    """Represents a relationship between code nodes.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        rel_type: Type of relationship.
        metadata: Additional relationship metadata.
    """

    source_id: str
    target_id: str
    rel_type: str  # "CALLS", "IMPORTS", "INHERITS", "CONTAINS", "USES"
    metadata: Optional[Dict[str, Any]] = None


class KuzuGraphStore:
    """KuzuDB graph store for code relationships.

    Stores and queries relationships between code elements:
    - CALLS: Function/method calls another function/method
    - IMPORTS: File/module imports another module
    - INHERITS: Class inherits from another class
    - CONTAINS: File contains function/class, class contains method
    - USES: Function/method uses a variable/class

    Advantages:
        - Embedded (no external server)
        - Fast startup (<100ms)
        - Low memory footprint (~50MB for 10M nodes)
        - Native Cypher support
        - ACID compliant
    """

    def __init__(self, persist_dir: Union[str, Path]) -> None:
        """Initialize the KuzuDB graph store.

        Args:
            persist_dir: Directory for KuzuDB persistence.
        """
        try:
            import kuzu
        except ImportError:
            raise ImportError(
                "KuzuDB is required for the graph store. "
                "Install with: pipx inject pycodesage kuzu (or pip install kuzu)"
            )

        self.persist_dir = Path(persist_dir)

        # Ensure parent directory exists (Kuzu 0.11+ creates the db dir itself)
        self.persist_dir.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database - Kuzu will create the persist_dir
        self._db = kuzu.Database(str(self.persist_dir))
        self._conn = kuzu.Connection(self._db)

        # Initialize schema
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize graph schema with node and relationship tables."""
        try:
            # Create node table for code elements
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS CodeNode (
                    id STRING,
                    name STRING,
                    node_type STRING,
                    file STRING,
                    line_start INT32,
                    line_end INT32,
                    language STRING,
                    PRIMARY KEY (id)
                )
            """)

            # Create relationship tables
            # CALLS: function/method calls another
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS CALLS (
                    FROM CodeNode TO CodeNode,
                    call_line INT32
                )
            """)

            # IMPORTS: file/module imports another
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS IMPORTS (
                    FROM CodeNode TO CodeNode,
                    import_type STRING
                )
            """)

            # INHERITS: class inherits from another
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS INHERITS (
                    FROM CodeNode TO CodeNode
                )
            """)

            # CONTAINS: file contains class/function, class contains method
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS CONTAINS (
                    FROM CodeNode TO CodeNode
                )
            """)

            # USES: function/method uses a variable/type
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS USES (
                    FROM CodeNode TO CodeNode,
                    usage_type STRING
                )
            """)

            logger.debug("KuzuDB schema initialized")

        except Exception as e:
            # Schema might already exist
            logger.debug(f"Schema init note: {e}")

    def add_node(self, node: CodeNode) -> None:
        """Add a code node to the graph.

        Args:
            node: CodeNode to add.
        """
        try:
            self._conn.execute(
                """
                MERGE (n:CodeNode {id: $id})
                SET n.name = $name,
                    n.node_type = $node_type,
                    n.file = $file,
                    n.line_start = $line_start,
                    n.line_end = $line_end,
                    n.language = $language
                """,
                {
                    "id": node.id,
                    "name": node.name,
                    "node_type": node.node_type,
                    "file": node.file,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                    "language": node.language,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to add node {node.id}: {e}")

    def add_nodes(self, nodes: List[CodeNode]) -> None:
        """Add multiple code nodes to the graph.

        Args:
            nodes: List of CodeNodes to add.
        """
        for node in nodes:
            self.add_node(node)

    def add_relationship(self, rel: CodeRelationship) -> None:
        """Add a relationship between nodes.

        Args:
            rel: CodeRelationship to add.
        """
        try:
            # Build query based on relationship type
            if rel.rel_type == "CALLS":
                call_line = rel.metadata.get("call_line", 0) if rel.metadata else 0
                self._conn.execute(
                    """
                    MATCH (a:CodeNode {id: $source_id}), (b:CodeNode {id: $target_id})
                    MERGE (a)-[r:CALLS]->(b)
                    SET r.call_line = $call_line
                    """,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "call_line": call_line,
                    },
                )
            elif rel.rel_type == "IMPORTS":
                import_type = rel.metadata.get("import_type", "module") if rel.metadata else "module"
                self._conn.execute(
                    """
                    MATCH (a:CodeNode {id: $source_id}), (b:CodeNode {id: $target_id})
                    MERGE (a)-[r:IMPORTS]->(b)
                    SET r.import_type = $import_type
                    """,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "import_type": import_type,
                    },
                )
            elif rel.rel_type == "INHERITS":
                self._conn.execute(
                    """
                    MATCH (a:CodeNode {id: $source_id}), (b:CodeNode {id: $target_id})
                    MERGE (a)-[:INHERITS]->(b)
                    """,
                    {"source_id": rel.source_id, "target_id": rel.target_id},
                )
            elif rel.rel_type == "CONTAINS":
                self._conn.execute(
                    """
                    MATCH (a:CodeNode {id: $source_id}), (b:CodeNode {id: $target_id})
                    MERGE (a)-[:CONTAINS]->(b)
                    """,
                    {"source_id": rel.source_id, "target_id": rel.target_id},
                )
            elif rel.rel_type == "USES":
                usage_type = rel.metadata.get("usage_type", "reference") if rel.metadata else "reference"
                self._conn.execute(
                    """
                    MATCH (a:CodeNode {id: $source_id}), (b:CodeNode {id: $target_id})
                    MERGE (a)-[r:USES]->(b)
                    SET r.usage_type = $usage_type
                    """,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "usage_type": usage_type,
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to add relationship {rel.source_id}->{rel.target_id}: {e}")

    def add_relationships(self, rels: List[CodeRelationship]) -> None:
        """Add multiple relationships.

        Args:
            rels: List of CodeRelationships to add.
        """
        for rel in rels:
            self.add_relationship(rel)

    def get_callers(self, function_id: str) -> List[Dict[str, Any]]:
        """Get all functions that call a given function.

        Args:
            function_id: ID of the target function.

        Returns:
            List of caller nodes with relationship info.
        """
        result = self._conn.execute(
            """
            MATCH (caller:CodeNode)-[r:CALLS]->(target:CodeNode {id: $id})
            RETURN caller.id AS id, caller.name AS name, caller.file AS file,
                   caller.line_start AS line_start, r.call_line AS call_line
            """,
            {"id": function_id},
        )
        return self._result_to_list(result)

    def get_callees(self, function_id: str) -> List[Dict[str, Any]]:
        """Get all functions called by a given function.

        Args:
            function_id: ID of the source function.

        Returns:
            List of callee nodes with relationship info.
        """
        result = self._conn.execute(
            """
            MATCH (source:CodeNode {id: $id})-[r:CALLS]->(callee:CodeNode)
            RETURN callee.id AS id, callee.name AS name, callee.file AS file,
                   callee.line_start AS line_start, r.call_line AS call_line
            """,
            {"id": function_id},
        )
        return self._result_to_list(result)

    def get_imports(self, file_id: str) -> List[Dict[str, Any]]:
        """Get all modules imported by a file.

        Args:
            file_id: ID of the source file.

        Returns:
            List of imported module nodes.
        """
        result = self._conn.execute(
            """
            MATCH (file:CodeNode {id: $id})-[r:IMPORTS]->(module:CodeNode)
            RETURN module.id AS id, module.name AS name, r.import_type AS import_type
            """,
            {"id": file_id},
        )
        return self._result_to_list(result)

    def get_importers(self, module_id: str) -> List[Dict[str, Any]]:
        """Get all files that import a given module.

        Args:
            module_id: ID of the target module.

        Returns:
            List of importer file nodes.
        """
        result = self._conn.execute(
            """
            MATCH (file:CodeNode)-[r:IMPORTS]->(module:CodeNode {id: $id})
            RETURN file.id AS id, file.name AS name, file.file AS file
            """,
            {"id": module_id},
        )
        return self._result_to_list(result)

    def get_superclasses(self, class_id: str) -> List[Dict[str, Any]]:
        """Get parent classes of a given class.

        Args:
            class_id: ID of the child class.

        Returns:
            List of parent class nodes.
        """
        result = self._conn.execute(
            """
            MATCH (child:CodeNode {id: $id})-[:INHERITS]->(parent:CodeNode)
            RETURN parent.id AS id, parent.name AS name, parent.file AS file
            """,
            {"id": class_id},
        )
        return self._result_to_list(result)

    def get_subclasses(self, class_id: str) -> List[Dict[str, Any]]:
        """Get child classes of a given class.

        Args:
            class_id: ID of the parent class.

        Returns:
            List of child class nodes.
        """
        result = self._conn.execute(
            """
            MATCH (child:CodeNode)-[:INHERITS]->(parent:CodeNode {id: $id})
            RETURN child.id AS id, child.name AS name, child.file AS file
            """,
            {"id": class_id},
        )
        return self._result_to_list(result)

    def get_dependencies(self, node_id: str) -> List[Dict[str, Any]]:
        """Get nodes this node depends on (calls, imports, inherits, uses).

        Args:
            node_id: ID of the source node.

        Returns:
            List of dependency nodes with relationship type.
        """
        # Use UNION ALL queries for KuzuDB 0.11.3+ compatibility (type(r) not supported)
        result = self._conn.execute(
            """
            MATCH (source:CodeNode {id: $id})-[:CALLS]->(t:CodeNode)
            RETURN t.id AS id, t.name AS name, t.file AS file, "CALLS" AS rel_type
            UNION ALL
            MATCH (source:CodeNode {id: $id})-[:IMPORTS]->(t:CodeNode)
            RETURN t.id AS id, t.name AS name, t.file AS file, "IMPORTS" AS rel_type
            UNION ALL
            MATCH (source:CodeNode {id: $id})-[:INHERITS]->(t:CodeNode)
            RETURN t.id AS id, t.name AS name, t.file AS file, "INHERITS" AS rel_type
            UNION ALL
            MATCH (source:CodeNode {id: $id})-[:USES]->(t:CodeNode)
            RETURN t.id AS id, t.name AS name, t.file AS file, "USES" AS rel_type
            """,
            {"id": node_id},
        )
        return self._result_to_list(result)

    def get_dependents(self, node_id: str) -> List[Dict[str, Any]]:
        """Get nodes that depend on this node.

        Args:
            node_id: ID of the target node.

        Returns:
            List of dependent nodes with relationship type.
        """
        # Use UNION ALL queries for KuzuDB 0.11.3+ compatibility (type(r) not supported)
        result = self._conn.execute(
            """
            MATCH (s:CodeNode)-[:CALLS]->(target:CodeNode {id: $id})
            RETURN s.id AS id, s.name AS name, s.file AS file, "CALLS" AS rel_type
            UNION ALL
            MATCH (s:CodeNode)-[:IMPORTS]->(target:CodeNode {id: $id})
            RETURN s.id AS id, s.name AS name, s.file AS file, "IMPORTS" AS rel_type
            UNION ALL
            MATCH (s:CodeNode)-[:INHERITS]->(target:CodeNode {id: $id})
            RETURN s.id AS id, s.name AS name, s.file AS file, "INHERITS" AS rel_type
            UNION ALL
            MATCH (s:CodeNode)-[:USES]->(target:CodeNode {id: $id})
            RETURN s.id AS id, s.name AS name, s.file AS file, "USES" AS rel_type
            """,
            {"id": node_id},
        )
        return self._result_to_list(result)

    def get_transitive_callers(
        self,
        node_id: str,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get transitive callers up to a max depth.

        Args:
            node_id: ID of the target node.
            max_depth: Maximum traversal depth.

        Returns:
            List of caller nodes.
        """
        result = self._conn.execute(
            """
            MATCH (caller:CodeNode)-[:CALLS*1..$depth]->(target:CodeNode {id: $id})
            RETURN DISTINCT caller.id AS id, caller.name AS name, caller.file AS file
            """,
            {"id": node_id, "depth": max_depth},
        )
        return self._result_to_list(result)

    def get_file_contents(self, file_id: str) -> List[Dict[str, Any]]:
        """Get all code elements contained in a file.

        Args:
            file_id: ID of the file.

        Returns:
            List of contained code element nodes.
        """
        result = self._conn.execute(
            """
            MATCH (file:CodeNode {id: $id})-[:CONTAINS]->(element:CodeNode)
            RETURN element.id AS id, element.name AS name, element.node_type AS type,
                   element.line_start AS line_start
            ORDER BY element.line_start
            """,
            {"id": file_id},
        )
        return self._result_to_list(result)

    def find_call_chain(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Find call chains between two functions.

        Args:
            start_id: ID of the starting function.
            end_id: ID of the ending function.
            max_depth: Maximum chain length to search.

        Returns:
            List of paths (each path is a list of nodes).
        """
        result = self._conn.execute(
            f"""
            MATCH path = (start:CodeNode {{id: $start_id}})-[:CALLS*1..{max_depth}]->(end:CodeNode {{id: $end_id}})
            RETURN nodes(path) AS chain
            LIMIT 10
            """,
            {"start_id": start_id, "end_id": end_id},
        )

        paths = []
        while result.has_next():
            row = result.get_next()
            chain = row[0] if row else []
            paths.append([{"id": n.get("id"), "name": n.get("name")} for n in chain])

        return paths

    def find_nodes_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find nodes by name.

        Args:
            name: Name of the code element to find.

        Returns:
            List of matching nodes.
        """
        result = self._conn.execute(
            """
            MATCH (n:CodeNode)
            WHERE n.name = $name
            RETURN n.id AS id, n.name AS name, n.node_type AS type,
                   n.file AS file, n.line_start AS line_start, n.line_end AS line_end
            """,
            {"name": name},
        )
        return self._result_to_list(result)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID.

        Args:
            node_id: ID of the node.

        Returns:
            Node data or None if not found.
        """
        result = self._conn.execute(
            """
            MATCH (n:CodeNode {id: $id})
            RETURN n.id AS id, n.name AS name, n.node_type AS type,
                   n.file AS file, n.line_start AS line_start, n.line_end AS line_end
            """,
            {"id": node_id},
        )
        nodes = self._result_to_list(result)
        return nodes[0] if nodes else None

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its relationships.

        Args:
            node_id: ID of the node to delete.
        """
        try:
            # Delete relationships first (must use directed patterns - Kuzu doesn't support undirected delete)
            for rel_type in ["CALLS", "IMPORTS", "INHERITS", "CONTAINS", "USES"]:
                # Delete outgoing relationships from this node
                self._conn.execute(
                    f"""
                    MATCH (n:CodeNode {{id: $id}})-[r:{rel_type}]->()
                    DELETE r
                    """,
                    {"id": node_id},
                )
                # Delete incoming relationships to this node
                self._conn.execute(
                    f"""
                    MATCH ()-[r:{rel_type}]->(n:CodeNode {{id: $id}})
                    DELETE r
                    """,
                    {"id": node_id},
                )

            # Delete node
            self._conn.execute(
                "MATCH (n:CodeNode {id: $id}) DELETE n",
                {"id": node_id},
            )
        except Exception as e:
            logger.warning(f"Failed to delete node {node_id}: {e}")

    def delete_by_file(self, file_path: Path) -> None:
        """Delete all nodes and relationships for a file.

        Args:
            file_path: Path to the file.
        """
        # Get all nodes for this file
        result = self._conn.execute(
            """
            MATCH (n:CodeNode {file: $file})
            RETURN n.id AS id
            """,
            {"file": str(file_path)},
        )

        nodes = self._result_to_list(result)
        for node in nodes:
            self.delete_node(node["id"])

    def count_nodes(self) -> int:
        """Get total count of nodes.

        Returns:
            Number of nodes in the graph.
        """
        try:
            result = self._conn.execute("MATCH (n:CodeNode) RETURN COUNT(n) AS count")
            if result.has_next():
                return result.get_next()[0]
            return 0
        except Exception:
            return 0

    def count_relationships(self) -> Dict[str, int]:
        """Get count of relationships by type.

        Returns:
            Dictionary with relationship type counts.
        """
        counts = {}
        for rel_type in ["CALLS", "IMPORTS", "INHERITS", "CONTAINS", "USES"]:
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
            # Delete all relationships first
            for rel_type in ["CALLS", "IMPORTS", "INHERITS", "CONTAINS", "USES"]:
                self._conn.execute(f"MATCH ()-[r:{rel_type}]->() DELETE r")

            # Delete all nodes
            self._conn.execute("MATCH (n:CodeNode) DELETE n")
            logger.info("Cleared all nodes and relationships from KuzuDB")
        except Exception as e:
            logger.warning(f"Error clearing graph: {e}")

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

    def get_metrics(self) -> Dict[str, Any]:
        """Get graph metrics.

        Returns:
            Dictionary with graph statistics.
        """
        return {
            "backend": "kuzudb",
            "node_count": self.count_nodes(),
            "relationship_counts": self.count_relationships(),
            "persist_dir": str(self.persist_dir),
        }
