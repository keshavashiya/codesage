"""Tests for KuzuDB graph store."""

import pytest
from pathlib import Path

# Skip all tests if kuzu not installed
pytest.importorskip("kuzu")

from codesage.storage.kuzu_store import KuzuGraphStore, CodeNode, CodeRelationship


@pytest.fixture
def kuzu_store(tmp_path):
    """Create KuzuGraphStore instance."""
    return KuzuGraphStore(persist_dir=tmp_path / "kuzu_test")


class TestKuzuGraphStoreInit:
    """Tests for KuzuGraphStore initialization."""

    def test_creates_persist_directory(self, tmp_path):
        """Test that persist directory is created."""
        store_path = tmp_path / "new_kuzu_store"
        assert not store_path.exists()

        store = KuzuGraphStore(store_path)

        assert store_path.exists()

    def test_initializes_schema(self, kuzu_store):
        """Test that schema is initialized."""
        # Should be able to query nodes table
        assert kuzu_store.count_nodes() == 0


class TestKuzuGraphStoreNodes:
    """Tests for node operations."""

    def test_add_node(self, kuzu_store):
        """Test adding a single node."""
        node = CodeNode(
            id="func1",
            name="my_function",
            node_type="function",
            file="test.py",
            line_start=1,
            line_end=10,
            language="python",
        )

        kuzu_store.add_node(node)

        assert kuzu_store.count_nodes() == 1

    def test_add_multiple_nodes(self, kuzu_store):
        """Test adding multiple nodes."""
        nodes = [
            CodeNode(id=f"node{i}", name=f"func{i}", node_type="function", file="test.py")
            for i in range(5)
        ]

        kuzu_store.add_nodes(nodes)

        assert kuzu_store.count_nodes() == 5

    def test_get_node(self, kuzu_store):
        """Test retrieving a node by ID."""
        node = CodeNode(
            id="func1",
            name="my_function",
            node_type="function",
            file="test.py",
            line_start=5,
            line_end=15,
        )
        kuzu_store.add_node(node)

        result = kuzu_store.get_node("func1")

        assert result is not None
        assert result["id"] == "func1"
        assert result["name"] == "my_function"
        assert result["type"] == "function"

    def test_get_nonexistent_node(self, kuzu_store):
        """Test retrieving a node that doesn't exist."""
        result = kuzu_store.get_node("nonexistent")
        assert result is None

    def test_delete_node(self, kuzu_store):
        """Test deleting a node."""
        kuzu_store.add_node(CodeNode(id="func1", name="func1", node_type="function", file="test.py"))
        kuzu_store.add_node(CodeNode(id="func2", name="func2", node_type="function", file="test.py"))
        assert kuzu_store.count_nodes() == 2

        kuzu_store.delete_node("func1")

        assert kuzu_store.count_nodes() == 1
        assert kuzu_store.get_node("func1") is None
        assert kuzu_store.get_node("func2") is not None


class TestKuzuGraphStoreRelationships:
    """Tests for relationship operations."""

    @pytest.fixture(autouse=True)
    def setup_nodes(self, kuzu_store):
        """Set up test nodes."""
        nodes = [
            CodeNode(id="main", name="main", node_type="function", file="app.py"),
            CodeNode(id="helper", name="helper", node_type="function", file="utils.py"),
            CodeNode(id="process", name="process", node_type="function", file="core.py"),
            CodeNode(id="BaseClass", name="BaseClass", node_type="class", file="base.py"),
            CodeNode(id="ChildClass", name="ChildClass", node_type="class", file="child.py"),
            CodeNode(id="app_file", name="app.py", node_type="file", file="app.py"),
            CodeNode(id="utils_module", name="utils", node_type="module", file="utils.py"),
        ]
        kuzu_store.add_nodes(nodes)

    def test_add_calls_relationship(self, kuzu_store):
        """Test adding a CALLS relationship."""
        rel = CodeRelationship(
            source_id="main",
            target_id="helper",
            rel_type="CALLS",
            metadata={"call_line": 10},
        )

        kuzu_store.add_relationship(rel)

        counts = kuzu_store.count_relationships()
        assert counts["CALLS"] >= 1

    def test_add_imports_relationship(self, kuzu_store):
        """Test adding an IMPORTS relationship."""
        rel = CodeRelationship(
            source_id="app_file",
            target_id="utils_module",
            rel_type="IMPORTS",
            metadata={"import_type": "from"},
        )

        kuzu_store.add_relationship(rel)

        counts = kuzu_store.count_relationships()
        assert counts["IMPORTS"] >= 1

    def test_add_inherits_relationship(self, kuzu_store):
        """Test adding an INHERITS relationship."""
        rel = CodeRelationship(
            source_id="ChildClass",
            target_id="BaseClass",
            rel_type="INHERITS",
        )

        kuzu_store.add_relationship(rel)

        counts = kuzu_store.count_relationships()
        assert counts["INHERITS"] >= 1

    def test_add_contains_relationship(self, kuzu_store):
        """Test adding a CONTAINS relationship."""
        rel = CodeRelationship(
            source_id="app_file",
            target_id="main",
            rel_type="CONTAINS",
        )

        kuzu_store.add_relationship(rel)

        counts = kuzu_store.count_relationships()
        assert counts["CONTAINS"] >= 1


class TestKuzuGraphStoreQueries:
    """Tests for graph queries."""

    @pytest.fixture(autouse=True)
    def setup_graph(self, kuzu_store):
        """Set up test graph."""
        # Add nodes
        nodes = [
            CodeNode(id="main", name="main", node_type="function", file="app.py", line_start=1),
            CodeNode(id="helper1", name="helper1", node_type="function", file="utils.py", line_start=5),
            CodeNode(id="helper2", name="helper2", node_type="function", file="utils.py", line_start=15),
            CodeNode(id="process", name="process", node_type="function", file="core.py", line_start=1),
            CodeNode(id="Base", name="Base", node_type="class", file="base.py"),
            CodeNode(id="Child", name="Child", node_type="class", file="child.py"),
            CodeNode(id="app_file", name="app.py", node_type="file", file="app.py"),
            CodeNode(id="utils_mod", name="utils", node_type="module", file="utils.py"),
        ]
        kuzu_store.add_nodes(nodes)

        # Add relationships
        rels = [
            # main calls helper1 and helper2
            CodeRelationship("main", "helper1", "CALLS", {"call_line": 10}),
            CodeRelationship("main", "helper2", "CALLS", {"call_line": 15}),
            # helper1 calls process
            CodeRelationship("helper1", "process", "CALLS", {"call_line": 8}),
            # Child inherits from Base
            CodeRelationship("Child", "Base", "INHERITS"),
            # app_file imports utils_mod
            CodeRelationship("app_file", "utils_mod", "IMPORTS", {"import_type": "import"}),
            # app_file contains main
            CodeRelationship("app_file", "main", "CONTAINS"),
        ]
        kuzu_store.add_relationships(rels)

    def test_get_callers(self, kuzu_store):
        """Test getting functions that call a function."""
        callers = kuzu_store.get_callers("helper1")

        assert len(callers) == 1
        assert callers[0]["name"] == "main"

    def test_get_callees(self, kuzu_store):
        """Test getting functions called by a function."""
        callees = kuzu_store.get_callees("main")

        assert len(callees) == 2
        names = {c["name"] for c in callees}
        assert names == {"helper1", "helper2"}

    def test_get_superclasses(self, kuzu_store):
        """Test getting parent classes."""
        parents = kuzu_store.get_superclasses("Child")

        assert len(parents) == 1
        assert parents[0]["name"] == "Base"

    def test_get_subclasses(self, kuzu_store):
        """Test getting child classes."""
        children = kuzu_store.get_subclasses("Base")

        assert len(children) == 1
        assert children[0]["name"] == "Child"

    def test_get_imports(self, kuzu_store):
        """Test getting imported modules."""
        imports = kuzu_store.get_imports("app_file")

        assert len(imports) == 1
        assert imports[0]["name"] == "utils"

    def test_get_importers(self, kuzu_store):
        """Test getting files that import a module."""
        importers = kuzu_store.get_importers("utils_mod")

        assert len(importers) == 1
        assert importers[0]["name"] == "app.py"

    def test_get_file_contents(self, kuzu_store):
        """Test getting contents of a file."""
        contents = kuzu_store.get_file_contents("app_file")

        assert len(contents) == 1
        assert contents[0]["name"] == "main"


class TestKuzuGraphStoreClear:
    """Tests for clearing the graph."""

    def test_clear(self, kuzu_store):
        """Test clearing all data."""
        # Add some data
        kuzu_store.add_node(CodeNode(id="n1", name="n1", node_type="function", file="test.py"))
        kuzu_store.add_node(CodeNode(id="n2", name="n2", node_type="function", file="test.py"))
        kuzu_store.add_relationship(CodeRelationship("n1", "n2", "CALLS"))

        assert kuzu_store.count_nodes() == 2

        kuzu_store.clear()

        assert kuzu_store.count_nodes() == 0
        counts = kuzu_store.count_relationships()
        assert all(c == 0 for c in counts.values())

    def test_delete_by_file(self, kuzu_store):
        """Test deleting all nodes for a file."""
        kuzu_store.add_node(CodeNode(id="f1_1", name="func1", node_type="function", file="file1.py"))
        kuzu_store.add_node(CodeNode(id="f1_2", name="func2", node_type="function", file="file1.py"))
        kuzu_store.add_node(CodeNode(id="f2_1", name="func3", node_type="function", file="file2.py"))

        assert kuzu_store.count_nodes() == 3

        kuzu_store.delete_by_file(Path("file1.py"))

        assert kuzu_store.count_nodes() == 1
        assert kuzu_store.get_node("f2_1") is not None


class TestKuzuGraphStoreMetrics:
    """Tests for graph metrics."""

    def test_get_metrics(self, kuzu_store):
        """Test getting graph metrics."""
        kuzu_store.add_node(CodeNode(id="n1", name="n1", node_type="function", file="test.py"))
        kuzu_store.add_node(CodeNode(id="n2", name="n2", node_type="function", file="test.py"))
        kuzu_store.add_relationship(CodeRelationship("n1", "n2", "CALLS"))

        metrics = kuzu_store.get_metrics()

        assert metrics["backend"] == "kuzudb"
        assert metrics["node_count"] == 2
        assert metrics["relationship_counts"]["CALLS"] == 1
