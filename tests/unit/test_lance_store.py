"""Tests for LanceDB vector store."""

import pytest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

# Skip all tests if lancedb not installed
pytest.importorskip("lancedb")


class MockEmbedder:
    """Mock embedding function for testing."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.call_count = 0

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings."""
        self.call_count += 1
        embeddings = []
        for i, text in enumerate(texts):
            # Generate deterministic embedding based on text hash
            seed = hash(text) % 10000
            embedding = [(seed + j) / 10000.0 for j in range(self.dim)]
            embeddings.append(embedding)
        return embeddings


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    return MockEmbedder(dim=1024)


@pytest.fixture
def lance_store(tmp_path, mock_embedder):
    """Create LanceVectorStore instance."""
    from codesage.storage.lance_store import LanceVectorStore

    return LanceVectorStore(
        persist_dir=tmp_path / "lance_test",
        embedding_fn=mock_embedder,
        vector_dim=1024,
    )


class TestLanceVectorStoreInit:
    """Tests for LanceVectorStore initialization."""

    def test_creates_persist_directory(self, tmp_path, mock_embedder):
        """Test that persist directory is created."""
        from codesage.storage.lance_store import LanceVectorStore

        store_path = tmp_path / "new_lance_store"
        assert not store_path.exists()

        store = LanceVectorStore(store_path, mock_embedder)

        assert store_path.exists()

    def test_creates_table(self, lance_store):
        """Test that table is created on init."""
        assert lance_store._table is not None
        assert lance_store.TABLE_NAME in lance_store._db.table_names()

    def test_reopens_existing_table(self, tmp_path, mock_embedder):
        """Test that existing table is reopened."""
        from codesage.storage.lance_store import LanceVectorStore

        # Create first store and add data
        store1 = LanceVectorStore(tmp_path / "lance", mock_embedder)
        store1.add(["id1"], ["test document"])

        # Create second store pointing to same location
        store2 = LanceVectorStore(tmp_path / "lance", mock_embedder)

        # Should have the same data
        assert store2.count() == 1


class TestLanceVectorStoreAdd:
    """Tests for adding documents."""

    def test_add_single_document(self, lance_store):
        """Test adding a single document."""
        lance_store.add(
            ids=["doc1"],
            documents=["This is a test document"],
            metadatas=[{"file": "test.py", "type": "function"}],
        )

        assert lance_store.count() == 1

    def test_add_multiple_documents(self, lance_store):
        """Test adding multiple documents."""
        lance_store.add(
            ids=["doc1", "doc2", "doc3"],
            documents=["First doc", "Second doc", "Third doc"],
            metadatas=[
                {"file": "a.py", "type": "function"},
                {"file": "b.py", "type": "class"},
                {"file": "c.py", "type": "method"},
            ],
        )

        assert lance_store.count() == 3

    def test_add_without_metadata(self, lance_store):
        """Test adding documents without metadata."""
        lance_store.add(
            ids=["doc1"],
            documents=["Test document"],
        )

        assert lance_store.count() == 1

    def test_add_empty_list(self, lance_store):
        """Test adding empty list does nothing."""
        lance_store.add(ids=[], documents=[])
        assert lance_store.count() == 0

    def test_add_truncates_long_documents(self, lance_store):
        """Test that long documents are truncated."""
        long_doc = "x" * 5000  # Longer than MAX_CHARS

        lance_store.add(ids=["doc1"], documents=[long_doc])

        # Document should still be added
        assert lance_store.count() == 1


class TestLanceVectorStoreQuery:
    """Tests for querying documents."""

    @pytest.fixture(autouse=True)
    def setup_data(self, lance_store):
        """Add test data before each test."""
        lance_store.add(
            ids=["func1", "func2", "class1"],
            documents=[
                "function: calculate_sum\nCode: def calculate_sum(a, b): return a + b",
                "function: calculate_product\nCode: def calculate_product(a, b): return a * b",
                "class: Calculator\nCode: class Calculator: pass",
            ],
            metadatas=[
                {"file": "math.py", "type": "function", "name": "calculate_sum", "language": "python", "line_start": 1, "line_end": 2},
                {"file": "math.py", "type": "function", "name": "calculate_product", "language": "python", "line_start": 4, "line_end": 5},
                {"file": "calc.py", "type": "class", "name": "Calculator", "language": "python", "line_start": 1, "line_end": 1},
            ],
        )

    def test_query_returns_results(self, lance_store):
        """Test that query returns results."""
        results = lance_store.query("calculate sum", n_results=2)

        assert len(results) <= 2
        assert all("id" in r for r in results)
        assert all("document" in r for r in results)
        assert all("similarity" in r for r in results)

    def test_query_with_filter(self, lance_store):
        """Test query with metadata filter."""
        results = lance_store.query(
            "function",
            n_results=10,
            where={"type": "function"},
        )

        # Should only return functions
        for r in results:
            assert r["metadata"]["type"] == "function"

    def test_query_respects_limit(self, lance_store):
        """Test that query respects n_results limit."""
        results = lance_store.query("code", n_results=1)
        assert len(results) <= 1

    def test_query_empty_store(self, tmp_path, mock_embedder):
        """Test querying empty store."""
        from codesage.storage.lance_store import LanceVectorStore

        empty_store = LanceVectorStore(tmp_path / "empty", mock_embedder)
        results = empty_store.query("test")

        assert results == []


class TestLanceVectorStoreDelete:
    """Tests for deleting documents."""

    def test_delete_by_id(self, lance_store):
        """Test deleting documents by ID."""
        lance_store.add(
            ids=["doc1", "doc2"],
            documents=["First", "Second"],
        )
        assert lance_store.count() == 2

        lance_store.delete(["doc1"])

        assert lance_store.count() == 1

    def test_delete_multiple_ids(self, lance_store):
        """Test deleting multiple documents."""
        lance_store.add(
            ids=["a", "b", "c"],
            documents=["A", "B", "C"],
        )

        lance_store.delete(["a", "c"])

        assert lance_store.count() == 1

    def test_delete_empty_list(self, lance_store):
        """Test deleting empty list does nothing."""
        lance_store.add(ids=["doc1"], documents=["Test"])
        lance_store.delete([])
        assert lance_store.count() == 1

    def test_delete_by_file(self, lance_store):
        """Test deleting all documents for a file."""
        lance_store.add(
            ids=["f1_1", "f1_2", "f2_1"],
            documents=["File 1 doc 1", "File 1 doc 2", "File 2 doc 1"],
            metadatas=[
                {"file": "file1.py"},
                {"file": "file1.py"},
                {"file": "file2.py"},
            ],
        )
        assert lance_store.count() == 3

        lance_store.delete_by_file(Path("file1.py"))

        assert lance_store.count() == 1


class TestLanceVectorStoreClear:
    """Tests for clearing the store."""

    def test_clear_removes_all_documents(self, lance_store):
        """Test that clear removes all documents."""
        lance_store.add(
            ids=["a", "b", "c"],
            documents=["A", "B", "C"],
        )
        assert lance_store.count() == 3

        lance_store.clear()

        assert lance_store.count() == 0

    def test_clear_empty_store(self, lance_store):
        """Test clearing empty store."""
        lance_store.clear()
        assert lance_store.count() == 0

    def test_can_add_after_clear(self, lance_store):
        """Test that we can add documents after clearing."""
        lance_store.add(ids=["a"], documents=["A"])
        lance_store.clear()
        lance_store.add(ids=["b"], documents=["B"])

        assert lance_store.count() == 1


class TestLanceVectorStoreMetrics:
    """Tests for storage metrics."""

    def test_get_metrics(self, lance_store):
        """Test getting storage metrics."""
        lance_store.add(ids=["a", "b"], documents=["A", "B"])

        metrics = lance_store.get_metrics()

        assert metrics["backend"] == "lancedb"
        assert metrics["document_count"] == 2
        assert metrics["vector_dim"] == 1024


class TestLanceVectorStoreCodeElement:
    """Tests for CodeElement integration."""

    def test_add_element(self, lance_store):
        """Test adding a CodeElement."""
        from codesage.models.code_element import CodeElement

        element = CodeElement.create(
            file=Path("test.py"),
            type="function",
            code="def hello(): pass",
            language="python",
            line_start=1,
            line_end=1,
            name="hello",
        )

        lance_store.add_element(element)

        assert lance_store.count() == 1

    def test_add_elements(self, lance_store):
        """Test adding multiple CodeElements."""
        from codesage.models.code_element import CodeElement

        elements = [
            CodeElement.create(
                file=Path("test.py"),
                type="function",
                code=f"def func{i}(): pass",
                language="python",
                line_start=i,
                line_end=i,
                name=f"func{i}",
            )
            for i in range(5)
        ]

        lance_store.add_elements(elements)

        assert lance_store.count() == 5


class TestCreateLanceEmbeddingFn:
    """Tests for embedding function wrapper."""

    def test_create_embedding_fn(self):
        """Test creating embedding function from LangChain embedder."""
        from codesage.storage.lance_store import create_lance_embedding_fn

        # Mock LangChain embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        embed_fn = create_lance_embedding_fn(mock_embedder)
        result = embed_fn(["test"])

        mock_embedder.embed_documents.assert_called_once_with(["test"])
        assert result == [[0.1, 0.2, 0.3]]
