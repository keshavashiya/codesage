"""Storage package exports."""

from codesage.storage.base import ConnectionManager, DatabaseError
from codesage.storage.database import Database
from codesage.storage.vector_base import VectorStoreBase
from codesage.storage.manager import StorageManager

# LanceDB is optional - only import if available
try:
    from codesage.storage.lance_store import LanceVectorStore, create_lance_embedding_fn
except ImportError:
    LanceVectorStore = None  # type: ignore
    create_lance_embedding_fn = None  # type: ignore

# KuzuDB is optional - only import if available
try:
    from codesage.storage.kuzu_store import KuzuGraphStore, CodeNode, CodeRelationship
except ImportError:
    KuzuGraphStore = None  # type: ignore
    CodeNode = None  # type: ignore
    CodeRelationship = None  # type: ignore

__all__ = [
    # SQLite
    "ConnectionManager",
    "Database",
    "DatabaseError",
    # Unified manager
    "StorageManager",
    # Vector stores
    "VectorStoreBase",
    "LanceVectorStore",
    "create_lance_embedding_fn",
    # Graph store
    "KuzuGraphStore",
    "CodeNode",
    "CodeRelationship",
]
