"""ChromaDB vector store via LangChain."""

from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from codesage.models.code_element import CodeElement


class VectorStore:
    """ChromaDB vector store via LangChain.

    Provides semantic search over code elements using
    embeddings stored in ChromaDB.
    """

    COLLECTION_NAME = "code_elements"

    # Max characters for embedding (prevents context length errors)
    # mxbai-embed-large has 512 token context (~1500 chars safely)
    # nomic-embed-text has 8192 tokens but we use conservative limit
    MAX_CHARS = 1500

    def __init__(self, persist_dir: Path, embeddings: Embeddings):
        """Initialize the vector store.

        Args:
            persist_dir: Directory for ChromaDB persistence
            embeddings: LangChain embeddings instance
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
        )

    @property
    def store(self) -> Chroma:
        """Get the underlying Chroma store."""
        return self._store

    def _truncate(self, text: str) -> str:
        """Truncate text to fit embedding model context.

        Uses smart truncation that tries to preserve:
        1. Function/class name and type
        2. Signature
        3. Docstring (first part)
        4. Code (truncated if needed)
        """
        if len(text) <= self.MAX_CHARS:
            return text

        # Try to find a good truncation point
        # Look for "Code:" marker and truncate the code portion
        code_marker = "\nCode:\n"
        if code_marker in text:
            before_code = text.split(code_marker)[0]
            code_part = text.split(code_marker)[1] if len(text.split(code_marker)) > 1 else ""

            # Calculate remaining space for code
            remaining = self.MAX_CHARS - len(before_code) - len(code_marker) - 30

            if remaining > 100:  # Only include code if we have reasonable space
                truncated_code = code_part[:remaining] + "\n... [truncated]"
                return before_code + code_marker + truncated_code
            else:
                # Just keep metadata without code
                return before_code[:self.MAX_CHARS - 20] + "\n... [truncated]"

        # Fallback: simple truncation
        return text[:self.MAX_CHARS - 20] + "\n... [truncated]"

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            ids: List of unique IDs
            documents: List of text documents
            metadatas: Optional list of metadata dicts
        """
        # Truncate documents to prevent context length errors
        truncated_docs = [self._truncate(doc) for doc in documents]

        self._store.add_texts(
            texts=truncated_docs,
            ids=ids,
            metadatas=metadatas,
        )

    def add_element(self, element: CodeElement) -> None:
        """Add a single code element.

        Args:
            element: Code element to add
        """
        self.add(
            ids=[element.id],
            documents=[element.get_embedding_text()],
            metadatas=[{
                "file": str(element.file),
                "type": element.type,
                "name": element.name or "",
                "language": element.language,
                "line_start": element.line_start,
                "line_end": element.line_end,
            }],
        )

    def add_elements(self, elements: List[CodeElement]) -> None:
        """Add multiple code elements.

        Args:
            elements: List of code elements to add
        """
        if not elements:
            return

        self.add(
            ids=[e.id for e in elements],
            documents=[e.get_embedding_text() for e in elements],
            metadatas=[{
                "file": str(e.file),
                "type": e.type,
                "name": e.name or "",
                "language": e.language,
                "line_start": e.line_start,
                "line_end": e.line_end,
            } for e in elements],
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query for similar documents.

        Args:
            query_text: Text to search for
            n_results: Number of results to return
            where: Optional filter dict

        Returns:
            List of results with id, document, metadata, and distance
        """
        # Truncate query if needed
        query_text = self._truncate(query_text)

        results = self._store.similarity_search_with_score(
            query=query_text,
            k=n_results,
            filter=where,
        )

        return [
            {
                "id": doc.metadata.get("id", ""),
                "document": doc.page_content,
                "metadata": doc.metadata,
                "distance": score,
                "similarity": 1 - score,  # Convert distance to similarity
            }
            for doc, score in results
        ]

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of IDs to delete
        """
        self._store.delete(ids=ids)

    def delete_by_file(self, file_path: Path) -> None:
        """Delete all documents from a specific file.

        Args:
            file_path: Path to file
        """
        # ChromaDB supports filtering on delete
        self._store.delete(
            where={"file": str(file_path)}
        )

    def count(self) -> int:
        """Get total count of documents.

        Returns:
            Number of documents in the store
        """
        return len(self._store.get()["ids"])

    def clear(self) -> None:
        """Clear all documents from the store."""
        # Get all IDs and delete them
        all_data = self._store.get()
        if all_data["ids"]:
            self._store.delete(ids=all_data["ids"])
