"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from codesage.models.code_element import CodeElement


class VectorStoreBase(ABC):
    """Abstract base class for vector storage implementations.

    Provides a common interface for vector database backends (LanceDB).
    """

    # Max characters for embedding (prevents context length errors)
    MAX_CHARS = 1500

    def _truncate(self, text: str) -> str:
        """Truncate text to fit embedding model context.

        Uses smart truncation that tries to preserve:
        1. Function/class name and type
        2. Signature
        3. Docstring (first part)
        4. Code (truncated if needed)

        Args:
            text: Text to truncate.

        Returns:
            Truncated text.
        """
        if len(text) <= self.MAX_CHARS:
            return text

        # Try to find a good truncation point
        code_marker = "\nCode:\n"
        if code_marker in text:
            before_code = text.split(code_marker)[0]
            code_part = text.split(code_marker)[1] if len(text.split(code_marker)) > 1 else ""

            remaining = self.MAX_CHARS - len(before_code) - len(code_marker) - 30

            if remaining > 100:
                truncated_code = code_part[:remaining] + "\n... [truncated]"
                return before_code + code_marker + truncated_code
            else:
                return before_code[: self.MAX_CHARS - 20] + "\n... [truncated]"

        return text[: self.MAX_CHARS - 20] + "\n... [truncated]"

    @abstractmethod
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
        pass

    def add_element(self, element: CodeElement) -> None:
        """Add a single code element.

        Args:
            element: Code element to add.
        """
        self.add(
            ids=[element.id],
            documents=[element.get_embedding_text()],
            metadatas=[
                {
                    "id": element.id,
                    "file": str(element.file),
                    "type": element.type,
                    "name": element.name or "",
                    "language": element.language,
                    "line_start": element.line_start,
                    "line_end": element.line_end,
                }
            ],
        )

    def add_elements(self, elements: List[CodeElement]) -> None:
        """Add multiple code elements.

        Args:
            elements: List of code elements to add.
        """
        if not elements:
            return

        self.add(
            ids=[e.id for e in elements],
            documents=[e.get_embedding_text() for e in elements],
            metadatas=[
                {
                    "id": e.id,
                    "file": str(e.file),
                    "type": e.type,
                    "name": e.name or "",
                    "language": e.language,
                    "line_start": e.line_start,
                    "line_end": e.line_end,
                }
                for e in elements
            ],
        )

    @abstractmethod
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
            where: Optional filter dict.

        Returns:
            List of results with id, document, metadata, similarity.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of IDs to delete.
        """
        pass

    @abstractmethod
    def delete_by_file(self, file_path: Path) -> None:
        """Delete all documents from a specific file.

        Args:
            file_path: Path to file.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total count of documents.

        Returns:
            Number of documents in the store.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass
