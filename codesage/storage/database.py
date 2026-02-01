"""SQLite database for storing code metadata."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from codesage.models.code_element import CodeElement
from codesage.storage.base import ConnectionManager, DatabaseError
from codesage.utils.logging import get_logger

logger = get_logger("storage.database")

# Re-export for backwards compatibility
__all__ = ["Database", "DatabaseError"]


class Database(ConnectionManager):
    """SQLite database for code element metadata.

    Stores metadata about indexed code elements, patterns,
    and indexing statistics.

    Security features:
        - File permissions hardened to owner-only (600)
        - Parameterized queries throughout
        - Transaction support for data integrity
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize the database.

        Args:
            db_path: Path to the SQLite database file.
        """
        super().__init__(db_path)
        self._init_schema()
        self._harden_permissions()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.executescript("""
            -- Code elements table
            CREATE TABLE IF NOT EXISTS code_elements (
                id TEXT PRIMARY KEY,
                file TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                docstring TEXT,
                signature TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_elements_file ON code_elements(file);
            CREATE INDEX IF NOT EXISTS idx_elements_type ON code_elements(type);
            CREATE INDEX IF NOT EXISTS idx_elements_name ON code_elements(name);
            CREATE INDEX IF NOT EXISTS idx_elements_language ON code_elements(language);

            -- Patterns table
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                occurrences INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Index statistics
            CREATE TABLE IF NOT EXISTS index_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_files INTEGER DEFAULT 0,
                total_elements INTEGER DEFAULT 0,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Initialize stats row
            INSERT OR IGNORE INTO index_stats (id) VALUES (1);

            -- File tracking for incremental indexing
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Code Element Operations
    # -------------------------------------------------------------------------

    def store_element(self, element: CodeElement) -> None:
        """Store a code element.

        Args:
            element: Code element to store.
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO code_elements
            (id, file, type, name, code, language, line_start, line_end,
             docstring, signature, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                element.id,
                str(element.file),
                element.type,
                element.name,
                element.code,
                element.language,
                element.line_start,
                element.line_end,
                element.docstring,
                element.signature,
            ),
        )
        self.conn.commit()

    def store_elements(self, elements: List[CodeElement]) -> None:
        """Bulk store code elements.

        Args:
            elements: List of code elements to store.
        """
        with self.transaction():
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO code_elements
                (id, file, type, name, code, language, line_start, line_end,
                 docstring, signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    (
                        e.id,
                        str(e.file),
                        e.type,
                        e.name,
                        e.code,
                        e.language,
                        e.line_start,
                        e.line_end,
                        e.docstring,
                        e.signature,
                    )
                    for e in elements
                ],
            )

    def get_element(self, element_id: str) -> Optional[CodeElement]:
        """Retrieve an element by ID.

        Args:
            element_id: Element ID.

        Returns:
            CodeElement or None if not found.
        """
        cursor = self.conn.execute(
            """
            SELECT id, file, type, name, code, language, line_start, line_end,
                   docstring, signature
            FROM code_elements WHERE id = ?
            """,
            (element_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        return CodeElement(
            id=row["id"],
            file=Path(row["file"]),
            type=row["type"],
            name=row["name"],
            code=row["code"],
            language=row["language"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            docstring=row["docstring"],
            signature=row["signature"],
        )

    def get_all_elements(self, batch_size: int = 1000) -> Iterator[CodeElement]:
        """Get all code elements with pagination.

        Uses a generator to avoid loading all elements into memory at once.

        Args:
            batch_size: Number of elements to fetch per batch.

        Yields:
            CodeElement instances.
        """
        offset = 0

        while True:
            cursor = self.conn.execute(
                """
                SELECT id, file, type, name, code, language, line_start, line_end,
                       docstring, signature
                FROM code_elements
                LIMIT ? OFFSET ?
                """,
                (batch_size, offset),
            )

            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                yield CodeElement(
                    id=row["id"],
                    file=Path(row["file"]),
                    type=row["type"],
                    name=row["name"],
                    code=row["code"],
                    language=row["language"],
                    line_start=row["line_start"],
                    line_end=row["line_end"],
                    docstring=row["docstring"],
                    signature=row["signature"],
                )

            offset += batch_size

    def delete_elements_for_file(self, file_path: Path) -> int:
        """Delete all elements from a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            Number of deleted elements.
        """
        cursor = self.conn.execute(
            "DELETE FROM code_elements WHERE file = ?",
            (str(file_path),),
        )
        self.conn.commit()
        return cursor.rowcount

    # -------------------------------------------------------------------------
    # Statistics Operations
    # -------------------------------------------------------------------------

    def update_stats(self, files: int, elements: int) -> None:
        """Update index statistics.

        Args:
            files: Total files indexed.
            elements: Total code elements.
        """
        self.conn.execute(
            """
            UPDATE index_stats
            SET total_files = ?, total_elements = ?, last_indexed = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (files, elements),
        )
        self.conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with stats.
        """
        cursor = self.conn.execute(
            """
            SELECT total_files, total_elements, last_indexed
            FROM index_stats WHERE id = 1
            """
        )

        row = cursor.fetchone()
        return {
            "files": row["total_files"],
            "elements": row["total_elements"],
            "last_indexed": row["last_indexed"],
        }

    def get_language_stats(self) -> Dict[str, Dict[str, int]]:
        """Get element counts grouped by language.

        Returns:
            Dictionary mapping language -> {"files": count, "elements": count}
        """
        # Get element counts per language
        element_cursor = self.conn.execute(
            """
            SELECT language, COUNT(*) as element_count
            FROM code_elements
            GROUP BY language
            ORDER BY element_count DESC
            """
        )

        # Get file counts per language
        file_cursor = self.conn.execute(
            """
            SELECT language, COUNT(DISTINCT file) as file_count
            FROM code_elements
            GROUP BY language
            ORDER BY file_count DESC
            """
        )

        element_rows = {row["language"]: row["element_count"] for row in element_cursor}
        file_rows = {row["language"]: row["file_count"] for row in file_cursor}

        # Combine into result
        result = {}
        all_langs = set(element_rows.keys()) | set(file_rows.keys())

        for lang in all_langs:
            result[lang] = {
                "files": file_rows.get(lang, 0),
                "elements": element_rows.get(lang, 0),
            }

        return result

    # -------------------------------------------------------------------------
    # File Tracking (Incremental Indexing)
    # -------------------------------------------------------------------------

    def set_file_hash(self, file_path: Path, file_hash: str) -> None:
        """Store file hash for incremental indexing.

        Args:
            file_path: Path to file.
            file_hash: Hash of file contents.
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO indexed_files (file_path, file_hash, indexed_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (str(file_path), file_hash),
        )
        self.conn.commit()

    def get_file_hash(self, file_path: Path) -> Optional[str]:
        """Get stored file hash.

        Args:
            file_path: Path to file.

        Returns:
            Stored hash or None.
        """
        cursor = self.conn.execute(
            "SELECT file_hash FROM indexed_files WHERE file_path = ?",
            (str(file_path),),
        )

        row = cursor.fetchone()
        return row["file_hash"] if row else None

    # -------------------------------------------------------------------------
    # Maintenance Operations
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all data from the database."""
        with self.transaction():
            self.conn.execute("DELETE FROM code_elements")
            self.conn.execute("DELETE FROM patterns")
            self.conn.execute("DELETE FROM indexed_files")
            self.conn.execute(
                "UPDATE index_stats SET total_files = 0, total_elements = 0"
            )
