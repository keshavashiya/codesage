"""SQLite database for storing code metadata."""

import os
import stat
import sqlite3
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
from contextlib import contextmanager

from codesage.models.code_element import CodeElement
from codesage.utils.logging import get_logger

logger = get_logger("storage.database")


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class Database:
    """SQLite database for code element metadata.

    Stores metadata about indexed code elements, patterns,
    and indexing statistics.

    Security features:
        - File permissions hardened to owner-only (600)
        - Parameterized queries throughout
        - Transaction support for data integrity
    """

    # File permissions: owner read/write only (rw-------)
    FILE_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR  # 0o600

    def __init__(self, db_path: Path):
        """Initialize the database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()
        self._harden_permissions()

    def _harden_permissions(self) -> None:
        """Harden file permissions for security.

        Sets database file to owner read/write only (600).
        """
        try:
            if self.db_path.exists():
                os.chmod(str(self.db_path), self.FILE_PERMISSIONS)
                logger.debug(f"Set permissions 600 on {self.db_path}")
        except OSError as e:
            # Log warning but don't fail - permissions might be restricted
            logger.warning(f"Could not set file permissions on {self.db_path}: {e}")

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,  # Add connection timeout
                check_same_thread=False,  # Allow multi-threaded access
            )
            self._conn.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode for better performance
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Transaction failed, rolled back: {e}")
            raise DatabaseError(f"Transaction failed: {e}") from e

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

    def store_element(self, element: CodeElement) -> None:
        """Store a code element.

        Args:
            element: Code element to store
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO code_elements
            (id, file, type, name, code, language, line_start, line_end, docstring, signature, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
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
        ))
        self.conn.commit()

    def store_elements(self, elements: List[CodeElement]) -> None:
        """Bulk store code elements.

        Args:
            elements: List of code elements to store
        """
        with self.transaction():
            self.conn.executemany("""
                INSERT OR REPLACE INTO code_elements
                (id, file, type, name, code, language, line_start, line_end, docstring, signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                (
                    e.id, str(e.file), e.type, e.name, e.code, e.language,
                    e.line_start, e.line_end, e.docstring, e.signature
                )
                for e in elements
            ])

    def get_element(self, element_id: str) -> Optional[CodeElement]:
        """Retrieve an element by ID.

        Args:
            element_id: Element ID

        Returns:
            CodeElement or None if not found
        """
        cursor = self.conn.execute("""
            SELECT id, file, type, name, code, language, line_start, line_end, docstring, signature
            FROM code_elements WHERE id = ?
        """, (element_id,))

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
            batch_size: Number of elements to fetch per batch

        Yields:
            CodeElement instances
        """
        offset = 0

        while True:
            cursor = self.conn.execute("""
                SELECT id, file, type, name, code, language, line_start, line_end, docstring, signature
                FROM code_elements
                LIMIT ? OFFSET ?
            """, (batch_size, offset))

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
            file_path: Path to the file

        Returns:
            Number of deleted elements
        """
        cursor = self.conn.execute("""
            DELETE FROM code_elements WHERE file = ?
        """, (str(file_path),))
        self.conn.commit()
        return cursor.rowcount

    def update_stats(self, files: int, elements: int) -> None:
        """Update index statistics.

        Args:
            files: Total files indexed
            elements: Total code elements
        """
        self.conn.execute("""
            UPDATE index_stats
            SET total_files = ?, total_elements = ?, last_indexed = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (files, elements))
        self.conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with stats
        """
        cursor = self.conn.execute("""
            SELECT total_files, total_elements, last_indexed
            FROM index_stats WHERE id = 1
        """)

        row = cursor.fetchone()
        return {
            "files": row["total_files"],
            "elements": row["total_elements"],
            "last_indexed": row["last_indexed"],
        }

    def set_file_hash(self, file_path: Path, file_hash: str) -> None:
        """Store file hash for incremental indexing.

        Args:
            file_path: Path to file
            file_hash: Hash of file contents
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO indexed_files (file_path, file_hash, indexed_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (str(file_path), file_hash))
        self.conn.commit()

    def get_file_hash(self, file_path: Path) -> Optional[str]:
        """Get stored file hash.

        Args:
            file_path: Path to file

        Returns:
            Stored hash or None
        """
        cursor = self.conn.execute("""
            SELECT file_hash FROM indexed_files WHERE file_path = ?
        """, (str(file_path),))

        row = cursor.fetchone()
        return row["file_hash"] if row else None

    def clear(self) -> None:
        """Clear all data from the database."""
        with self.transaction():
            self.conn.execute("DELETE FROM code_elements")
            self.conn.execute("DELETE FROM patterns")
            self.conn.execute("DELETE FROM indexed_files")
            self.conn.execute("""
                UPDATE index_stats SET total_files = 0, total_elements = 0
            """)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
