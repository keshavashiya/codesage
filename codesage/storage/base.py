"""Base storage utilities and exceptions."""

import os
import stat
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from codesage.utils.logging import get_logger

logger = get_logger("storage.base")


class DatabaseError(Exception):
    """Base exception for database errors."""

    pass


class ConnectionManager:
    """Manages SQLite database connections with security hardening.

    Provides:
        - Lazy connection initialization
        - File permission hardening (600)
        - WAL mode for better performance
        - Transaction support
    """

    # File permissions: owner read/write only (rw-------)
    FILE_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR  # 0o600

    def __init__(self, db_path: Path) -> None:
        """Initialize the connection manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def _harden_permissions(self) -> None:
        """Harden file permissions for security.

        Sets database file to owner read/write only (600).
        """
        try:
            if self.db_path.exists():
                os.chmod(str(self.db_path), self.FILE_PERMISSIONS)
                logger.debug(f"Set permissions 600 on {self.db_path}")
        except OSError as e:
            logger.warning(f"Could not set file permissions on {self.db_path}: {e}")

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions.

        Yields:
            The database connection within a transaction.

        Raises:
            DatabaseError: If the transaction fails.
        """
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Transaction failed, rolled back: {e}")
            raise DatabaseError(f"Transaction failed: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
