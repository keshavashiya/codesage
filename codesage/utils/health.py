"""Health check system for CodeSage.

Provides health checks for all system dependencies:
- Ollama LLM service
- SQLite database
- ChromaDB vector store
- Disk space
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import time


@dataclass
class HealthStatus:
    """Overall system health status."""

    ollama_available: bool = False
    database_accessible: bool = False
    vector_store_accessible: bool = False
    disk_space_ok: bool = False

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    ollama_latency_ms: Optional[float] = None
    database_size_mb: Optional[float] = None
    vector_count: Optional[int] = None

    @property
    def is_healthy(self) -> bool:
        """Check if all critical components are healthy."""
        return all([
            self.ollama_available,
            self.database_accessible,
            self.vector_store_accessible,
        ])

    @property
    def is_ready(self) -> bool:
        """Check if system is ready for queries."""
        return self.is_healthy and self.disk_space_ok

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "healthy": self.is_healthy,
            "ready": self.is_ready,
            "components": {
                "ollama": {
                    "status": "ok" if self.ollama_available else "error",
                    "latency_ms": self.ollama_latency_ms,
                },
                "database": {
                    "status": "ok" if self.database_accessible else "error",
                    "size_mb": self.database_size_mb,
                },
                "vector_store": {
                    "status": "ok" if self.vector_store_accessible else "error",
                    "document_count": self.vector_count,
                },
                "disk": {
                    "status": "ok" if self.disk_space_ok else "warning",
                },
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


def check_ollama(base_url: str, timeout: float = 5.0) -> Tuple[bool, Optional[str], Optional[float]]:
    """Check if Ollama is running and responsive.

    Args:
        base_url: Ollama base URL
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (is_available, error_message, latency_ms)
    """
    try:
        import urllib.request
        import urllib.error

        start = time.perf_counter()
        url = f"{base_url}/api/version"

        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            latency = (time.perf_counter() - start) * 1000
            if response.status == 200:
                return True, None, latency
            return False, f"Ollama returned status {response.status}", latency

    except urllib.error.URLError as e:
        return False, f"Cannot connect to Ollama at {base_url}: {e.reason}", None
    except TimeoutError:
        return False, f"Ollama connection timed out after {timeout}s", None
    except Exception as e:
        return False, f"Ollama check failed: {e}", None


def check_database(db_path: Path) -> Tuple[bool, Optional[str], Optional[float]]:
    """Check if SQLite database is accessible.

    Args:
        db_path: Path to SQLite database

    Returns:
        Tuple of (is_accessible, error_message, size_mb)
    """
    try:
        if not db_path.exists():
            return False, f"Database not found: {db_path}", None

        size_mb = db_path.stat().st_size / (1024 * 1024)

        # Try to open and query
        import sqlite3
        conn = sqlite3.connect(str(db_path), timeout=5)
        cursor = conn.execute("SELECT COUNT(*) FROM code_elements")
        cursor.fetchone()
        conn.close()

        return True, None, size_mb

    except sqlite3.Error as e:
        return False, f"Database error: {e}", None
    except Exception as e:
        return False, f"Database check failed: {e}", None


def check_vector_store(chroma_path: Path) -> Tuple[bool, Optional[str], Optional[int]]:
    """Check if ChromaDB vector store is accessible.

    Args:
        chroma_path: Path to ChromaDB directory

    Returns:
        Tuple of (is_accessible, error_message, document_count)
    """
    try:
        if not chroma_path.exists():
            return False, f"Vector store not found: {chroma_path}", None

        # Check for ChromaDB files
        chroma_files = list(chroma_path.glob("**/*"))
        if not chroma_files:
            return False, "Vector store is empty", 0

        # Try to get document count (basic check without loading full ChromaDB)
        # This is a lightweight check - full validation requires loading the store
        sqlite_file = chroma_path / "chroma.sqlite3"
        if sqlite_file.exists():
            import sqlite3
            conn = sqlite3.connect(str(sqlite_file), timeout=5)
            try:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM embeddings"
                )
                count = cursor.fetchone()[0]
                conn.close()
                return True, None, count
            except sqlite3.Error:
                conn.close()
                # Table might not exist yet, that's ok
                return True, None, 0

        return True, None, None

    except Exception as e:
        return False, f"Vector store check failed: {e}", None


def check_disk_space(path: Path, min_free_gb: float = 1.0) -> Tuple[bool, Optional[str]]:
    """Check if there's enough disk space.

    Args:
        path: Path to check
        min_free_gb: Minimum free space in GB

    Returns:
        Tuple of (is_ok, warning_message)
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)

        if free_gb < min_free_gb:
            return False, f"Low disk space: {free_gb:.1f}GB free (need {min_free_gb}GB)"

        return True, None

    except Exception as e:
        return False, f"Disk space check failed: {e}"


def check_system_health(config) -> HealthStatus:
    """Perform full system health check.

    Args:
        config: CodeSage configuration

    Returns:
        HealthStatus with all component statuses
    """
    status = HealthStatus()

    # Check Ollama
    ollama_ok, ollama_err, ollama_latency = check_ollama(
        config.llm.base_url or "http://localhost:11434"
    )
    status.ollama_available = ollama_ok
    status.ollama_latency_ms = ollama_latency
    if ollama_err:
        status.errors.append(ollama_err)

    # Check Database
    if config.storage.db_path:
        db_ok, db_err, db_size = check_database(config.storage.db_path)
        status.database_accessible = db_ok
        status.database_size_mb = db_size
        if db_err:
            status.errors.append(db_err)
    else:
        status.errors.append("Database path not configured")

    # Check Vector Store
    if config.storage.chroma_path:
        vs_ok, vs_err, vs_count = check_vector_store(config.storage.chroma_path)
        status.vector_store_accessible = vs_ok
        status.vector_count = vs_count
        if vs_err:
            status.errors.append(vs_err)
    else:
        status.errors.append("Vector store path not configured")

    # Check Disk Space
    disk_ok, disk_warning = check_disk_space(config.project_path)
    status.disk_space_ok = disk_ok
    if disk_warning:
        status.warnings.append(disk_warning)

    return status
