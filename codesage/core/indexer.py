"""Repository indexer for code intelligence."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TypedDict
import hashlib


class IndexStats(TypedDict):
    """Statistics from indexing operation."""

    files_scanned: int
    files_indexed: int
    files_skipped: int
    elements_found: int
    nodes_added: int  # Graph nodes
    relationships_added: int  # Graph relationships
    errors: int

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from codesage.utils.config import Config
from codesage.utils.logging import get_logger
from codesage.parsers import ParserRegistry
from codesage.llm.embeddings import EmbeddingService
from codesage.storage.manager import StorageManager
from codesage.models.code_element import CodeElement
from codesage.memory.hooks import MemoryHooks
from codesage.core.relationship_extractor import extract_relationships_from_file

logger = get_logger("indexer")


class Indexer:
    """Indexes repository files and extracts code elements.

    Walks the repository, parses code files, generates embeddings,
    and stores everything in SQLite, LanceDB, and KuzuDB.
    """

    def __init__(self, config: Config):
        """Initialize the indexer.

        Args:
            config: CodeSage configuration
        """
        self.config = config

        # Initialize embedding service
        self.embedder = EmbeddingService(config.llm, config.cache_dir)

        # Initialize unified storage manager
        self.storage = StorageManager(
            config=config,
            embedding_fn=self.embedder.embedder,
        )

        # Legacy compatibility
        self.db = self.storage.db
        self.vector_store = self.storage.vector_store

        # Initialize memory hooks for pattern learning
        self._memory_hooks: Optional[MemoryHooks] = None
        if config.memory.enabled and config.memory.learn_on_index:
            self._memory_hooks = MemoryHooks(
                embedding_fn=self.embedder.embedder,
                enabled=True,
            )
            logger.debug("Memory learning enabled for indexer")

        self.stats: IndexStats = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "elements_found": 0,
            "nodes_added": 0,
            "relationships_added": 0,
            "errors": 0,
        }

    def walk_repository(self) -> Iterator[Path]:
        """Walk repository and yield code files.

        Yields:
            Paths to code files that should be indexed
        """
        root = self.config.project_path
        root_resolved = root.resolve()

        for path in root.rglob("*"):
            # Skip directories
            if not path.is_file():
                continue

            # Prevent symlink traversal outside project root
            try:
                resolved = path.resolve()
                if not str(resolved).startswith(str(root_resolved)):
                    continue  # Skip files outside project root (symlink escape)
            except (OSError, ValueError):
                continue

            # Skip excluded directories
            if any(excluded in path.parts for excluded in self.config.exclude_dirs):
                continue

            # Only process files with supported extensions
            if path.suffix.lower() in self.config.include_extensions:
                self.stats["files_scanned"] += 1
                yield path

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file contents for change detection.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash of file contents
        """
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""

    def _should_reindex(self, file_path: Path) -> bool:
        """Check if file needs re-indexing.

        Args:
            file_path: Path to check

        Returns:
            True if file should be re-indexed
        """
        current_hash = self._compute_file_hash(file_path)
        stored_hash = self.db.get_file_hash(file_path)

        return current_hash != stored_hash

    def index_file(self, file_path: Path, force: bool = False) -> int:
        """Index a single file.

        Args:
            file_path: Path to file
            force: Force re-indexing even if unchanged

        Returns:
            Number of elements indexed
        """
        # Check if re-indexing is needed
        if not force and not self._should_reindex(file_path):
            self.stats["files_skipped"] += 1
            return 0

        # Get parser for file type
        parser = ParserRegistry.get_parser_for_file(file_path)
        if not parser:
            return 0

        try:
            # Parse file
            elements = parser.parse_file(file_path)

            if not elements:
                return 0

            # Clear old data for this file (all backends)
            self.storage.delete_by_file(file_path)

            # Store elements in all backends (SQLite, LanceDB, KuzuDB)
            self.storage.add_elements(elements)

            # Extract and store relationships (calls, imports, inheritance)
            if self.config.storage.use_graph:
                try:
                    nodes, relationships = extract_relationships_from_file(
                        file_path, elements
                    )
                    # Add file/module nodes
                    if nodes and self.storage.graph_store:
                        self.storage.graph_store.add_nodes(nodes)
                        self.stats["nodes_added"] += len(nodes)

                    # Add relationships
                    if relationships:
                        self.storage.add_relationships(relationships)
                        self.stats["relationships_added"] += len(relationships)

                except Exception as e:
                    logger.warning(f"Error extracting relationships from {file_path}: {e}")

            # Update file hash
            self.db.set_file_hash(file_path, self._compute_file_hash(file_path))

            # Learn patterns from indexed elements
            if self._memory_hooks:
                element_dicts = [el.to_dict() for el in elements]
                self._memory_hooks.on_elements_indexed(
                    element_dicts,
                    self.config.project_name,
                    file_path,
                )

            self.stats["files_indexed"] += 1
            self.stats["elements_found"] += len(elements)
            self.stats["nodes_added"] += len(elements)

            return len(elements)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error indexing {file_path}: {e}")
            return 0

    def index_repository(
        self,
        incremental: bool = True,
        show_progress: bool = True,
    ) -> IndexStats:
        """Index the entire repository.

        Args:
            incremental: Only index changed files
            show_progress: Show progress bar

        Returns:
            Dictionary with indexing statistics
        """
        # Reset stats
        self.stats: IndexStats = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "elements_found": 0,
            "nodes_added": 0,
            "relationships_added": 0,
            "errors": 0,
        }

        # Collect files first to show accurate progress
        files = list(self.walk_repository())

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("Indexing...", total=len(files))

                for file_path in files:
                    progress.update(
                        task,
                        description=f"[cyan]{file_path.name}",
                        advance=1,
                    )

                    self.index_file(file_path, force=not incremental)

                progress.update(task, description="[green]✓ Complete")
        else:
            for file_path in files:
                self.index_file(file_path, force=not incremental)

        # Update database stats
        self.db.update_stats(
            self.stats["files_indexed"],
            self.stats["elements_found"],
        )

        # Notify memory system that project indexing is complete
        if self._memory_hooks:
            self._memory_hooks.on_project_indexed(
                project_name=self.config.project_name,
                project_path=self.config.project_path,
                total_files=self.stats["files_indexed"],
                total_elements=self.stats["elements_found"],
            )
            logger.info(
                f"Memory learning complete: {self._memory_hooks.get_stats()}"
            )

        return self.stats

    def clear_index(self) -> None:
        """Clear all indexed data from all backends."""
        self.storage.clear()

        self.stats: IndexStats = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "elements_found": 0,
            "nodes_added": 0,
            "relationships_added": 0,
            "errors": 0,
        }

    def set_memory_learning(self, enabled: bool) -> None:
        """Enable or disable memory learning.

        Args:
            enabled: Whether to enable learning
        """
        if self._memory_hooks:
            self._memory_hooks.enabled = enabled
        elif enabled and self.config.memory.enabled:
            # Create hooks if they don't exist and learning is requested
            self._memory_hooks = MemoryHooks(
                embedding_fn=self.embedder.embedder,
                enabled=True,
            )

    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Get memory learning statistics.

        Returns:
            Dictionary with memory stats, or None if not enabled
        """
        if self._memory_hooks:
            return self._memory_hooks.get_stats()
        return None

