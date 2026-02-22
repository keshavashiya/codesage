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
from codesage.core.relationship_extractor import (
    extract_relationships_from_file,
    extract_cross_file_calls,
)
from codesage.utils.language_detector import EXTENSION_TO_LANGUAGE

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
        self.embedder = EmbeddingService(
            config.llm,
            config.cache_dir,
            config.performance,
        )

        # Detect embedding dimension from model
        self._vector_dim = self.embedder.get_dimension()

        # Initialize unified storage manager
        self.storage = StorageManager(
            config=config,
            embedding_fn=self.embedder,
            vector_dim=self._vector_dim,
        )

        # Legacy compatibility
        self.db = self.storage.db
        self.vector_store = self.storage.vector_store

        # Initialize memory hooks for pattern learning
        self._memory_hooks: Optional[MemoryHooks] = None
        if config.memory.enabled and config.memory.learn_on_index:
            self._memory_hooks = MemoryHooks(
                embedding_fn=self.embedder.embed_batch,
                enabled=True,
                vector_dim=self._vector_dim,
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

            # Learn patterns from indexed elements (language-aware)
            if self._memory_hooks:
                element_dicts = [el.to_dict() for el in elements]
                file_language = EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), "python")
                self._memory_hooks.on_elements_indexed(
                    element_dicts,
                    self.config.project_name,
                    file_path,
                    language=file_language,
                )

            self.stats["files_indexed"] += 1
            self.stats["elements_found"] += len(elements)
            self.stats["nodes_added"] += len(elements)

            return len(elements)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error indexing {file_path}: {e}")
            return 0

    # Sub-batch size for embedding generation (smaller = more frequent progress updates)
    EMBEDDING_SUB_BATCH = 32

    def index_repository(
        self,
        incremental: bool = True,
        show_progress: bool = True,
    ) -> IndexStats:
        """Index the entire repository.

        Two-phase approach for better UX:
        1. Parse files and store metadata/graph (fast, with file progress bar)
        2. Generate embeddings in sub-batches (slower, with element progress bar)

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
        # Collect ALL elements for deferred embedding (no mid-parse flushes)
        self._batch_elements = []
        self._batch_file_count = 0
        self._defer_embeddings = True

        # Collect files first to show accurate progress
        files = list(self.walk_repository())

        # Phase 1: Parse files and store metadata/graph (fast)
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("Parsing files...", total=len(files))

                for file_path in files:
                    progress.update(
                        task,
                        description=f"[cyan]Parsing {file_path.name}",
                        advance=1,
                    )

                    self._index_file_for_batch(
                        file_path,
                        force=not incremental,
                    )

                progress.update(task, description="[green]✓ Parsing complete")
        else:
            for file_path in files:
                self._index_file_for_batch(
                    file_path,
                    force=not incremental,
                )

        # Phase 2: Generate embeddings with progress
        self._flush_batch_embeddings(show_progress=show_progress)

        # Phase 3: Cross-file CALLS pass (Python only)
        # After all elements are stored, we have a global name→element map
        # and can find calls that cross file boundaries.
        if self.config.storage.use_graph and self.storage.graph_store:
            self._run_cross_file_calls_pass()

        # Update database stats from actual DB counts (not run delta)
        if self.stats["files_indexed"] > 0:
            total_files = self.db.conn.execute(
                "SELECT COUNT(*) FROM indexed_files"
            ).fetchone()[0]
            total_elements = self.db.conn.execute(
                "SELECT COUNT(*) FROM code_elements"
            ).fetchone()[0]
            self.db.update_stats(total_files, total_elements)

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

    def _index_file_for_batch(self, file_path: Path, force: bool = False) -> int:
        """Index a file and queue elements for batched embedding storage.

        This stores metadata and graph data immediately, while deferring
        vector embeddings to a batch call.
        """
        if not hasattr(self, "_batch_elements"):
            self._batch_elements: List[CodeElement] = []
            self._batch_file_count = 0

        # Check if re-indexing is needed
        if not force and not self._should_reindex(file_path):
            self.stats["files_skipped"] += 1
            return 0

        # Get parser for file type
        parser = ParserRegistry.get_parser_for_file(file_path)
        if not parser:
            return 0

        try:
            elements = parser.parse_file(file_path)
            if not elements:
                return 0

            # Clear old data for this file
            self.storage.delete_by_file(file_path)

            # Store metadata immediately (SQLite)
            self.db.store_elements(elements)

            # Add element nodes to graph immediately for relationship inserts
            if self.storage.graph_store:
                self.storage._add_to_graph(elements)

            # Extract and store relationships (calls, imports, inheritance)
            if self.config.storage.use_graph:
                try:
                    nodes, relationships = extract_relationships_from_file(
                        file_path, elements
                    )
                    if nodes and self.storage.graph_store:
                        self.storage.graph_store.add_nodes(nodes)
                        self.stats["nodes_added"] += len(nodes)

                    if relationships:
                        self.storage.add_relationships(relationships)
                        self.stats["relationships_added"] += len(relationships)
                except Exception as e:
                    logger.warning(f"Error extracting relationships from {file_path}: {e}")

            # Update file hash
            self.db.set_file_hash(file_path, self._compute_file_hash(file_path))

            # Learn patterns from indexed elements (language-aware)
            if self._memory_hooks:
                element_dicts = [el.to_dict() for el in elements]
                file_language = EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), "python")
                self._memory_hooks.on_elements_indexed(
                    element_dicts,
                    self.config.project_name,
                    file_path,
                    language=file_language,
                )

            self.stats["files_indexed"] += 1
            self.stats["elements_found"] += len(elements)
            self.stats["nodes_added"] += len(elements)

            # Queue for batched embedding insertion
            self._batch_elements.extend(elements)
            self._batch_file_count += 1

            # Only flush mid-parse when not deferring (non-progress mode)
            if not getattr(self, "_defer_embeddings", False):
                max_files = self.config.performance.embedding_batch_size
                max_elements = self.config.performance.max_elements_per_batch

                if (
                    self._batch_file_count >= max_files
                    or len(self._batch_elements) >= max_elements
                ):
                    self._flush_batch_embeddings()

            return len(elements)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error indexing {file_path}: {e}")
            return 0

    def _flush_batch_embeddings(self, show_progress: bool = False) -> None:
        """Flush queued elements to the vector store with sub-batching.

        Args:
            show_progress: Show a progress bar for embedding generation.
        """
        if not hasattr(self, "_batch_elements"):
            return
        if not self._batch_elements:
            return
        if not self.storage.vector_store:
            self._batch_elements = []
            self._batch_file_count = 0
            return

        total = len(self._batch_elements)
        batch_size = self.EMBEDDING_SUB_BATCH

        if show_progress and total > batch_size:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[dim]{task.completed}/{task.total} elements[/dim]"),
            ) as progress:
                task = progress.add_task(
                    "[cyan]Generating embeddings...", total=total
                )

                for i in range(0, total, batch_size):
                    sub_batch = self._batch_elements[i : i + batch_size]
                    self.storage.vector_store.add_elements(sub_batch)
                    progress.update(task, advance=len(sub_batch))

                progress.update(
                    task, description="[green]✓ Embeddings complete"
                )
        else:
            # Small batch or no progress — flush in one go
            self.storage.vector_store.add_elements(self._batch_elements)

        self._batch_elements = []
        self._batch_file_count = 0
        self._defer_embeddings = False

    def _run_cross_file_calls_pass(self) -> None:
        """Third indexing phase: extract cross-file CALLS relationships.

        Queries all elements from SQLite, builds a global name→element map,
        then re-parses every Python file to find calls where the callee is
        defined in a different file.  Same-file edges were already stored in
        the first pass; KuzuDB MERGE semantics prevent duplicates.
        """
        try:
            # Build global element map from SQLite (include line numbers for cross-file callers)
            rows = self.db.conn.execute(
                "SELECT id, name, file, type, line_start, line_end "
                "FROM code_elements WHERE type IN ('function','method','class')"
            ).fetchall()

            if not rows:
                return

            global_elements: Dict[str, Any] = {}
            file_to_elements: Dict[str, List[Any]] = {}
            for row_id, name, fpath, etype, line_start, line_end in rows:
                # Build a lightweight stub with all fields the extractors need
                stub = type(
                    "Stub",
                    (),
                    {
                        "id": row_id,
                        "name": name,
                        "file": fpath,
                        "type": etype,
                        "line_start": line_start or 1,
                        "line_end": line_end or 1,
                    },
                )()
                if name not in global_elements:  # first occurrence wins
                    global_elements[name] = stub
                file_to_elements.setdefault(fpath, []).append(stub)

            cross_file_total = 0
            for fpath_str, file_stubs in file_to_elements.items():
                fpath = Path(fpath_str)
                if not fpath.exists():
                    continue
                try:
                    new_rels = extract_cross_file_calls(fpath, file_stubs, global_elements)  # type: ignore[arg-type]
                    if new_rels:
                        self.storage.add_relationships(new_rels)
                        cross_file_total += len(new_rels)
                except Exception as e:
                    logger.warning(f"Cross-file call extraction failed for {fpath}: {e}")

            if cross_file_total:
                logger.info(f"Cross-file pass: added {cross_file_total} CALLS edges")
                self.stats["relationships_added"] += cross_file_total

        except Exception as e:
            logger.warning(f"Cross-file calls pass failed: {e}")

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
                embedding_fn=self.embedder.embed_batch,
                enabled=True,
                vector_dim=self._vector_dim,
            )

    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Get memory learning statistics.

        Returns:
            Dictionary with memory stats, or None if not enabled
        """
        if self._memory_hooks:
            return self._memory_hooks.get_stats()
        return None
