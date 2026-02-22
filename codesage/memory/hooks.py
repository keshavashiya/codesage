"""Integration hooks for the memory system.

Provides hooks that can be called from the indexer and other components
to automatically learn patterns from code as it's indexed.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from codesage.utils.logging import get_logger

from .learning_engine import LearningEngine
from .memory_manager import MemoryManager
from .models import LearnedPattern, ProjectInfo

logger = get_logger("memory.hooks")


class MemoryHooks:
    """Hooks for integrating memory learning with the indexer.

    Provides callbacks that can be triggered during indexing to:
        - Learn patterns from indexed elements
        - Track project statistics
        - Record interactions

    Usage:
        hooks = MemoryHooks()
        hooks.on_elements_indexed(elements, project_name, file_path)
    """

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        enabled: bool = True,
        vector_dim: int = 0,
    ) -> None:
        """Initialize memory hooks.

        Args:
            memory_manager: Optional existing MemoryManager.
            embedding_fn: Optional embedding function.
            enabled: Whether learning is enabled.
            vector_dim: Embedding vector dimension (0 = use store default).
        """
        self._enabled = enabled
        self._memory: Optional[MemoryManager] = memory_manager
        self._engine: Optional[LearningEngine] = None
        self._embedding_fn = embedding_fn
        self._vector_dim = vector_dim

        # Statistics
        self._elements_processed = 0
        self._patterns_learned = 0
        self._files_processed = 0

    @property
    def enabled(self) -> bool:
        """Check if hooks are enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable hooks."""
        self._enabled = value

    def _ensure_initialized(self) -> None:
        """Ensure memory manager and engine are initialized."""
        if self._memory is None:
            self._memory = MemoryManager(
                embedding_fn=self._embedding_fn,
                vector_dim=self._vector_dim,
            )

        if self._engine is None:
            self._engine = LearningEngine(
                self._memory,
                embedding_fn=self._embedding_fn,
            )

    def set_embedding_fn(
        self,
        embedding_fn: Callable[[List[str]], List[List[float]]],
    ) -> None:
        """Set the embedding function.

        Args:
            embedding_fn: Function to generate embeddings.
        """
        self._embedding_fn = embedding_fn

        if self._memory:
            self._memory.set_embedding_fn(embedding_fn)

    def on_elements_indexed(
        self,
        elements: List[Dict[str, Any]],
        project_name: str,
        file_path: Optional[Path] = None,
        language: Optional[str] = None,
    ) -> List[LearnedPattern]:
        """Called when code elements are indexed.

        Analyzes elements for patterns and stores them in memory.

        Args:
            elements: List of code element dictionaries.
            project_name: Name of the project.
            file_path: Optional path to the source file.
            language: Primary programming language of the elements (default: auto-detected).

        Returns:
            List of learned patterns.
        """
        if not self._enabled or not elements:
            return []

        self._ensure_initialized()

        # Determine language dynamically
        language = language or (elements[0].get("language", "python") if elements else "python")

        try:
            patterns = self._engine.learn_from_elements(
                elements,
                project_name,
                file_path.parent if file_path else None,
                language=language,
            )

            self._elements_processed += len(elements)
            self._patterns_learned += len(patterns)
            self._files_processed += 1

            logger.debug(
                f"Processed {len(elements)} elements, learned {len(patterns)} patterns"
            )

            return patterns

        except Exception as e:
            logger.warning(f"Failed to learn from elements: {e}")
            return []

    def on_file_indexed(
        self,
        file_path: Path,
        project_name: str,
        element_count: int,
    ) -> None:
        """Called when a file is indexed.

        Updates project statistics in memory.

        Args:
            file_path: Path to the indexed file.
            project_name: Project name.
            element_count: Number of elements indexed.
        """
        if not self._enabled:
            return

        self._ensure_initialized()

        try:
            # Update project stats
            project = self._memory.get_project(project_name)

            if project:
                self._memory.preference_store.update_project_stats(
                    project_name,
                    total_elements=(project.total_elements or 0) + element_count,
                    total_files=(project.total_files or 0) + 1,
                )
        except Exception as e:
            logger.debug(f"Failed to update project stats: {e}")

    def on_project_indexed(
        self,
        project_name: str,
        project_path: Path,
        total_files: int,
        total_elements: int,
    ) -> None:
        """Called when a project is fully indexed.

        Records project information and computes similarities.

        Args:
            project_name: Project name.
            project_path: Path to the project.
            total_files: Total files indexed.
            total_elements: Total elements indexed.
        """
        if not self._enabled:
            return

        self._ensure_initialized()

        try:
            # Get existing project or create new
            project = self._memory.get_project(project_name)

            if project:
                # Update stats
                self._memory.preference_store.update_project_stats(
                    project_name,
                    total_files=total_files,
                    total_elements=total_elements,
                )
            else:
                # Create new project
                project = ProjectInfo.create(
                    name=project_name,
                    path=project_path,
                    total_files=total_files,
                    total_elements=total_elements,
                )
                self._memory.add_project(project)

            # Compute similarities with other projects
            self._engine.update_all_project_similarities()

            logger.info(
                f"Project {project_name} indexed: {total_files} files, "
                f"{total_elements} elements"
            )

        except Exception as e:
            logger.debug(f"Failed to record project: {e}")

    def on_query(
        self,
        query: str,
        project_name: str,
        results: List[Any],
    ) -> None:
        """Called when a query is made.

        Records the interaction for learning.

        Args:
            query: The search query.
            project_name: Project name.
            results: Query results.
        """
        if not self._enabled:
            return

        self._ensure_initialized()

        try:
            self._memory.record_interaction(
                interaction_type="query",
                project_name=project_name,
                query=query,
                response=f"Found {len(results)} results",
                metadata={"result_count": len(results)},
            )
        except Exception as e:
            logger.debug(f"Failed to record query interaction: {e}")

    def on_suggestion(
        self,
        query: str,
        project_name: str,
        suggestion: str,
        accepted: Optional[bool] = None,
    ) -> None:
        """Called when a suggestion is made.

        Records the suggestion interaction.

        Args:
            query: The input query.
            project_name: Project name.
            suggestion: The generated suggestion.
            accepted: Whether the suggestion was accepted.
        """
        if not self._enabled:
            return

        self._ensure_initialized()

        try:
            self._memory.record_interaction(
                interaction_type="suggestion",
                project_name=project_name,
                query=query,
                response=suggestion[:500],  # Truncate long suggestions
                accepted=accepted,
            )
        except Exception as e:
            logger.debug(f"Failed to record suggestion interaction: {e}")

    def on_suggestion_feedback(
        self,
        interaction_id: str,
        accepted: bool,
        feedback: str = "",
    ) -> None:
        """Called when feedback is given on a suggestion.

        Updates the interaction record with feedback.

        Args:
            interaction_id: ID of the original interaction.
            accepted: Whether the suggestion was accepted.
            feedback: Optional feedback text.
        """
        if not self._enabled:
            return

        self._ensure_initialized()

        # Note: This would require updating the interaction record
        # For now, we just log it
        logger.debug(
            f"Suggestion {interaction_id} feedback: accepted={accepted}, "
            f"feedback={feedback}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get hook statistics.

        Returns:
            Dictionary with statistics.
        """
        stats = {
            "enabled": self._enabled,
            "elements_processed": self._elements_processed,
            "patterns_learned": self._patterns_learned,
            "files_processed": self._files_processed,
        }

        if self._memory:
            try:
                memory_metrics = self._memory.get_metrics()
                stats["memory_metrics"] = memory_metrics
            except Exception:
                pass

        return stats

    def reset_stats(self) -> None:
        """Reset hook statistics."""
        self._elements_processed = 0
        self._patterns_learned = 0
        self._files_processed = 0


# Global hooks instance for easy access
_global_hooks: Optional[MemoryHooks] = None


def get_memory_hooks() -> MemoryHooks:
    """Get the global memory hooks instance.

    Returns:
        MemoryHooks instance.
    """
    global _global_hooks
    if _global_hooks is None:
        _global_hooks = MemoryHooks()
    return _global_hooks


def set_memory_hooks(hooks: MemoryHooks) -> None:
    """Set the global memory hooks instance.

    Args:
        hooks: MemoryHooks instance to use.
    """
    global _global_hooks
    _global_hooks = hooks


def enable_memory_learning(enabled: bool = True) -> None:
    """Enable or disable memory learning globally.

    Args:
        enabled: Whether to enable learning.
    """
    hooks = get_memory_hooks()
    hooks.enabled = enabled
