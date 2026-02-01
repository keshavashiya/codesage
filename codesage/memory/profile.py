"""Developer profile management.

Provides high-level interface for managing developer profile,
including preferences, patterns, and style information.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from codesage.utils.logging import get_logger

from .memory_manager import MemoryManager
from .models import (
    LearnedPattern,
    PatternCategory,
    ProjectInfo,
)

logger = get_logger("memory.profile")


class DeveloperProfileManager:
    """Manages developer profile and preferences.

    Provides a high-level interface for:
        - Setting and getting preferences
        - Viewing learned patterns
        - Generating style guides
        - Tracking projects
    """

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        global_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the profile manager.

        Args:
            memory_manager: Optional existing MemoryManager.
            global_dir: Optional custom global directory.
        """
        self._memory = memory_manager or MemoryManager(global_dir=global_dir)

    @property
    def memory(self) -> MemoryManager:
        """Get the underlying memory manager."""
        return self._memory

    # ==================== Preferences ====================

    def set_preference(
        self,
        key: str,
        value: Any,
        description: str = "",
    ) -> None:
        """Set a profile preference.

        Args:
            key: Preference key.
            value: Preference value.
            description: Optional description.
        """
        self._memory.set_preference(key, value, "profile", description)
        logger.info(f"Set preference: {key} = {value}")

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a profile preference.

        Args:
            key: Preference key.
            default: Default value if not set.

        Returns:
            Preference value or default.
        """
        value = self._memory.get_preference(key)
        return value if value is not None else default

    def get_all_preferences(self) -> Dict[str, Any]:
        """Get all profile preferences.

        Returns:
            Dictionary of preferences.
        """
        return self._memory.get_all_preferences()

    def reset_preferences(self) -> None:
        """Reset all preferences to defaults."""
        # Delete all preferences from SQLite
        for pref in self._memory.preference_store.get_all_preferences():
            self._memory.preference_store.delete_preference(pref.key)
        logger.info("Reset all preferences")

    # ==================== Profile Summary ====================

    def get_profile_summary(self) -> Dict[str, Any]:
        """Get a summary of the developer profile.

        Returns:
            Dictionary with profile summary.
        """
        metrics = self._memory.get_metrics()

        # Get pattern counts by category
        pattern_counts = {}
        for category in PatternCategory:
            count = self._memory.preference_store.get_pattern_count(category)
            if count > 0:
                pattern_counts[category.value] = count

        # Get top patterns
        top_patterns = self._memory.preference_store.get_patterns(
            min_confidence=0.7, limit=5
        )

        # Get projects
        projects = self._memory.preference_store.get_all_projects()

        return {
            "total_patterns": metrics.get("preference_store", {}).get("pattern_count", 0),
            "patterns_by_category": pattern_counts,
            "top_patterns": [p.to_dict() for p in top_patterns],
            "total_projects": len(projects),
            "projects": [p.to_dict() for p in projects],
            "interaction_stats": metrics.get("preference_store", {}).get(
                "interaction_stats", {}
            ),
        }

    def get_profile_json(self) -> Dict[str, Any]:
        """Get full profile as JSON-serializable dict.

        Returns:
            Complete profile data.
        """
        summary = self.get_profile_summary()
        preferences = self.get_all_preferences()

        # Get all patterns
        patterns = self._memory.preference_store.get_patterns(limit=1000)

        return {
            "preferences": preferences,
            "summary": summary,
            "patterns": [p.to_dict() for p in patterns],
        }

    # ==================== Pattern Management ====================

    def get_patterns(
        self,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> List[LearnedPattern]:
        """Get learned patterns.

        Args:
            category: Optional category filter (as string).
            min_confidence: Minimum confidence threshold.
            limit: Maximum patterns to return.

        Returns:
            List of patterns.
        """
        cat = PatternCategory(category) if category else None
        return self._memory.preference_store.get_patterns(
            category=cat,
            min_confidence=min_confidence,
            limit=limit,
        )

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a pattern with full context.

        Args:
            pattern_id: Pattern ID.

        Returns:
            Pattern context or None.
        """
        return self._memory.get_pattern_context(pattern_id)

    def search_patterns(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search patterns by query.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching patterns.
        """
        return self._memory.find_similar_patterns(query, limit)

    # ==================== Project Management ====================

    def get_projects(self) -> List[ProjectInfo]:
        """Get all tracked projects.

        Returns:
            List of projects.
        """
        return self._memory.preference_store.get_all_projects()

    def get_project(self, name: str) -> Optional[ProjectInfo]:
        """Get a project by name.

        Args:
            name: Project name.

        Returns:
            Project info or None.
        """
        return self._memory.get_project(name)

    def find_similar_projects(
        self,
        project_name: str,
    ) -> List[Dict[str, Any]]:
        """Find projects similar to a given project.

        Args:
            project_name: Reference project name.

        Returns:
            List of similar projects.
        """
        return self._memory.find_similar_projects(project_name)

    # ==================== Style Guide Generation ====================

    def generate_style_guide(self) -> str:
        """Generate a markdown style guide from learned patterns.

        Returns:
            Markdown-formatted style guide.
        """
        lines = ["# Developer Style Guide", ""]
        lines.append("*Generated from learned coding patterns*")
        lines.append("")

        # Get patterns by category
        for category in PatternCategory:
            patterns = self._memory.preference_store.get_patterns(
                category=category,
                min_confidence=0.6,
                limit=10,
            )

            if not patterns:
                continue

            lines.append(f"## {category.value.title()}")
            lines.append("")

            for pattern in patterns:
                confidence_bar = "█" * int(pattern.confidence_score * 5)
                lines.append(
                    f"### {pattern.name} [{confidence_bar}] ({pattern.confidence_score:.0%})"
                )
                lines.append("")
                lines.append(pattern.description)
                lines.append("")

                if pattern.pattern_text:
                    lines.append("**Pattern:**")
                    lines.append(f"```")
                    lines.append(pattern.pattern_text)
                    lines.append("```")
                    lines.append("")

                if pattern.examples:
                    lines.append("**Examples:**")
                    for example in pattern.examples[:3]:
                        lines.append(f"- `{example}`")
                    lines.append("")

                if pattern.source_projects:
                    lines.append(
                        f"*Learned from: {', '.join(pattern.source_projects[:3])}*"
                    )
                    lines.append("")

        # Add statistics
        lines.append("## Statistics")
        lines.append("")
        summary = self.get_profile_summary()
        lines.append(f"- **Total patterns learned:** {summary['total_patterns']}")
        lines.append(f"- **Projects analyzed:** {summary['total_projects']}")

        if summary.get("patterns_by_category"):
            lines.append("")
            lines.append("**Patterns by category:**")
            for cat, count in summary["patterns_by_category"].items():
                lines.append(f"- {cat}: {count}")

        return "\n".join(lines)

    # ==================== Graph Operations ====================

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the pattern relationship graph.

        Returns:
            Graph statistics and highlights.
        """
        metrics = self._memory.memory_graph.get_metrics()

        # Get cross-project patterns
        cross_project = self._memory.get_cross_project_patterns(min_projects=2)

        return {
            "node_counts": metrics.get("node_counts", {}),
            "relationship_counts": metrics.get("relationship_counts", {}),
            "cross_project_patterns": cross_project[:5],
        }

    def export_graph(self) -> Dict[str, Any]:
        """Export the full style graph for visualization.

        Returns:
            Graph data with nodes and relationships.
        """
        return self._memory.get_developer_style_graph()

    # ==================== Reset ====================

    def reset(self, keep_preferences: bool = False) -> None:
        """Reset the profile.

        Args:
            keep_preferences: If True, keep preferences but clear patterns.
        """
        if keep_preferences:
            # Only clear patterns, projects, and interactions
            self._memory.pattern_store.clear()
            self._memory.memory_graph.clear()

            # Clear patterns from preference store
            patterns = self._memory.preference_store.get_patterns(limit=10000)
            for pattern in patterns:
                self._memory.preference_store.delete_pattern(pattern.id)

            # Clear projects
            projects = self._memory.preference_store.get_all_projects()
            for project in projects:
                self._memory.preference_store.delete_project(project.name)

            logger.info("Reset profile (kept preferences)")
        else:
            self._memory.clear()
            logger.info("Reset entire profile")
