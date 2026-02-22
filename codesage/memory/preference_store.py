"""SQLite-based preference store for developer memory.

Stores preferences, interaction history, and metrics in SQLite.
Follows the ConnectionManager pattern from codesage.storage.base.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from codesage.storage.base import ConnectionManager
from codesage.utils.logging import get_logger

from .models import (
    DeveloperPreference,
    InteractionRecord,
    LearnedPattern,
    PatternCategory,
    ProjectInfo,
)

logger = get_logger("memory.preference_store")


class PreferenceStore(ConnectionManager):
    """SQLite store for developer preferences and metrics.

    Extends ConnectionManager to provide preference-specific operations.
    Stores:
        - Developer preferences (key-value settings)
        - Pattern metadata and metrics
        - Project information
        - Interaction history
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize the preference store.

        Args:
            db_path: Path to the SQLite database file.
        """
        super().__init__(db_path)
        self._init_schema()
        self._harden_permissions()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self.transaction() as conn:
            # Preferences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    description TEXT DEFAULT '',
                    updated_at TEXT NOT NULL
                )
            """)

            # Patterns metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    pattern_text TEXT NOT NULL,
                    examples TEXT DEFAULT '[]',
                    occurrence_count INTEGER DEFAULT 1,
                    confidence_score REAL DEFAULT 0.5,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
            """)

            # Pattern-project relationship table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_projects (
                    pattern_id TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    occurrences INTEGER DEFAULT 1,
                    PRIMARY KEY (pattern_id, project_name),
                    FOREIGN KEY (pattern_id) REFERENCES patterns(id)
                )
            """)

            # Projects table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    path TEXT NOT NULL,
                    language TEXT DEFAULT 'python',
                    total_files INTEGER DEFAULT 0,
                    total_elements INTEGER DEFAULT 0,
                    first_indexed TEXT NOT NULL,
                    last_indexed TEXT NOT NULL,
                    patterns_learned INTEGER DEFAULT 0
                )
            """)

            # Interactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    interaction_type TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    query TEXT DEFAULT '',
                    response TEXT DEFAULT '',
                    accepted INTEGER DEFAULT NULL,
                    feedback TEXT DEFAULT '',
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_category ON patterns(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence_score)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(interaction_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_project ON interactions(project_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)"
            )

        logger.debug("PreferenceStore schema initialized")

    # ==================== Preference Operations ====================

    def set_preference(self, preference: DeveloperPreference) -> None:
        """Set a developer preference.

        Args:
            preference: Preference to set.
        """
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO preferences (key, value, category, description, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    preference.key,
                    json.dumps(preference.value),
                    preference.category,
                    preference.description,
                    preference.updated_at.isoformat()
                    if preference.updated_at
                    else datetime.now().isoformat(),
                ),
            )

    def get_preference(self, key: str) -> Optional[DeveloperPreference]:
        """Get a preference by key.

        Args:
            key: Preference key.

        Returns:
            DeveloperPreference or None if not found.
        """
        row = self.conn.execute(
            "SELECT key, value, category, description, updated_at FROM preferences WHERE key = ?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        return DeveloperPreference(
            key=row["key"],
            value=json.loads(row["value"]),
            category=row["category"],
            description=row["description"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def get_all_preferences(
        self, category: Optional[str] = None
    ) -> List[DeveloperPreference]:
        """Get all preferences, optionally filtered by category.

        Args:
            category: Optional category filter.

        Returns:
            List of preferences.
        """
        if category:
            rows = self.conn.execute(
                "SELECT key, value, category, description, updated_at FROM preferences WHERE category = ?",
                (category,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT key, value, category, description, updated_at FROM preferences"
            ).fetchall()

        return [
            DeveloperPreference(
                key=row["key"],
                value=json.loads(row["value"]),
                category=row["category"],
                description=row["description"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def delete_preference(self, key: str) -> None:
        """Delete a preference.

        Args:
            key: Preference key to delete.
        """
        with self.transaction() as conn:
            conn.execute("DELETE FROM preferences WHERE key = ?", (key,))

    # ==================== Pattern Operations ====================

    def add_pattern(self, pattern: LearnedPattern) -> None:
        """Add or update a pattern.

        Args:
            pattern: Pattern to add.
        """
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO patterns
                (id, name, category, description, pattern_text, examples,
                 occurrence_count, confidence_score, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern.id,
                    pattern.name,
                    pattern.category.value,
                    pattern.description,
                    pattern.pattern_text,
                    json.dumps(pattern.examples),
                    pattern.occurrence_count,
                    pattern.confidence_score,
                    pattern.first_seen.isoformat()
                    if pattern.first_seen
                    else datetime.now().isoformat(),
                    pattern.last_seen.isoformat()
                    if pattern.last_seen
                    else datetime.now().isoformat(),
                ),
            )

    def get_pattern(self, pattern_id: str) -> Optional[LearnedPattern]:
        """Get a pattern by ID.

        Args:
            pattern_id: Pattern ID.

        Returns:
            LearnedPattern or None if not found.
        """
        row = self.conn.execute(
            """
            SELECT id, name, category, description, pattern_text, examples,
                   occurrence_count, confidence_score, first_seen, last_seen
            FROM patterns WHERE id = ?
            """,
            (pattern_id,),
        ).fetchone()

        if row is None:
            return None

        # Get source projects
        project_rows = self.conn.execute(
            "SELECT project_name FROM pattern_projects WHERE pattern_id = ?",
            (pattern_id,),
        ).fetchall()

        return LearnedPattern(
            id=row["id"],
            name=row["name"],
            category=PatternCategory(row["category"]),
            description=row["description"],
            pattern_text=row["pattern_text"],
            examples=json.loads(row["examples"]),
            occurrence_count=row["occurrence_count"],
            confidence_score=row["confidence_score"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            source_projects=[r["project_name"] for r in project_rows],
        )

    def get_patterns(
        self,
        category: Optional[PatternCategory] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> List[LearnedPattern]:
        """Get patterns with optional filtering.

        Args:
            category: Optional category filter.
            min_confidence: Minimum confidence score.
            limit: Maximum number of patterns to return.

        Returns:
            List of patterns.
        """
        query = """
            SELECT id, name, category, description, pattern_text, examples,
                   occurrence_count, confidence_score, first_seen, last_seen
            FROM patterns
            WHERE confidence_score >= ?
        """
        params: List[Any] = [min_confidence]

        if category:
            query += " AND category = ?"
            params.append(category.value)

        query += " ORDER BY confidence_score DESC, occurrence_count DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()

        return [
            LearnedPattern(
                id=row["id"],
                name=row["name"],
                category=PatternCategory(row["category"]),
                description=row["description"],
                pattern_text=row["pattern_text"],
                examples=json.loads(row["examples"]),
                occurrence_count=row["occurrence_count"],
                confidence_score=row["confidence_score"],
                first_seen=datetime.fromisoformat(row["first_seen"]),
                last_seen=datetime.fromisoformat(row["last_seen"]),
            )
            for row in rows
        ]

    def update_pattern_occurrence(self, pattern_id: str, increment: int = 1) -> None:
        """Update pattern occurrence count.

        Args:
            pattern_id: Pattern ID.
            increment: Amount to increment by.
        """
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE patterns
                SET occurrence_count = occurrence_count + ?, last_seen = ?
                WHERE id = ?
                """,
                (increment, datetime.now().isoformat(), pattern_id),
            )

    def link_pattern_to_project(self, pattern_id: str, project_name: str) -> None:
        """Link a pattern to a project.

        Args:
            pattern_id: Pattern ID.
            project_name: Project name.
        """
        with self.transaction() as conn:
            now = datetime.now().isoformat()
            conn.execute(
                """
                INSERT INTO pattern_projects (pattern_id, project_name, first_seen, occurrences)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(pattern_id, project_name) DO UPDATE SET
                    occurrences = occurrences + 1
                """,
                (pattern_id, project_name, now),
            )

    def get_pattern_count(self, category: Optional[PatternCategory] = None) -> int:
        """Get count of patterns.

        Args:
            category: Optional category filter.

        Returns:
            Number of patterns.
        """
        if category:
            row = self.conn.execute(
                "SELECT COUNT(*) as count FROM patterns WHERE category = ?",
                (category.value,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) as count FROM patterns").fetchone()

        return row["count"] if row else 0

    def delete_pattern(self, pattern_id: str) -> None:
        """Delete a pattern.

        Args:
            pattern_id: Pattern ID to delete.
        """
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM pattern_projects WHERE pattern_id = ?", (pattern_id,)
            )
            conn.execute("DELETE FROM patterns WHERE id = ?", (pattern_id,))

    # ==================== Project Operations ====================

    def add_project(self, project: ProjectInfo) -> None:
        """Add or update a project.

        Args:
            project: Project info to add.
        """
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO projects
                (id, name, path, language, total_files, total_elements,
                 first_indexed, last_indexed, patterns_learned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.name,
                    str(project.path),
                    project.language,
                    project.total_files,
                    project.total_elements,
                    project.first_indexed.isoformat()
                    if project.first_indexed
                    else datetime.now().isoformat(),
                    project.last_indexed.isoformat()
                    if project.last_indexed
                    else datetime.now().isoformat(),
                    project.patterns_learned,
                ),
            )

    def get_project(self, name: str) -> Optional[ProjectInfo]:
        """Get a project by name.

        Args:
            name: Project name.

        Returns:
            ProjectInfo or None if not found.
        """
        row = self.conn.execute(
            """
            SELECT id, name, path, language, total_files, total_elements,
                   first_indexed, last_indexed, patterns_learned
            FROM projects WHERE name = ?
            """,
            (name,),
        ).fetchone()

        if row is None:
            return None

        return ProjectInfo(
            id=row["id"],
            name=row["name"],
            path=Path(row["path"]),
            language=row["language"],
            total_files=row["total_files"],
            total_elements=row["total_elements"],
            first_indexed=datetime.fromisoformat(row["first_indexed"]),
            last_indexed=datetime.fromisoformat(row["last_indexed"]),
            patterns_learned=row["patterns_learned"],
        )

    def get_all_projects(self) -> List[ProjectInfo]:
        """Get all projects.

        Returns:
            List of projects.
        """
        rows = self.conn.execute(
            """
            SELECT id, name, path, language, total_files, total_elements,
                   first_indexed, last_indexed, patterns_learned
            FROM projects ORDER BY last_indexed DESC
            """
        ).fetchall()

        return [
            ProjectInfo(
                id=row["id"],
                name=row["name"],
                path=Path(row["path"]),
                language=row["language"],
                total_files=row["total_files"],
                total_elements=row["total_elements"],
                first_indexed=datetime.fromisoformat(row["first_indexed"]),
                last_indexed=datetime.fromisoformat(row["last_indexed"]),
                patterns_learned=row["patterns_learned"],
            )
            for row in rows
        ]

    def update_project_stats(
        self,
        name: str,
        total_files: Optional[int] = None,
        total_elements: Optional[int] = None,
        patterns_learned: Optional[int] = None,
    ) -> None:
        """Update project statistics.

        Args:
            name: Project name.
            total_files: New file count (optional).
            total_elements: New element count (optional).
            patterns_learned: New patterns count (optional).
        """
        updates: List[str] = []
        params: List[Any] = []

        if total_files is not None:
            updates.append("total_files = ?")
            params.append(total_files)
        if total_elements is not None:
            updates.append("total_elements = ?")
            params.append(total_elements)
        if patterns_learned is not None:
            updates.append("patterns_learned = ?")
            params.append(patterns_learned)

        if not updates:
            return

        # Always update last_indexed
        updates.append("last_indexed = ?")
        params.append(datetime.now().isoformat())
        params.append(name)

        with self.transaction() as conn:
            # Build query safely with validated column names only
            query = f"UPDATE projects SET {', '.join(updates)} WHERE name = ?"
            conn.execute(query, params)

    def delete_project(self, name: str) -> None:
        """Delete a project.

        Args:
            name: Project name to delete.
        """
        with self.transaction() as conn:
            conn.execute("DELETE FROM pattern_projects WHERE project_name = ?", (name,))
            conn.execute("DELETE FROM projects WHERE name = ?", (name,))

    # ==================== Interaction Operations ====================

    def add_interaction(self, interaction: InteractionRecord) -> None:
        """Add an interaction record.

        Args:
            interaction: Interaction to add.
        """
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO interactions
                (id, interaction_type, project_name, query, response, accepted,
                 feedback, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction.id,
                    interaction.interaction_type,
                    interaction.project_name,
                    interaction.query,
                    interaction.response,
                    1
                    if interaction.accepted
                    else (0 if interaction.accepted is False else None),
                    interaction.feedback,
                    interaction.timestamp.isoformat()
                    if interaction.timestamp
                    else datetime.now().isoformat(),
                    json.dumps(interaction.metadata),
                ),
            )

    def get_interactions(
        self,
        project_name: Optional[str] = None,
        interaction_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[InteractionRecord]:
        """Get interaction records with optional filtering.

        Args:
            project_name: Optional project filter.
            interaction_type: Optional type filter.
            limit: Maximum number of records to return.

        Returns:
            List of interaction records.
        """
        query = """
            SELECT id, interaction_type, project_name, query, response,
                   accepted, feedback, timestamp, metadata
            FROM interactions WHERE 1=1
        """
        params: List[Any] = []

        if project_name:
            query += " AND project_name = ?"
            params.append(project_name)
        if interaction_type:
            query += " AND interaction_type = ?"
            params.append(interaction_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()

        return [
            InteractionRecord(
                id=row["id"],
                interaction_type=row["interaction_type"],
                project_name=row["project_name"],
                query=row["query"],
                response=row["response"],
                accepted=True
                if row["accepted"] == 1
                else (False if row["accepted"] == 0 else None),
                feedback=row["feedback"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def get_interaction_stats(self) -> Dict[str, Any]:
        """Get interaction statistics.

        Returns:
            Dictionary with interaction stats.
        """
        stats = {}

        # Total count
        row = self.conn.execute("SELECT COUNT(*) as count FROM interactions").fetchone()
        stats["total"] = row["count"] if row else 0

        # By type
        rows = self.conn.execute(
            "SELECT interaction_type, COUNT(*) as count FROM interactions GROUP BY interaction_type"
        ).fetchall()
        stats["by_type"] = {row["interaction_type"]: row["count"] for row in rows}

        # Acceptance rate
        row = self.conn.execute(
            "SELECT AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as rate FROM interactions WHERE accepted IS NOT NULL"
        ).fetchone()
        stats["acceptance_rate"] = (
            row["rate"] if row and row["rate"] is not None else 0.0
        )

        return stats

    # ==================== Metrics ====================

    def get_metrics(self) -> Dict[str, Any]:
        """Get storage metrics.

        Returns:
            Dictionary with storage statistics.
        """
        return {
            "backend": "sqlite",
            "db_path": str(self.db_path),
            "pattern_count": self.get_pattern_count(),
            "project_count": len(self.get_all_projects()),
            "preference_count": len(self.get_all_preferences()),
            "interaction_stats": self.get_interaction_stats(),
        }

    def clear(self) -> None:
        """Clear all data from the store."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM interactions")
            conn.execute("DELETE FROM pattern_projects")
            conn.execute("DELETE FROM patterns")
            conn.execute("DELETE FROM projects")
            conn.execute("DELETE FROM preferences")

        logger.info("Cleared all data from PreferenceStore")
