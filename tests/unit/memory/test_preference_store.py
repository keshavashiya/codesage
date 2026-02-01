"""Tests for PreferenceStore (SQLite)."""

from datetime import datetime
from pathlib import Path

import pytest

from codesage.memory.models import (
    DeveloperPreference,
    InteractionRecord,
    LearnedPattern,
    PatternCategory,
    ProjectInfo,
)
from codesage.memory.preference_store import PreferenceStore


@pytest.fixture
def preference_store(tmp_path):
    """Create a PreferenceStore instance for testing."""
    db_path = tmp_path / "test_profile.db"
    return PreferenceStore(db_path)


class TestPreferenceStoreInit:
    """Tests for PreferenceStore initialization."""

    def test_creates_database_file(self, tmp_path):
        """Test that database file is created."""
        db_path = tmp_path / "test.db"
        assert not db_path.exists()

        store = PreferenceStore(db_path)
        assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        db_path = tmp_path / "nested" / "dirs" / "test.db"
        assert not db_path.parent.exists()

        store = PreferenceStore(db_path)
        assert db_path.exists()


class TestPreferenceOperations:
    """Tests for preference CRUD operations."""

    def test_set_and_get_preference(self, preference_store):
        """Test setting and getting a preference."""
        pref = DeveloperPreference(
            key="theme",
            value="dark",
            category="ui",
        )
        preference_store.set_preference(pref)

        result = preference_store.get_preference("theme")
        assert result is not None
        assert result.key == "theme"
        assert result.value == "dark"
        assert result.category == "ui"

    def test_get_nonexistent_preference(self, preference_store):
        """Test getting a preference that doesn't exist."""
        result = preference_store.get_preference("nonexistent")
        assert result is None

    def test_update_preference(self, preference_store):
        """Test updating an existing preference."""
        pref1 = DeveloperPreference(key="theme", value="dark")
        preference_store.set_preference(pref1)

        pref2 = DeveloperPreference(key="theme", value="light")
        preference_store.set_preference(pref2)

        result = preference_store.get_preference("theme")
        assert result.value == "light"

    def test_get_all_preferences(self, preference_store):
        """Test getting all preferences."""
        preference_store.set_preference(DeveloperPreference(key="a", value=1))
        preference_store.set_preference(DeveloperPreference(key="b", value=2))

        all_prefs = preference_store.get_all_preferences()
        assert len(all_prefs) == 2

    def test_get_preferences_by_category(self, preference_store):
        """Test getting preferences filtered by category."""
        preference_store.set_preference(
            DeveloperPreference(key="a", value=1, category="ui")
        )
        preference_store.set_preference(
            DeveloperPreference(key="b", value=2, category="formatting")
        )

        ui_prefs = preference_store.get_all_preferences(category="ui")
        assert len(ui_prefs) == 1
        assert ui_prefs[0].key == "a"

    def test_delete_preference(self, preference_store):
        """Test deleting a preference."""
        preference_store.set_preference(DeveloperPreference(key="test", value="x"))
        preference_store.delete_preference("test")

        result = preference_store.get_preference("test")
        assert result is None


class TestPatternOperations:
    """Tests for pattern CRUD operations."""

    def test_add_and_get_pattern(self, preference_store):
        """Test adding and getting a pattern."""
        pattern = LearnedPattern.create(
            name="snake_case",
            category=PatternCategory.NAMING,
            description="Snake case naming",
            pattern_text=r"^[a-z_]+$",
        )
        preference_store.add_pattern(pattern)

        result = preference_store.get_pattern(pattern.id)
        assert result is not None
        assert result.name == "snake_case"
        assert result.category == PatternCategory.NAMING

    def test_get_patterns_with_filters(self, preference_store):
        """Test getting patterns with filters."""
        p1 = LearnedPattern.create(
            name="naming1",
            category=PatternCategory.NAMING,
            description="test",
            pattern_text="test",
            confidence_score=0.8,
        )
        p2 = LearnedPattern.create(
            name="typing1",
            category=PatternCategory.TYPING,
            description="test",
            pattern_text="test",
            confidence_score=0.3,
        )
        preference_store.add_pattern(p1)
        preference_store.add_pattern(p2)

        # Filter by category
        naming = preference_store.get_patterns(category=PatternCategory.NAMING)
        assert len(naming) == 1
        assert naming[0].name == "naming1"

        # Filter by confidence
        confident = preference_store.get_patterns(min_confidence=0.5)
        assert len(confident) == 1
        assert confident[0].name == "naming1"

    def test_update_pattern_occurrence(self, preference_store):
        """Test updating pattern occurrence count."""
        pattern = LearnedPattern.create(
            name="test",
            category=PatternCategory.STRUCTURE,
            description="test",
            pattern_text="test",
        )
        preference_store.add_pattern(pattern)

        preference_store.update_pattern_occurrence(pattern.id, increment=5)

        result = preference_store.get_pattern(pattern.id)
        assert result.occurrence_count == 6  # 1 initial + 5

    def test_link_pattern_to_project(self, preference_store):
        """Test linking a pattern to a project."""
        pattern = LearnedPattern.create(
            name="test",
            category=PatternCategory.IMPORTS,
            description="test",
            pattern_text="test",
        )
        preference_store.add_pattern(pattern)

        preference_store.link_pattern_to_project(pattern.id, "my_project")

        result = preference_store.get_pattern(pattern.id)
        assert "my_project" in result.source_projects


class TestProjectOperations:
    """Tests for project CRUD operations."""

    def test_add_and_get_project(self, preference_store):
        """Test adding and getting a project."""
        project = ProjectInfo.create(
            name="my_project",
            path=Path("/tmp/my_project"),
            total_files=10,
        )
        preference_store.add_project(project)

        result = preference_store.get_project("my_project")
        assert result is not None
        assert result.name == "my_project"
        assert result.total_files == 10

    def test_get_all_projects(self, preference_store):
        """Test getting all projects."""
        preference_store.add_project(
            ProjectInfo.create(name="project1", path=Path("/tmp/p1"))
        )
        preference_store.add_project(
            ProjectInfo.create(name="project2", path=Path("/tmp/p2"))
        )

        projects = preference_store.get_all_projects()
        assert len(projects) == 2

    def test_update_project_stats(self, preference_store):
        """Test updating project statistics."""
        project = ProjectInfo.create(
            name="test",
            path=Path("/tmp/test"),
            total_files=5,
        )
        preference_store.add_project(project)

        preference_store.update_project_stats(
            "test",
            total_files=10,
            total_elements=100,
        )

        result = preference_store.get_project("test")
        assert result.total_files == 10
        assert result.total_elements == 100


class TestInteractionOperations:
    """Tests for interaction operations."""

    def test_add_and_get_interactions(self, preference_store):
        """Test adding and getting interactions."""
        interaction = InteractionRecord.create(
            interaction_type="query",
            project_name="my_project",
            query="find foo",
        )
        preference_store.add_interaction(interaction)

        results = preference_store.get_interactions()
        assert len(results) == 1
        assert results[0].interaction_type == "query"

    def test_filter_interactions(self, preference_store):
        """Test filtering interactions."""
        preference_store.add_interaction(
            InteractionRecord.create(
                interaction_type="query",
                project_name="project1",
            )
        )
        preference_store.add_interaction(
            InteractionRecord.create(
                interaction_type="suggestion",
                project_name="project2",
            )
        )

        queries = preference_store.get_interactions(interaction_type="query")
        assert len(queries) == 1
        assert queries[0].interaction_type == "query"

        p1 = preference_store.get_interactions(project_name="project1")
        assert len(p1) == 1

    def test_interaction_stats(self, preference_store):
        """Test interaction statistics."""
        preference_store.add_interaction(
            InteractionRecord.create(
                interaction_type="suggestion",
                project_name="test",
                accepted=True,
            )
        )
        preference_store.add_interaction(
            InteractionRecord.create(
                interaction_type="suggestion",
                project_name="test",
                accepted=False,
            )
        )

        stats = preference_store.get_interaction_stats()
        assert stats["total"] == 2
        assert stats["acceptance_rate"] == 0.5


class TestMetrics:
    """Tests for metrics operations."""

    def test_get_metrics(self, preference_store):
        """Test getting store metrics."""
        preference_store.add_pattern(
            LearnedPattern.create(
                name="test",
                category=PatternCategory.NAMING,
                description="test",
                pattern_text="test",
            )
        )

        metrics = preference_store.get_metrics()
        assert metrics["backend"] == "sqlite"
        assert metrics["pattern_count"] == 1

    def test_clear(self, preference_store):
        """Test clearing all data."""
        preference_store.set_preference(DeveloperPreference(key="a", value=1))
        preference_store.add_pattern(
            LearnedPattern.create(
                name="test",
                category=PatternCategory.TYPING,
                description="test",
                pattern_text="test",
            )
        )

        preference_store.clear()

        assert len(preference_store.get_all_preferences()) == 0
        assert preference_store.get_pattern_count() == 0
