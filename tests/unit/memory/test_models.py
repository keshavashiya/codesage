"""Tests for memory data models."""

from datetime import datetime

import pytest

from codesage.memory.models import (
    CodeStructure,
    DeveloperPreference,
    InteractionRecord,
    LearnedPattern,
    PatternCategory,
    PatternRelationship,
    ProjectInfo,
    RelationshipType,
    StructureType,
)


class TestLearnedPattern:
    """Tests for LearnedPattern model."""

    def test_create_pattern(self):
        """Test pattern creation with factory method."""
        pattern = LearnedPattern.create(
            name="snake_case_functions",
            category=PatternCategory.NAMING,
            description="Uses snake_case for function names",
            pattern_text=r"^[a-z][a-z0-9_]*$",
        )

        assert pattern.id is not None
        assert len(pattern.id) == 16
        assert pattern.name == "snake_case_functions"
        assert pattern.category == PatternCategory.NAMING
        assert pattern.occurrence_count == 1
        assert pattern.confidence_score == 0.5

    def test_pattern_serialization(self):
        """Test pattern to_dict and from_dict."""
        pattern = LearnedPattern.create(
            name="test_pattern",
            category=PatternCategory.TYPING,
            description="Test description",
            pattern_text="test",
            examples=["example1", "example2"],
            occurrence_count=5,
            confidence_score=0.8,
        )

        data = pattern.to_dict()
        restored = LearnedPattern.from_dict(data)

        assert restored.id == pattern.id
        assert restored.name == pattern.name
        assert restored.category == pattern.category
        assert restored.examples == pattern.examples
        assert restored.occurrence_count == pattern.occurrence_count

    def test_pattern_timestamps(self):
        """Test pattern timestamp initialization."""
        pattern = LearnedPattern.create(
            name="test",
            category=PatternCategory.STRUCTURE,
            description="test",
            pattern_text="test",
        )

        assert pattern.first_seen is not None
        assert pattern.last_seen is not None
        assert pattern.first_seen <= pattern.last_seen


class TestCodeStructure:
    """Tests for CodeStructure model."""

    def test_create_structure(self):
        """Test structure creation."""
        structure = CodeStructure.create(
            structure_type=StructureType.CLASS_HIERARCHY,
            name="dataclass_preference",
            description="Prefers dataclasses",
            example_code="@dataclass\nclass Foo: ...",
        )

        assert structure.id is not None
        assert structure.structure_type == StructureType.CLASS_HIERARCHY
        assert structure.occurrence_count == 1

    def test_structure_serialization(self):
        """Test structure to_dict and from_dict."""
        structure = CodeStructure.create(
            structure_type=StructureType.CALL_PATTERN,
            name="test",
            description="test",
            example_code="test",
            occurrence_count=10,
            confidence=0.9,
        )

        data = structure.to_dict()
        restored = CodeStructure.from_dict(data)

        assert restored.id == structure.id
        assert restored.structure_type == structure.structure_type
        assert restored.occurrence_count == 10
        assert restored.confidence == 0.9


class TestPatternRelationship:
    """Tests for PatternRelationship model."""

    def test_create_relationship(self):
        """Test relationship creation."""
        rel = PatternRelationship(
            source_id="pattern1",
            target_id="pattern2",
            rel_type=RelationshipType.CO_OCCURS,
            weight=0.8,
        )

        assert rel.source_id == "pattern1"
        assert rel.target_id == "pattern2"
        assert rel.rel_type == RelationshipType.CO_OCCURS
        assert rel.weight == 0.8

    def test_relationship_serialization(self):
        """Test relationship to_dict and from_dict."""
        rel = PatternRelationship(
            source_id="a",
            target_id="b",
            rel_type=RelationshipType.EVOLVES_TO,
            weight=1.0,
            metadata={"reason": "test"},
        )

        data = rel.to_dict()
        restored = PatternRelationship.from_dict(data)

        assert restored.source_id == rel.source_id
        assert restored.rel_type == rel.rel_type
        assert restored.metadata == {"reason": "test"}


class TestProjectInfo:
    """Tests for ProjectInfo model."""

    def test_create_project(self):
        """Test project creation."""
        from pathlib import Path

        project = ProjectInfo.create(
            name="my_project",
            path=Path("/tmp/my_project"),
            total_files=10,
            total_elements=100,
        )

        assert project.id is not None
        assert project.name == "my_project"
        assert project.total_files == 10

    def test_project_serialization(self):
        """Test project to_dict and from_dict."""
        from pathlib import Path

        project = ProjectInfo.create(
            name="test",
            path=Path("/tmp/test"),
            language="python",
        )

        data = project.to_dict()
        restored = ProjectInfo.from_dict(data)

        assert restored.id == project.id
        assert restored.name == "test"
        assert restored.language == "python"


class TestDeveloperPreference:
    """Tests for DeveloperPreference model."""

    def test_create_preference(self):
        """Test preference creation."""
        pref = DeveloperPreference(
            key="theme",
            value="dark",
            category="ui",
            description="UI theme preference",
        )

        assert pref.key == "theme"
        assert pref.value == "dark"
        assert pref.category == "ui"
        assert pref.updated_at is not None

    def test_preference_serialization(self):
        """Test preference to_dict and from_dict."""
        pref = DeveloperPreference(
            key="indent_size",
            value=4,
            category="formatting",
        )

        data = pref.to_dict()
        restored = DeveloperPreference.from_dict(data)

        assert restored.key == pref.key
        assert restored.value == 4
        assert restored.category == "formatting"


class TestInteractionRecord:
    """Tests for InteractionRecord model."""

    def test_create_interaction(self):
        """Test interaction creation."""
        interaction = InteractionRecord.create(
            interaction_type="query",
            project_name="my_project",
            query="find function foo",
            response="Found 2 results",
        )

        assert interaction.id is not None
        assert interaction.interaction_type == "query"
        assert interaction.project_name == "my_project"
        assert interaction.timestamp is not None

    def test_interaction_serialization(self):
        """Test interaction to_dict and from_dict."""
        interaction = InteractionRecord.create(
            interaction_type="suggestion",
            project_name="test",
            accepted=True,
            metadata={"count": 5},
        )

        data = interaction.to_dict()
        restored = InteractionRecord.from_dict(data)

        assert restored.id == interaction.id
        assert restored.accepted is True
        assert restored.metadata == {"count": 5}
