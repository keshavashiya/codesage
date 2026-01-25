"""Tests for data models."""

import pytest
from pathlib import Path

from codesage.models.code_element import CodeElement
from codesage.models.suggestion import Suggestion, Pattern


class TestCodeElement:
    """Tests for CodeElement model."""

    def test_create_function(self):
        """Test creating a function element."""
        element = CodeElement.create(
            file=Path("test.py"),
            type="function",
            code="def hello(): pass",
            language="python",
            line_start=1,
            line_end=1,
            name="hello",
        )

        assert element.id is not None
        assert element.name == "hello"
        assert element.type == "function"
        assert element.language == "python"

    def test_unique_ids(self):
        """Test that different elements get unique IDs."""
        elem1 = CodeElement.create(
            file=Path("test.py"),
            type="function",
            code="def a(): pass",
            language="python",
            line_start=1,
            line_end=1,
        )

        elem2 = CodeElement.create(
            file=Path("test.py"),
            type="function",
            code="def b(): pass",
            language="python",
            line_start=2,
            line_end=2,
        )

        assert elem1.id != elem2.id

    def test_to_dict(self):
        """Test converting element to dictionary."""
        element = CodeElement.create(
            file=Path("test.py"),
            type="function",
            code="def hello(): pass",
            language="python",
            line_start=1,
            line_end=1,
            name="hello",
            docstring="A greeting function.",
        )

        data = element.to_dict()

        assert data["name"] == "hello"
        assert data["type"] == "function"
        assert data["docstring"] == "A greeting function."

    def test_from_dict(self):
        """Test creating element from dictionary."""
        data = {
            "id": "abc123",
            "file": "test.py",
            "type": "function",
            "name": "hello",
            "code": "def hello(): pass",
            "language": "python",
            "line_start": 1,
            "line_end": 1,
        }

        element = CodeElement.from_dict(data)

        assert element.id == "abc123"
        assert element.name == "hello"

    def test_get_embedding_text(self):
        """Test generating embedding text."""
        element = CodeElement.create(
            file=Path("test.py"),
            type="function",
            code="def hello(): pass",
            language="python",
            line_start=1,
            line_end=1,
            name="hello",
            docstring="A greeting function.",
        )

        text = element.get_embedding_text()

        assert "function: hello" in text
        assert "Description: A greeting function." in text
        assert "def hello(): pass" in text


class TestSuggestion:
    """Tests for Suggestion model."""

    def test_create_suggestion(self):
        """Test creating a suggestion."""
        suggestion = Suggestion(
            file=Path("test.py"),
            line=10,
            code="def hello(): pass",
            similarity=0.85,
            language="python",
            element_type="function",
            name="hello",
        )

        assert suggestion.similarity == 0.85
        assert suggestion.name == "hello"

    def test_to_dict(self):
        """Test converting suggestion to dictionary."""
        suggestion = Suggestion(
            file=Path("test.py"),
            line=10,
            code="def hello(): pass",
            similarity=0.85,
            language="python",
            element_type="function",
        )

        data = suggestion.to_dict()

        assert data["line"] == 10
        assert data["similarity"] == 0.85


class TestPattern:
    """Tests for Pattern model."""

    def test_create_pattern(self):
        """Test creating a pattern."""
        pattern = Pattern(
            name="Exception Handling",
            description="Uses try-except blocks",
            occurrences=15,
            examples=["try: ... except: ..."],
            category="error-handling",
        )

        assert pattern.name == "Exception Handling"
        assert pattern.occurrences == 15
        assert pattern.category == "error-handling"
