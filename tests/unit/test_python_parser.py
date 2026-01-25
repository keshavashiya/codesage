"""Tests for Python parser."""

import pytest
from pathlib import Path

from codesage.parsers.python_parser import PythonParser


class TestPythonParser:
    """Tests for Python AST parser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return PythonParser()

    def test_language_property(self, parser):
        """Test language property."""
        assert parser.language == "python"

    def test_file_extensions(self, parser):
        """Test file extensions."""
        assert ".py" in parser.file_extensions

    def test_can_parse(self, parser):
        """Test can_parse method."""
        assert parser.can_parse(Path("test.py"))
        assert not parser.can_parse(Path("test.js"))

    def test_parse_simple_function(self, parser):
        """Test parsing a simple function."""
        code = '''
def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"
'''
        elements = parser.parse_code(code, Path("test.py"))

        assert len(elements) == 1
        assert elements[0].type == "function"
        assert elements[0].name == "hello"
        assert elements[0].docstring == "Greet someone."
        assert "def hello" in elements[0].code

    def test_parse_async_function(self, parser):
        """Test parsing an async function."""
        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    pass
'''
        elements = parser.parse_code(code, Path("test.py"))

        assert len(elements) == 1
        assert elements[0].type == "function"
        assert elements[0].name == "fetch_data"
        assert "async def" in elements[0].signature

    def test_parse_class(self, parser):
        """Test parsing a class."""
        code = '''
class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
'''
        elements = parser.parse_code(code, Path("test.py"))

        # Should find class and method
        assert len(elements) == 2

        class_elem = next(e for e in elements if e.type == "class")
        assert class_elem.name == "Calculator"
        assert class_elem.docstring == "A simple calculator."

        method_elem = next(e for e in elements if e.type == "method")
        assert method_elem.name == "add"

    def test_parse_class_with_inheritance(self, parser):
        """Test parsing a class with inheritance."""
        code = '''
class Dog(Animal):
    """A dog class."""
    pass
'''
        elements = parser.parse_code(code, Path("test.py"))

        assert len(elements) == 1
        assert "Animal" in elements[0].signature

    def test_parse_function_with_defaults(self, parser):
        """Test parsing function with default arguments."""
        code = '''
def greet(name: str = "World", times: int = 1) -> None:
    pass
'''
        elements = parser.parse_code(code, Path("test.py"))

        assert len(elements) == 1
        assert "=..." in elements[0].signature  # Default indicated

    def test_parse_syntax_error(self, parser):
        """Test parsing invalid Python syntax."""
        code = '''
def broken(
    # Missing closing paren
'''
        elements = parser.parse_code(code, Path("test.py"))

        assert len(elements) == 0  # Should not crash

    def test_element_id_generation(self, parser):
        """Test that elements get unique IDs."""
        code = '''
def func1():
    pass

def func2():
    pass
'''
        elements = parser.parse_code(code, Path("test.py"))

        assert len(elements) == 2
        assert elements[0].id != elements[1].id
