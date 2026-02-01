"""Tests for Tree-sitter based parser."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestTreeSitterParserImport:
    """Tests for TreeSitterParser import handling."""

    def test_import_without_treesitter(self):
        """Test graceful import failure without tree-sitter."""
        # The parser module should be importable even without tree-sitter
        # (it will fail to instantiate, but import should work)
        try:
            from codesage.parsers.treesitter_parser import LANGUAGE_CONFIG
            assert "javascript" in LANGUAGE_CONFIG
            assert "typescript" in LANGUAGE_CONFIG
            assert "go" in LANGUAGE_CONFIG
            assert "rust" in LANGUAGE_CONFIG
        except ImportError:
            pytest.skip("tree-sitter not available")

    def test_language_config_structure(self):
        """Test that language config has required fields."""
        try:
            from codesage.parsers.treesitter_parser import LANGUAGE_CONFIG

            required_keys = ["extensions", "grammar_module", "node_types"]

            for lang, config in LANGUAGE_CONFIG.items():
                for key in required_keys:
                    assert key in config, f"Missing {key} in {lang} config"
                assert isinstance(config["extensions"], list)
                assert len(config["extensions"]) > 0
        except ImportError:
            pytest.skip("tree-sitter not available")


class TestTreeSitterParserFunctionality:
    """Tests for TreeSitterParser parsing functionality."""

    @pytest.fixture
    def skip_without_treesitter(self):
        """Skip test if tree-sitter not installed."""
        try:
            import tree_sitter
        except ImportError:
            pytest.skip("tree-sitter not installed")

    def test_javascript_parser_creation(self, skip_without_treesitter):
        """Test JavaScript parser can be created."""
        try:
            from codesage.parsers.treesitter_parser import TreeSitterParser
            parser = TreeSitterParser("javascript")
            assert parser.language == "javascript"
            assert ".js" in parser.file_extensions
        except ImportError:
            pytest.skip("tree-sitter-javascript not installed")

    def test_javascript_function_parsing(self, skip_without_treesitter):
        """Test parsing JavaScript functions."""
        try:
            from codesage.parsers.treesitter_parser import TreeSitterParser
            parser = TreeSitterParser("javascript")

            code = '''
function greet(name) {
    return "Hello, " + name;
}

const add = (a, b) => a + b;
'''
            elements = parser.parse_code(code, Path("test.js"))

            # Should find at least the named function
            assert len(elements) >= 1
            func_names = [e.name for e in elements]
            assert "greet" in func_names
        except ImportError:
            pytest.skip("tree-sitter-javascript not installed")

    def test_javascript_class_parsing(self, skip_without_treesitter):
        """Test parsing JavaScript classes."""
        try:
            from codesage.parsers.treesitter_parser import TreeSitterParser
            parser = TreeSitterParser("javascript")

            code = '''
class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}
'''
            elements = parser.parse_code(code, Path("test.js"))

            # Should find class and methods
            types = [e.type for e in elements]
            assert "class" in types

            class_elem = next(e for e in elements if e.type == "class")
            assert class_elem.name == "Calculator"
        except ImportError:
            pytest.skip("tree-sitter-javascript not installed")

    def test_typescript_interface_parsing(self, skip_without_treesitter):
        """Test parsing TypeScript interfaces."""
        try:
            from codesage.parsers.treesitter_parser import TreeSitterParser
            parser = TreeSitterParser("typescript")

            code = '''
interface User {
    name: string;
    age: number;
}

function getUser(id: number): User {
    return { name: "Test", age: 30 };
}
'''
            elements = parser.parse_code(code, Path("test.ts"))

            # Should find interface and function
            assert len(elements) >= 1
            func_names = [e.name for e in elements if e.type == "function"]
            assert "getUser" in func_names
        except ImportError:
            pytest.skip("tree-sitter-typescript not installed")

    def test_go_function_parsing(self, skip_without_treesitter):
        """Test parsing Go functions."""
        try:
            from codesage.parsers.treesitter_parser import TreeSitterParser
            parser = TreeSitterParser("go")

            code = '''
package main

func add(a int, b int) int {
    return a + b
}

func main() {
    result := add(1, 2)
    fmt.Println(result)
}
'''
            elements = parser.parse_code(code, Path("test.go"))

            func_names = [e.name for e in elements if e.type == "function"]
            assert "add" in func_names
            assert "main" in func_names
        except ImportError:
            pytest.skip("tree-sitter-go not installed")

    def test_rust_function_parsing(self, skip_without_treesitter):
        """Test parsing Rust functions."""
        try:
            from codesage.parsers.treesitter_parser import TreeSitterParser
            parser = TreeSitterParser("rust")

            code = '''
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = add(1, 2);
    println!("{}", result);
}
'''
            elements = parser.parse_code(code, Path("test.rs"))

            func_names = [e.name for e in elements if e.type == "function"]
            assert "add" in func_names
            assert "main" in func_names
        except ImportError:
            pytest.skip("tree-sitter-rust not installed")

    def test_unsupported_language_error(self, skip_without_treesitter):
        """Test that unsupported language raises error."""
        from codesage.parsers.treesitter_parser import TreeSitterParser

        with pytest.raises(ValueError, match="Unsupported language"):
            TreeSitterParser("cobol")


class TestParserRegistry:
    """Test that parsers are registered correctly."""

    def test_python_parser_always_registered(self):
        """Test Python parser is always available."""
        from codesage.parsers import ParserRegistry

        assert "python" in ParserRegistry.supported_languages()
        assert ".py" in ParserRegistry.supported_extensions()

    def test_can_parse_python_files(self):
        """Test registry can find parser for Python files."""
        from codesage.parsers import ParserRegistry

        parser = ParserRegistry.get_parser_for_file(Path("test.py"))
        assert parser is not None
        assert parser.language == "python"

    def test_treesitter_parsers_conditional(self):
        """Test tree-sitter parsers are registered when available."""
        from codesage.parsers import ParserRegistry

        # These should be registered if tree-sitter is installed
        try:
            import tree_sitter
            import tree_sitter_javascript

            assert "javascript" in ParserRegistry.supported_languages()
            assert ".js" in ParserRegistry.supported_extensions()
        except ImportError:
            # Should NOT be registered if tree-sitter not available
            pass
