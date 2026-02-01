"""Tests for language detection module."""

import pytest
from pathlib import Path
import tempfile
import os


class TestLanguageDetector:
    """Tests for language detection functionality."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project with mixed language files."""
        # Create Python files
        py_dir = tmp_path / "src"
        py_dir.mkdir()
        (py_dir / "main.py").write_text("print('hello')")
        (py_dir / "utils.py").write_text("def helper(): pass")

        # Create JavaScript files
        js_dir = tmp_path / "frontend"
        js_dir.mkdir()
        (js_dir / "app.js").write_text("console.log('hello');")
        (js_dir / "index.jsx").write_text("export default App;")

        # Create TypeScript files
        (js_dir / "types.ts").write_text("interface User {}")

        # Create Go files
        go_dir = tmp_path / "backend"
        go_dir.mkdir()
        (go_dir / "main.go").write_text("package main")

        # Create excluded directories
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.js").write_text("// should be ignored")

        return tmp_path

    def test_detect_single_language(self, tmp_path):
        """Test detection with only Python files."""
        from codesage.utils.language_detector import detect_languages

        # Create only Python files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        result = detect_languages(tmp_path)

        assert len(result) == 1
        assert result[0].name == "python"
        assert result[0].file_count == 2

    def test_detect_multiple_languages(self, temp_project):
        """Test detection with multiple languages."""
        from codesage.utils.language_detector import detect_languages

        result = detect_languages(temp_project)

        # Should find Python, JavaScript, TypeScript, Go
        lang_names = [r.name for r in result]
        assert "python" in lang_names
        assert "javascript" in lang_names
        assert "typescript" in lang_names
        assert "go" in lang_names

    def test_excludes_node_modules(self, temp_project):
        """Test that node_modules is excluded."""
        from codesage.utils.language_detector import detect_languages

        result = detect_languages(temp_project)

        # Should not count the file in node_modules
        js_info = next((r for r in result if r.name == "javascript"), None)
        if js_info:
            # Should only be 2 JS files (app.js, index.jsx), not the one in node_modules
            assert js_info.file_count == 2

    def test_sorted_by_file_count(self, temp_project):
        """Test results are sorted by file count descending."""
        from codesage.utils.language_detector import detect_languages

        result = detect_languages(temp_project)

        # Verify sorted by file count descending
        counts = [r.file_count for r in result]
        assert counts == sorted(counts, reverse=True)

    def test_get_extensions_for_languages(self):
        """Test getting extensions for multiple languages."""
        from codesage.utils.language_detector import get_extensions_for_languages

        extensions = get_extensions_for_languages(["python", "javascript"])

        assert ".py" in extensions
        assert ".js" in extensions
        assert ".jsx" in extensions

    def test_get_exclude_dirs_for_languages(self):
        """Test getting exclude directories for languages."""
        from codesage.utils.language_detector import get_exclude_dirs_for_languages

        excludes = get_exclude_dirs_for_languages(["python", "javascript", "rust"])

        # Should have common excludes
        assert ".git" in excludes

        # Should have language-specific excludes
        assert "node_modules" in excludes  # JavaScript
        assert "venv" in excludes          # Python
        assert "target" in excludes        # Rust

    def test_empty_project(self, tmp_path):
        """Test detection on empty project."""
        from codesage.utils.language_detector import detect_languages

        result = detect_languages(tmp_path)

        assert len(result) == 0

    def test_only_unsupported_files(self, tmp_path):
        """Test detection with only unsupported file types."""
        from codesage.utils.language_detector import detect_languages

        (tmp_path / "readme.md").write_text("# Hello")
        (tmp_path / "config.yaml").write_text("key: value")

        result = detect_languages(tmp_path)

        assert len(result) == 0


class TestLanguageInfoDataclass:
    """Tests for LanguageInfo dataclass."""

    def test_language_info_creation(self):
        """Test LanguageInfo can be created."""
        from codesage.utils.language_detector import LanguageInfo

        info = LanguageInfo(
            name="python",
            extensions=[".py"],
            file_count=10
        )

        assert info.name == "python"
        assert info.extensions == [".py"]
        assert info.file_count == 10
