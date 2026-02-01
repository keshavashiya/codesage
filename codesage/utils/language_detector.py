"""Language detection for multi-language projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

from codesage.utils.logging import get_logger

logger = get_logger("language_detector")


@dataclass
class LanguageInfo:
    """Information about a detected language."""

    name: str
    extensions: List[str]
    file_count: int


# Mapping of file extensions to language names
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    # Python
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
}

# Supported languages (those we have parsers for)
SUPPORTED_LANGUAGES: Set[str] = {"python", "javascript", "typescript", "go", "rust"}

# Default exclude directories per language
LANGUAGE_EXCLUDE_DIRS: Dict[str, List[str]] = {
    "javascript": ["node_modules", "dist", "build", ".next"],
    "typescript": ["node_modules", "dist", "build", ".next"],
    "go": ["vendor"],
    "rust": ["target"],
    "python": ["venv", "env", ".venv", ".env", "__pycache__", ".pytest_cache"],
}

# Default include extensions per language
LANGUAGE_EXTENSIONS: Dict[str, List[str]] = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx", ".mts", ".cts"],
    "go": [".go"],
    "rust": [".rs"],
}


def detect_languages(
    project_path: Path,
    exclude_dirs: List[str] | None = None,
    max_files: int = 10000,
) -> List[LanguageInfo]:
    """Detect programming languages used in a project.

    Scans the project directory for source files and determines
    which languages are present based on file extensions.

    Args:
        project_path: Path to the project root
        exclude_dirs: Directories to skip (default: common build/vendor dirs)
        max_files: Maximum files to scan (prevents hanging on huge repos)

    Returns:
        List of LanguageInfo sorted by file count (descending)
    """
    if exclude_dirs is None:
        exclude_dirs = [
            # Common across languages
            ".git", ".svn", ".hg",
            ".codesage", ".idea", ".vscode",
            # Python
            "venv", "env", ".venv", ".env",
            "__pycache__", ".pytest_cache", ".mypy_cache",
            "*.egg-info", "build", "dist",
            # JavaScript/TypeScript
            "node_modules", ".next", ".nuxt",
            # Go
            "vendor",
            # Rust
            "target",
        ]

    project_path = Path(project_path).resolve()
    extension_counts: Counter[str] = Counter()
    files_scanned = 0

    try:
        for path in project_path.rglob("*"):
            if files_scanned >= max_files:
                logger.warning(f"Reached max files limit ({max_files}), stopping scan")
                break

            if not path.is_file():
                continue

            # Skip excluded directories
            if any(excluded in path.parts for excluded in exclude_dirs):
                continue

            ext = path.suffix.lower()
            if ext in EXTENSION_TO_LANGUAGE:
                extension_counts[ext] += 1
                files_scanned += 1

    except PermissionError:
        logger.warning(f"Permission denied accessing some files in {project_path}")
    except Exception as e:
        logger.error(f"Error scanning project: {e}")

    # Group by language
    language_files: Dict[str, Dict[str, int]] = {}
    for ext, count in extension_counts.items():
        lang = EXTENSION_TO_LANGUAGE[ext]
        if lang not in SUPPORTED_LANGUAGES:
            continue
        if lang not in language_files:
            language_files[lang] = {}
        language_files[lang][ext] = count

    # Build result
    results = []
    for lang, ext_counts in language_files.items():
        total_files = sum(ext_counts.values())
        extensions = sorted(ext_counts.keys())
        results.append(LanguageInfo(
            name=lang,
            extensions=extensions,
            file_count=total_files,
        ))

    # Sort by file count descending
    results.sort(key=lambda x: x.file_count, reverse=True)

    return results


def get_extensions_for_languages(languages: List[str]) -> List[str]:
    """Get all file extensions for a list of languages.

    Args:
        languages: List of language names

    Returns:
        Combined list of file extensions
    """
    extensions = []
    for lang in languages:
        if lang in LANGUAGE_EXTENSIONS:
            extensions.extend(LANGUAGE_EXTENSIONS[lang])
    return list(set(extensions))


def get_exclude_dirs_for_languages(languages: List[str]) -> List[str]:
    """Get combined exclude directories for a list of languages.

    Args:
        languages: List of language names

    Returns:
        Combined list of directories to exclude
    """
    # Start with common excludes
    excludes = {".git", ".svn", ".hg", ".codesage", ".idea", ".vscode"}

    for lang in languages:
        if lang in LANGUAGE_EXCLUDE_DIRS:
            excludes.update(LANGUAGE_EXCLUDE_DIRS[lang])

    return sorted(excludes)
