"""Shared tree-sitter utilities: availability probe and cached parser factory.

All modules that need tree-sitter parsers should import from here rather than
duplicating the availability check and parser-initialization logic.

Usage::

    from codesage.utils.treesitter_utils import TREESITTER_AVAILABLE, get_parser

    if TREESITTER_AVAILABLE:
        parser = get_parser("rust")   # returns tree_sitter.Parser or None
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional

from codesage.utils.logging import get_logger

logger = get_logger("utils.treesitter")

# ---------------------------------------------------------------------------
# Single availability probe for the whole process
# ---------------------------------------------------------------------------

TREESITTER_AVAILABLE: bool = importlib.util.find_spec("tree_sitter") is not None

# ---------------------------------------------------------------------------
# Module-level parser cache (language → Parser instance or None)
# Shared across ALL callers so each language is initialized at most once.
# ---------------------------------------------------------------------------

_PARSER_CACHE: Dict[str, Any] = {}


def get_parser(language: str) -> Optional[Any]:
    """Return a cached tree-sitter Parser for *language*.

    Supported languages: ``rust``, ``go``, ``javascript``, ``typescript``.

    Returns ``None`` if tree-sitter is not installed or the grammar package
    for the requested language is missing.  Failures are cached so subsequent
    calls for the same language return immediately without retrying the import.
    """
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    if not TREESITTER_AVAILABLE:
        _PARSER_CACHE[language] = None
        return None

    parser: Optional[Any] = None
    try:
        import tree_sitter

        if language == "rust":
            import tree_sitter_rust
            ts_lang = tree_sitter.Language(tree_sitter_rust.language())
        elif language == "go":
            import tree_sitter_go
            ts_lang = tree_sitter.Language(tree_sitter_go.language())
        elif language == "javascript":
            import tree_sitter_javascript
            ts_lang = tree_sitter.Language(tree_sitter_javascript.language())
        elif language == "typescript":
            import tree_sitter_typescript
            # TSX grammar handles both .ts and .tsx files
            ts_lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
        else:
            logger.debug(f"No tree-sitter grammar registered for language '{language}'")
            _PARSER_CACHE[language] = None
            return None

        parser = tree_sitter.Parser(ts_lang)
        logger.debug(f"tree-sitter parser initialized for {language}")

    except ImportError as exc:
        logger.debug(
            f"tree-sitter grammar package for '{language}' not installed: {exc}"
        )
        parser = None
    except Exception as exc:
        logger.debug(f"Failed to initialize tree-sitter parser for '{language}': {exc}")
        parser = None

    _PARSER_CACHE[language] = parser
    return parser
