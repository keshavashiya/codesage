"""Parsers package - Language parsing for code intelligence."""

from codesage.parsers.base import BaseParser
from codesage.parsers.registry import ParserRegistry
from codesage.parsers.python_parser import PythonParser

# Auto-register the Python parser (always available, uses built-in ast)
_python_parser = PythonParser()
ParserRegistry.register(_python_parser)

# Conditionally register Tree-sitter parsers for other languages
# These require the optional 'multi-language' dependencies
_TREESITTER_LANGUAGES = ["javascript", "typescript", "go", "rust"]

try:
    from codesage.parsers.treesitter_parser import TreeSitterParser

    for _lang in _TREESITTER_LANGUAGES:
        try:
            _parser = TreeSitterParser(_lang)
            ParserRegistry.register(_parser)
        except ImportError:
            # Skip if specific language grammar not installed
            pass
        except Exception:
            # Skip on any other initialization error
            pass
except ImportError:
    # tree-sitter not installed, skip all multi-language parsers
    pass

__all__ = ["BaseParser", "ParserRegistry", "PythonParser"]

