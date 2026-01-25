"""Parsers package - Language parsing for code intelligence."""

from codesage.parsers.base import BaseParser
from codesage.parsers.registry import ParserRegistry
from codesage.parsers.python_parser import PythonParser

# Auto-register the Python parser
_python_parser = PythonParser()
ParserRegistry.register(_python_parser)

__all__ = ["BaseParser", "ParserRegistry", "PythonParser"]
