"""Parser registry for multi-language support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List

from codesage.parsers.base import BaseParser


class ParserRegistry:
    """Registry for language parsers.

    Provides automatic parser discovery based on file extensions.
    New parsers can be registered at runtime.
    """

    _parsers: Dict[str, BaseParser] = {}
    _extension_map: Dict[str, str] = {}

    @classmethod
    def register(cls, parser: BaseParser) -> None:
        """Register a parser instance.

        Args:
            parser: Parser instance to register
        """
        cls._parsers[parser.language] = parser
        for ext in parser.file_extensions:
            cls._extension_map[ext.lower()] = parser.language

    @classmethod
    def get_parser(cls, language: str) -> Optional[BaseParser]:
        """Get a parser by language name.

        Args:
            language: Language identifier (e.g., 'python')

        Returns:
            Parser for the language, or None if not found
        """
        return cls._parsers.get(language)

    @classmethod
    def get_parser_for_file(cls, file_path: Path) -> Optional[BaseParser]:
        """Get appropriate parser for a file based on extension.

        Args:
            file_path: Path to the file

        Returns:
            Parser that can handle this file, or None
        """
        ext = file_path.suffix.lower()
        language = cls._extension_map.get(ext)
        return cls._parsers.get(language) if language else None

    @classmethod
    def supported_languages(cls) -> List[str]:
        """Get list of supported language names.

        Returns:
            List of registered language identifiers
        """
        return list(cls._parsers.keys())

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions.

        Returns:
            List of registered file extensions
        """
        return list(cls._extension_map.keys())

    @classmethod
    def can_parse(cls, file_path: Path) -> bool:
        """Check if any registered parser can handle the file.

        Args:
            file_path: Path to check

        Returns:
            True if a parser is available for this file type
        """
        return cls.get_parser_for_file(file_path) is not None

    @classmethod
    def clear(cls) -> None:
        """Clear all registered parsers (useful for testing)."""
        cls._parsers.clear()
        cls._extension_map.clear()
