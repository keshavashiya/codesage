"""Abstract base class for language parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from codesage.models.code_element import CodeElement


class BaseParser(ABC):
    """Abstract base class for language parsers.

    Implement this class to add support for a new programming language.
    Register the parser with the ParserRegistry to enable automatic
    file type detection.
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """Get the language identifier.

        Returns:
            Language name (e.g., 'python', 'javascript', 'typescript')
        """
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of extensions including the dot (e.g., ['.py'])
        """
        pass

    @abstractmethod
    def parse_file(self, file_path: Path) -> List[CodeElement]:
        """Parse a file and extract code elements.

        Args:
            file_path: Path to the file to parse

        Returns:
            List of extracted code elements (functions, classes, etc.)
        """
        pass

    @abstractmethod
    def parse_code(self, code: str, file_path: Path) -> List[CodeElement]:
        """Parse code string and extract code elements.

        Args:
            code: Source code string
            file_path: Path for reference (used in CodeElement)

        Returns:
            List of extracted code elements
        """
        pass

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to check

        Returns:
            True if this parser supports the file type
        """
        return file_path.suffix.lower() in self.file_extensions
