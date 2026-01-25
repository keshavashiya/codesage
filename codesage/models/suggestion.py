"""Data models for suggestions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Suggestion:
    """Represents a code suggestion result."""

    file: Path
    line: int
    code: str
    similarity: float
    language: str
    element_type: str  # function, class, method
    name: Optional[str] = None
    explanation: Optional[str] = None
    docstring: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file": str(self.file),
            "line": self.line,
            "code": self.code,
            "similarity": self.similarity,
            "language": self.language,
            "element_type": self.element_type,
            "name": self.name,
            "explanation": self.explanation,
            "docstring": self.docstring,
        }


@dataclass
class Pattern:
    """Represents a detected code pattern."""

    name: str
    description: str
    occurrences: int
    examples: list
    category: str = "general"
    confidence: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "occurrences": self.occurrences,
            "examples": self.examples,
            "category": self.category,
            "confidence": self.confidence,
        }
