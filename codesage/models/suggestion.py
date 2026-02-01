"""Data models for suggestions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    # Graph context (from KuzuDB)
    callers: List[Dict[str, Any]] = field(default_factory=list)
    callees: List[Dict[str, Any]] = field(default_factory=list)
    superclasses: List[Dict[str, Any]] = field(default_factory=list)
    subclasses: List[Dict[str, Any]] = field(default_factory=list)

    # Pattern context (from memory system)
    matching_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
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

        # Include graph context if present
        if self.callers:
            result["callers"] = self.callers
        if self.callees:
            result["callees"] = self.callees
        if self.superclasses:
            result["superclasses"] = self.superclasses
        if self.subclasses:
            result["subclasses"] = self.subclasses
        if self.matching_patterns:
            result["matching_patterns"] = self.matching_patterns

        return result

    def has_graph_context(self) -> bool:
        """Check if suggestion has graph context."""
        return bool(self.callers or self.callees or self.superclasses or self.subclasses)


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
