"""Models for code smell detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CodeSmell:
    """Represents a detected code smell."""

    type: str
    file: str
    line: Optional[int]
    element: Optional[str]
    severity: str
    confidence: float
    message: str
    suggestion: Optional[str] = None
    pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "file": self.file,
            "line": self.line,
            "element": self.element,
            "severity": self.severity,
            "confidence": self.confidence,
            "message": self.message,
            "suggestion": self.suggestion,
            "pattern": self.pattern,
        }
