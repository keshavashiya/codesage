"""Data models for code elements."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import hashlib


@dataclass
class CodeElement:
    """Represents a code element (function, class, method, etc.)."""

    id: str
    file: Path
    type: str  # "function", "class", "method"
    name: Optional[str]
    code: str
    language: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parameters: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        file: Path,
        type: str,
        code: str,
        language: str,
        line_start: int,
        line_end: int,
        name: Optional[str] = None,
        docstring: Optional[str] = None,
        signature: Optional[str] = None,
        parameters: Optional[List[str]] = None,
    ) -> "CodeElement":
        """Create a new code element with auto-generated ID."""
        # Generate ID from file path, line numbers, and code hash
        id_str = f"{file}:{line_start}:{line_end}:{hash(code)}"
        element_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

        return cls(
            id=element_id,
            file=file,
            type=type,
            name=name,
            code=code,
            language=language,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            signature=signature,
            parameters=parameters or [],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "file": str(self.file),
            "type": self.type,
            "name": self.name,
            "code": self.code,
            "language": self.language,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "docstring": self.docstring,
            "signature": self.signature,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeElement":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            file=Path(data["file"]),
            type=data["type"],
            name=data.get("name"),
            code=data["code"],
            language=data["language"],
            line_start=data["line_start"],
            line_end=data["line_end"],
            docstring=data.get("docstring"),
            signature=data.get("signature"),
            parameters=data.get("parameters", []),
        )

    def get_embedding_text(self) -> str:
        """Get text representation for embedding generation."""
        parts = []

        if self.name:
            parts.append(f"{self.type}: {self.name}")

        if self.docstring:
            parts.append(f"Description: {self.docstring}")

        if self.signature:
            parts.append(f"Signature: {self.signature}")

        parts.append(f"Code:\n{self.code}")

        return "\n".join(parts)
