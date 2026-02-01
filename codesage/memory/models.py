"""Data models for the Developer Memory System.

This module defines all data structures used for storing and managing
developer patterns, preferences, and code style information.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PatternCategory(Enum):
    """Categories of developer coding patterns."""

    NAMING = "naming"
    DOCSTRING = "docstring"
    TYPING = "typing"
    STRUCTURE = "structure"
    IMPORTS = "imports"
    ERROR_HANDLING = "error_handling"
    LIBRARY = "library"
    TESTING = "testing"
    FORMATTING = "formatting"


class StructureType(Enum):
    """Types of code structures tracked."""

    CLASS_HIERARCHY = "class_hierarchy"
    CALL_PATTERN = "call_pattern"
    IMPORT_GROUP = "import_group"
    MODULE_LAYOUT = "module_layout"
    TEST_PATTERN = "test_pattern"
    ERROR_PATTERN = "error_pattern"


class RelationshipType(Enum):
    """Types of relationships between patterns."""

    CO_OCCURS = "CO_OCCURS"
    LEARNED_FROM = "LEARNED_FROM"
    SIMILAR_TO = "SIMILAR_TO"
    PREFERS = "PREFERS"
    EVOLVES_TO = "EVOLVES_TO"


@dataclass
class LearnedPattern:
    """Represents a coding pattern learned from developer's code.

    Attributes:
        id: Unique identifier for the pattern.
        name: Human-readable name of the pattern.
        category: Category of the pattern (naming, docstring, etc.).
        description: Detailed description of the pattern.
        pattern_text: The actual pattern or regex.
        examples: List of code examples demonstrating the pattern.
        occurrence_count: Number of times pattern was observed.
        confidence_score: Confidence in the pattern (0.0-1.0).
        first_seen: When the pattern was first observed.
        last_seen: When the pattern was last observed.
        co_occurring_patterns: IDs of patterns that appear together.
        source_projects: Names of projects where pattern was found.
        embedding: Vector embedding for semantic search.
    """

    id: str
    name: str
    category: PatternCategory
    description: str
    pattern_text: str
    examples: List[str] = field(default_factory=list)

    # Metrics
    occurrence_count: int = 1
    confidence_score: float = 0.5
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    # Graph relationships (populated from KuzuDB)
    co_occurring_patterns: List[str] = field(default_factory=list)
    source_projects: List[str] = field(default_factory=list)

    # Embedding for semantic search
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        now = datetime.now()
        if self.first_seen is None:
            self.first_seen = now
        if self.last_seen is None:
            self.last_seen = now

    @classmethod
    def create(
        cls,
        name: str,
        category: PatternCategory,
        description: str,
        pattern_text: str,
        **kwargs,
    ) -> "LearnedPattern":
        """Factory method to create a pattern with auto-generated ID.

        Args:
            name: Human-readable name.
            category: Pattern category.
            description: Pattern description.
            pattern_text: The pattern or regex.
            **kwargs: Additional attributes.

        Returns:
            New LearnedPattern instance.
        """
        id_str = f"{category.value}:{name}:{pattern_text}"
        pattern_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

        return cls(
            id=pattern_id,
            name=name,
            category=category,
            description=description,
            pattern_text=pattern_text,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pattern to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "pattern_text": self.pattern_text,
            "examples": self.examples,
            "occurrence_count": self.occurrence_count,
            "confidence_score": self.confidence_score,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "co_occurring_patterns": self.co_occurring_patterns,
            "source_projects": self.source_projects,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedPattern":
        """Deserialize pattern from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            LearnedPattern instance.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            category=PatternCategory(data["category"]),
            description=data["description"],
            pattern_text=data["pattern_text"],
            examples=data.get("examples", []),
            occurrence_count=data.get("occurrence_count", 1),
            confidence_score=data.get("confidence_score", 0.5),
            first_seen=datetime.fromisoformat(data["first_seen"])
            if data.get("first_seen")
            else None,
            last_seen=datetime.fromisoformat(data["last_seen"])
            if data.get("last_seen")
            else None,
            co_occurring_patterns=data.get("co_occurring_patterns", []),
            source_projects=data.get("source_projects", []),
        )


@dataclass
class CodeStructure:
    """Represents a preferred code structure pattern.

    Attributes:
        id: Unique identifier.
        structure_type: Type of structure (class_hierarchy, call_pattern, etc.).
        name: Human-readable name.
        description: Description of the structure.
        example_code: Example code demonstrating the structure.
        occurrence_count: Number of times observed.
        confidence: Confidence in preference (0.0-1.0).
        first_seen: When first observed.
        last_seen: When last observed.
    """

    id: str
    structure_type: StructureType
    name: str
    description: str
    example_code: str
    occurrence_count: int = 1
    confidence: float = 0.5
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        now = datetime.now()
        if self.first_seen is None:
            self.first_seen = now
        if self.last_seen is None:
            self.last_seen = now

    @classmethod
    def create(
        cls,
        structure_type: StructureType,
        name: str,
        description: str,
        example_code: str,
        **kwargs,
    ) -> "CodeStructure":
        """Factory method to create a structure with auto-generated ID.

        Args:
            structure_type: Type of structure.
            name: Human-readable name.
            description: Structure description.
            example_code: Example code.
            **kwargs: Additional attributes.

        Returns:
            New CodeStructure instance.
        """
        id_str = f"{structure_type.value}:{name}:{hash(example_code)}"
        structure_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

        return cls(
            id=structure_id,
            structure_type=structure_type,
            name=name,
            description=description,
            example_code=example_code,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize structure to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "id": self.id,
            "structure_type": self.structure_type.value,
            "name": self.name,
            "description": self.description,
            "example_code": self.example_code,
            "occurrence_count": self.occurrence_count,
            "confidence": self.confidence,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeStructure":
        """Deserialize structure from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            CodeStructure instance.
        """
        return cls(
            id=data["id"],
            structure_type=StructureType(data["structure_type"]),
            name=data["name"],
            description=data["description"],
            example_code=data["example_code"],
            occurrence_count=data.get("occurrence_count", 1),
            confidence=data.get("confidence", 0.5),
            first_seen=datetime.fromisoformat(data["first_seen"])
            if data.get("first_seen")
            else None,
            last_seen=datetime.fromisoformat(data["last_seen"])
            if data.get("last_seen")
            else None,
        )


@dataclass
class ProjectInfo:
    """Information about an indexed project.

    Attributes:
        id: Unique identifier.
        name: Project name.
        path: Absolute path to project.
        language: Primary programming language.
        total_files: Number of files indexed.
        total_elements: Number of code elements indexed.
        first_indexed: When project was first indexed.
        last_indexed: When project was last indexed.
        patterns_learned: Number of patterns learned from this project.
    """

    id: str
    name: str
    path: Path
    language: str = "python"
    total_files: int = 0
    total_elements: int = 0
    first_indexed: Optional[datetime] = None
    last_indexed: Optional[datetime] = None
    patterns_learned: int = 0

    def __post_init__(self):
        """Initialize timestamps and ensure path is Path object."""
        now = datetime.now()
        if self.first_indexed is None:
            self.first_indexed = now
        if self.last_indexed is None:
            self.last_indexed = now
        if isinstance(self.path, str):
            self.path = Path(self.path)

    @classmethod
    def create(cls, name: str, path: Path, **kwargs) -> "ProjectInfo":
        """Factory method to create project info with auto-generated ID.

        Args:
            name: Project name.
            path: Project path.
            **kwargs: Additional attributes.

        Returns:
            New ProjectInfo instance.
        """
        id_str = f"{name}:{str(path)}"
        project_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

        return cls(id=project_id, name=name, path=path, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize project info to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.path),
            "language": self.language,
            "total_files": self.total_files,
            "total_elements": self.total_elements,
            "first_indexed": self.first_indexed.isoformat()
            if self.first_indexed
            else None,
            "last_indexed": self.last_indexed.isoformat()
            if self.last_indexed
            else None,
            "patterns_learned": self.patterns_learned,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectInfo":
        """Deserialize project info from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ProjectInfo instance.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            path=Path(data["path"]),
            language=data.get("language", "python"),
            total_files=data.get("total_files", 0),
            total_elements=data.get("total_elements", 0),
            first_indexed=datetime.fromisoformat(data["first_indexed"])
            if data.get("first_indexed")
            else None,
            last_indexed=datetime.fromisoformat(data["last_indexed"])
            if data.get("last_indexed")
            else None,
            patterns_learned=data.get("patterns_learned", 0),
        )


@dataclass
class DeveloperPreference:
    """A developer preference or setting.

    Attributes:
        key: Preference key.
        value: Preference value.
        category: Category of preference.
        description: Description of the preference.
        updated_at: When preference was last updated.
    """

    key: str
    value: Any
    category: str = "general"
    description: str = ""
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize preference to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "description": self.description,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeveloperPreference":
        """Deserialize preference from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            DeveloperPreference instance.
        """
        return cls(
            key=data["key"],
            value=data["value"],
            category=data.get("category", "general"),
            description=data.get("description", ""),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
        )


@dataclass
class InteractionRecord:
    """Record of a developer interaction.

    Attributes:
        id: Unique identifier.
        interaction_type: Type of interaction (query, suggestion, etc.).
        project_name: Project where interaction occurred.
        query: Query or input text.
        response: Response or output.
        accepted: Whether suggestion was accepted.
        feedback: Optional feedback from developer.
        timestamp: When interaction occurred.
        metadata: Additional metadata.
    """

    id: str
    interaction_type: str
    project_name: str
    query: str = ""
    response: str = ""
    accepted: Optional[bool] = None
    feedback: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @classmethod
    def create(
        cls,
        interaction_type: str,
        project_name: str,
        **kwargs,
    ) -> "InteractionRecord":
        """Factory method to create an interaction record with auto-generated ID.

        Args:
            interaction_type: Type of interaction.
            project_name: Project name.
            **kwargs: Additional attributes.

        Returns:
            New InteractionRecord instance.
        """
        timestamp = datetime.now()
        id_str = f"{interaction_type}:{project_name}:{timestamp.isoformat()}"
        record_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

        return cls(
            id=record_id,
            interaction_type=interaction_type,
            project_name=project_name,
            timestamp=timestamp,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize interaction record to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "id": self.id,
            "interaction_type": self.interaction_type,
            "project_name": self.project_name,
            "query": self.query,
            "response": self.response,
            "accepted": self.accepted,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionRecord":
        """Deserialize interaction record from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            InteractionRecord instance.
        """
        return cls(
            id=data["id"],
            interaction_type=data["interaction_type"],
            project_name=data["project_name"],
            query=data.get("query", ""),
            response=data.get("response", ""),
            accepted=data.get("accepted"),
            feedback=data.get("feedback", ""),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else None,
            metadata=data.get("metadata", {}),
        )
