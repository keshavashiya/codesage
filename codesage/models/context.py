"""Models for context provider outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImplementationPlan:
    """Plan for implementing a task."""

    steps: List[str] = field(default_factory=list)
    estimated_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": list(self.steps),
            "estimated_files": list(self.estimated_files),
        }


@dataclass
class CodeReference:
    """Reference to relevant code."""

    file: str
    line: int
    name: Optional[str]
    element_type: str
    similarity: float
    snippet: str
    graph: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "name": self.name,
            "type": self.element_type,
            "similarity": self.similarity,
            "snippet": self.snippet,
            "graph": self.graph,
        }


@dataclass
class ImplementationContext:
    """Comprehensive context for implementing a task."""

    task: str
    implementation_plan: ImplementationPlan
    relevant_code: List[CodeReference] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    suggested_files: List[str] = field(default_factory=list)
    cross_project_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "implementation_plan": self.implementation_plan.to_dict(),
            "relevant_code": [r.to_dict() for r in self.relevant_code],
            "patterns": self.patterns,
            "dependencies": self.dependencies,
            "preferences": self.preferences,
            "security": self.security,
            "suggested_files": self.suggested_files,
            "cross_project_recommendations": self.cross_project_recommendations,
        }
