"""Review data models.

Contains all data structures used by the review module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class IssueSeverity(str, Enum):
    """Severity levels for review issues."""

    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    PRAISE = "praise"


@dataclass
class FileChange:
    """Represents changes to a single file."""

    path: Path
    status: str  # A=added, M=modified, D=deleted, R=renamed
    additions: int = 0
    deletions: int = 0
    diff: str = ""
    old_path: Optional[Path] = None
    content: Optional[str] = None  # Full file content for single file review


@dataclass
class ReviewIssue:
    """A single issue found during review."""

    severity: IssueSeverity
    file: Path
    line: Optional[int]
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "file": str(self.file),
            "line": self.line,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class ReviewResult:
    """Complete code review result."""

    files_changed: List[FileChange] = field(default_factory=list)
    issues: List[ReviewIssue] = field(default_factory=list)
    summary: str = ""
    pr_description: str = ""
    reviewed_at: datetime = field(default_factory=datetime.now)

    @property
    def critical_count(self) -> int:
        return len([i for i in self.issues if i.severity == IssueSeverity.CRITICAL])

    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == IssueSeverity.WARNING])

    @property
    def suggestion_count(self) -> int:
        return len([i for i in self.issues if i.severity == IssueSeverity.SUGGESTION])

    @property
    def praise_count(self) -> int:
        return len([i for i in self.issues if i.severity == IssueSeverity.PRAISE])

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files_changed)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files_changed)

    @property
    def has_blocking_issues(self) -> bool:
        return self.critical_count > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "files_changed": len(self.files_changed),
            "additions": self.total_additions,
            "deletions": self.total_deletions,
            "critical": self.critical_count,
            "warnings": self.warning_count,
            "suggestions": self.suggestion_count,
            "issues": [i.to_dict() for i in self.issues],
            "pr_description": self.pr_description,
        }
