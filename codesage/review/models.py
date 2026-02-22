"""Review data models.

Contains all data structures used by the review module.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class IssueSeverity(str, Enum):
    """Severity levels for review issues."""

    CRITICAL = "critical"
    HIGH = "high"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    PRAISE = "praise"

    def __lt__(self, other: "IssueSeverity") -> bool:
        order = [
            IssueSeverity.PRAISE,
            IssueSeverity.SUGGESTION,
            IssueSeverity.WARNING,
            IssueSeverity.HIGH,
            IssueSeverity.CRITICAL,
        ]
        return order.index(self) < order.index(other)

    def __le__(self, other: "IssueSeverity") -> bool:
        return self == other or self < other

    def __gt__(self, other: "IssueSeverity") -> bool:
        return not self <= other

    def __ge__(self, other: "IssueSeverity") -> bool:
        return not self < other


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


# --- Unified review pipeline models ---

_SEVERITY_THRESHOLD_ORDER = ["praise", "suggestion", "warning", "high", "critical"]


@dataclass
class ReviewFinding:
    """Unified finding from any review source.

    Normalizes SecurityFinding, ReviewIssue, and code smell results
    into a single structure for the combined review pipeline.
    """

    severity: str  # critical, high, warning, suggestion, praise
    category: str  # security, smell, pattern, duplication, practice, structure, naming, complexity
    file: Path
    line: Optional[int] = None
    rule_id: Optional[str] = None
    message: str = ""
    suggestion: Optional[str] = None
    source: str = "static"  # static, semantic, llm
    suppressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity,
            "category": self.category,
            "file": str(self.file),
            "line": self.line,
            "rule_id": self.rule_id,
            "message": self.message,
            "suggestion": self.suggestion,
            "source": self.source,
            "suppressed": self.suppressed,
        }

    def to_sarif_result(self) -> Dict[str, Any]:
        """Convert to a SARIF result object."""
        level_map = {
            "critical": "error",
            "high": "error",
            "warning": "warning",
            "suggestion": "note",
            "praise": "note",
        }
        result: Dict[str, Any] = {
            "ruleId": self.rule_id or f"{self.category}-finding",
            "level": level_map.get(self.severity, "warning"),
            "message": {"text": self.message},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": str(self.file)},
                    }
                }
            ],
        }
        if self.line is not None:
            result["locations"][0]["physicalLocation"]["region"] = {
                "startLine": self.line,
            }
        return result


@dataclass
class UnifiedReviewResult:
    """Result from the unified review pipeline.

    Aggregates findings from all sources: security scanner, static checks,
    pattern deviation, semantic similarity, and LLM synthesis.
    """

    findings: List[ReviewFinding] = field(default_factory=list)
    files_changed: List[FileChange] = field(default_factory=list)
    summary: str = ""
    mode: str = "fast"  # fast or full
    duration_ms: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    stages_skipped: List[str] = field(default_factory=list)
    reviewed_at: datetime = field(default_factory=datetime.now)

    @property
    def active_findings(self) -> List[ReviewFinding]:
        """Findings that are not suppressed."""
        return [f for f in self.findings if not f.suppressed]

    @property
    def suppressed_count(self) -> int:
        return len([f for f in self.findings if f.suppressed])

    def count_by_severity(self, severity: str) -> int:
        return len([f for f in self.active_findings if f.severity == severity])

    @property
    def critical_count(self) -> int:
        return self.count_by_severity("critical")

    @property
    def high_count(self) -> int:
        return self.count_by_severity("high")

    @property
    def warning_count(self) -> int:
        return self.count_by_severity("warning")

    @property
    def suggestion_count(self) -> int:
        return self.count_by_severity("suggestion")

    @property
    def praise_count(self) -> int:
        return self.count_by_severity("praise")

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files_changed)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files_changed)

    def has_blocking_issues(self, threshold: str = "critical") -> bool:
        """Check if there are findings at or above the given severity threshold."""
        if threshold not in _SEVERITY_THRESHOLD_ORDER:
            threshold = "critical"
        threshold_idx = _SEVERITY_THRESHOLD_ORDER.index(threshold)
        for f in self.active_findings:
            if f.severity in _SEVERITY_THRESHOLD_ORDER:
                f_idx = _SEVERITY_THRESHOLD_ORDER.index(f.severity)
                if f_idx >= threshold_idx:
                    return True
        return False

    @property
    def exit_code(self) -> int:
        """Return 0 for clean, 1 for blocking issues (critical only by default)."""
        return 1 if self.has_blocking_issues("critical") else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "summary": {
                "files_changed": len(self.files_changed),
                "additions": self.total_additions,
                "deletions": self.total_deletions,
                "critical": self.critical_count,
                "high": self.high_count,
                "warnings": self.warning_count,
                "suggestions": self.suggestion_count,
                "praise": self.praise_count,
                "suppressed": self.suppressed_count,
                "mode": self.mode,
                "duration_ms": round(self.duration_ms, 1),
                "reviewed_at": self.reviewed_at.isoformat(),
            },
            "findings": [f.to_dict() for f in self.active_findings],
            "changed_files": [
                {
                    "path": str(fc.path),
                    "status": fc.status,
                    "additions": fc.additions,
                    "deletions": fc.deletions,
                }
                for fc in self.files_changed
            ],
            "metadata": {
                "stage_timings": self.stage_timings,
                "stages_skipped": self.stages_skipped,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_sarif(self) -> Dict[str, Any]:
        """Convert to SARIF v2.1.0 format for GitHub Actions / VS Code."""
        rules_seen: Dict[str, Dict[str, Any]] = {}
        results = []

        for f in self.active_findings:
            sarif_result = f.to_sarif_result()
            results.append(sarif_result)

            rule_id = sarif_result["ruleId"]
            if rule_id not in rules_seen:
                rules_seen[rule_id] = {
                    "id": rule_id,
                    "shortDescription": {"text": f.message[:200]},
                    "defaultConfiguration": {"level": sarif_result["level"]},
                }
                if f.suggestion:
                    rules_seen[rule_id]["helpUri"] = ""
                    rules_seen[rule_id]["help"] = {"text": f.suggestion}

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "CodeSage",
                            "version": "0.3.2",
                            "informationUri": "https://github.com/keshavashiya/codesage",
                            "rules": list(rules_seen.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def to_sarif_json(self, indent: int = 2) -> str:
        """Serialize SARIF to JSON string."""
        return json.dumps(self.to_sarif(), indent=indent)
