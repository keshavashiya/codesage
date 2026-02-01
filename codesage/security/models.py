"""Security data models.

Contains all data structures used by the security module.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Pattern


class Severity(str, Enum):
    """Severity levels for security findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def __lt__(self, other: "Severity") -> bool:
        order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: "Severity") -> bool:
        return self == other or self < other


@dataclass
class SecurityRule:
    """Definition of a security rule for pattern detection."""

    id: str
    name: str
    pattern: str
    severity: Severity
    message: str
    description: str = ""
    category: str = "general"
    cwe_id: Optional[str] = None
    fix_suggestion: Optional[str] = None
    enabled: bool = True
    file_patterns: Optional[List[str]] = None  # e.g. ["*.js", "*.html"]
    _compiled_pattern: Optional[Pattern] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Compile the regex pattern."""
        self._compile_pattern()

    def _compile_pattern(self) -> None:
        """Compile the regex pattern."""
        try:
            self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern for rule {self.id}: {e}")

    @property
    def compiled_pattern(self) -> Pattern:
        """Get the compiled regex pattern."""
        if self._compiled_pattern is None:
            self._compile_pattern()
        return self._compiled_pattern

    def matches(self, content: str) -> List[re.Match]:
        """Find all matches of this rule in the content."""
        if not self.enabled:
            return []
        return list(self.compiled_pattern.finditer(content))

    def applies_to_file(self, file_path: Path) -> bool:
        """Check if this rule should be applied to the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if rule applies, False if skipped due to file_patterns
        """
        if not self.file_patterns:
            return True  # No restrictions, applies to all files

        import fnmatch
        filename = file_path.name
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.file_patterns)


@dataclass
class SecurityFinding:
    """A single security finding from a scan."""

    rule: SecurityRule
    file: Path
    line_number: int
    line_content: str
    match_text: str
    column_start: int = 0
    column_end: int = 0
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)

    @property
    def severity(self) -> Severity:
        """Get the severity from the rule."""
        return self.rule.severity

    @property
    def location(self) -> str:
        """Get formatted location string."""
        return f"{self.file}:{self.line_number}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            "rule_id": self.rule.id,
            "rule_name": self.rule.name,
            "severity": self.severity.value,
            "file": str(self.file),
            "line": self.line_number,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "message": self.rule.message,
            "match": self.match_text,
            "fix_suggestion": self.rule.fix_suggestion,
            "cwe_id": self.rule.cwe_id,
        }


@dataclass
class SecurityReport:
    """Complete security scan report."""

    findings: List[SecurityFinding] = field(default_factory=list)
    files_scanned: int = 0
    scan_duration_ms: float = 0
    scanned_at: datetime = field(default_factory=datetime.now)
    severity_threshold: Severity = Severity.LOW

    @property
    def critical_count(self) -> int:
        return len([f for f in self.findings if f.severity == Severity.CRITICAL])

    @property
    def high_count(self) -> int:
        return len([f for f in self.findings if f.severity == Severity.HIGH])

    @property
    def medium_count(self) -> int:
        return len([f for f in self.findings if f.severity == Severity.MEDIUM])

    @property
    def low_count(self) -> int:
        return len([f for f in self.findings if f.severity == Severity.LOW])

    @property
    def info_count(self) -> int:
        return len([f for f in self.findings if f.severity == Severity.INFO])

    @property
    def total_count(self) -> int:
        return len(self.findings)

    @property
    def has_blocking_issues(self) -> bool:
        return self.critical_count > 0

    @property
    def is_clean(self) -> bool:
        return self.total_count == 0

    def get_findings_by_severity(self, severity: Severity) -> List[SecurityFinding]:
        """Get all findings of a specific severity."""
        return [f for f in self.findings if f.severity == severity]

    def get_findings_by_file(self, file_path: Path) -> List[SecurityFinding]:
        """Get all findings for a specific file."""
        return [f for f in self.findings if f.file == file_path]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            "summary": {
                "files_scanned": self.files_scanned,
                "total_findings": self.total_count,
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "info": self.info_count,
                "scan_duration_ms": self.scan_duration_ms,
                "scanned_at": self.scanned_at.isoformat(),
                "has_blocking_issues": self.has_blocking_issues,
            },
            "findings": [f.to_dict() for f in self.findings],
        }
