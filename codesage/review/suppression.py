"""Inline and project-level finding suppression.

Supports:
- `# codesage:ignore` — suppress all findings on this line
- `# codesage:ignore RULE-ID` — suppress specific rule on this line
- `# codesage:ignore-next-line` — suppress all findings on the next line
- `# codesage:ignore-next-line RULE-ID` — suppress specific rule on the next line
- `.codesageignore` file — project-wide suppressions with glob patterns
"""

import fnmatch
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from codesage.review.models import ReviewFinding

logger = logging.getLogger(__name__)

# Matches: # / // / -- codesage:ignore [RULE-ID [RULE-ID ...]]
_IGNORE_RE = re.compile(
    r"(?:#|//|--)\s*codesage:ignore(?:-next-line)?\s*([\w\-,\s]*)", re.IGNORECASE
)
_IGNORE_NEXT_RE = re.compile(
    r"(?:#|//|--)\s*codesage:ignore-next-line\s*([\w\-,\s]*)", re.IGNORECASE
)


class SuppressionParser:
    """Parse inline suppression comments from source files."""

    def parse_file(self, file_path: Path) -> Dict[int, Set[str]]:
        """Parse a file for suppression comments.

        Args:
            file_path: Path to the source file.

        Returns:
            Mapping of line_number -> set of suppressed rule IDs.
            An empty set means suppress ALL rules on that line.
        """
        try:
            content = file_path.read_text(errors="replace")
        except Exception:
            return {}

        return self.parse_content(content)

    def parse_content(self, content: str) -> Dict[int, Set[str]]:
        """Parse suppression comments from source code content.

        Args:
            content: Source code string.

        Returns:
            Mapping of line_number -> set of suppressed rule IDs.
        """
        suppressions: Dict[int, Set[str]] = {}
        lines = content.splitlines()

        for i, line in enumerate(lines):
            lineno = i + 1  # 1-indexed

            # Check for ignore-next-line first (more specific)
            next_match = _IGNORE_NEXT_RE.search(line)
            if next_match:
                rule_ids = self._parse_rule_ids(next_match.group(1))
                next_lineno = lineno + 1
                if next_lineno not in suppressions:
                    suppressions[next_lineno] = set()
                if rule_ids:
                    suppressions[next_lineno].update(rule_ids)
                # Empty set means suppress all — don't overwrite with empty
                # if specific rules were already set
                continue

            # Check for inline ignore (on current line)
            ignore_match = _IGNORE_RE.search(line)
            if ignore_match and "next-line" not in line.lower():
                rule_ids = self._parse_rule_ids(ignore_match.group(1))
                if lineno not in suppressions:
                    suppressions[lineno] = set()
                if rule_ids:
                    suppressions[lineno].update(rule_ids)

        return suppressions

    @staticmethod
    def _parse_rule_ids(raw: str) -> Set[str]:
        """Parse rule IDs from the comment text.

        Examples:
            "" -> set() (suppress all)
            "SEC001" -> {"SEC001"}
            "SEC001, PY-BARE-EXCEPT" -> {"SEC001", "PY-BARE-EXCEPT"}
        """
        raw = raw.strip()
        if not raw:
            return set()

        ids = set()
        for part in re.split(r"[,\s]+", raw):
            part = part.strip()
            if part:
                ids.add(part)
        return ids


class ProjectSuppressions:
    """Project-wide suppressions from .codesageignore file.

    Format (one entry per line):
        RULE-ID              — suppress rule globally
        RULE-ID:path/glob    — suppress rule for matching files
        tests/**             — exclude all files under tests/ from all checks
        **/test_*.py         — exclude matching files from all checks
        # comment            — ignored
        (blank lines)        — ignored

    Lines that look like path patterns (contain '/' or '*', or start with '.')
    are treated as file exclusions rather than rule suppressions.
    """

    def __init__(
        self,
        entries: Optional[List[tuple]] = None,
        file_patterns: Optional[List[str]] = None,
    ):
        self._entries: List[tuple] = entries or []  # [(rule_id, glob_pattern|None)]
        self._file_patterns: List[str] = file_patterns or []

    @staticmethod
    def _is_path_pattern(s: str) -> bool:
        """Heuristic: looks like a file path or glob, not a rule ID."""
        return "/" in s or "*" in s or s.startswith(".")

    @classmethod
    def load(cls, project_path: Path) -> "ProjectSuppressions":
        """Load .codesageignore from the project root."""
        ignore_file = project_path / ".codesageignore"
        if not ignore_file.exists():
            return cls()

        entries: List[tuple] = []
        file_patterns: List[str] = []
        try:
            for line in ignore_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if ":" in line:
                    rule_id, pattern = line.split(":", 1)
                    rule_id = rule_id.strip()
                    pattern = pattern.strip()
                    # If the LHS looks like a path, treat the whole thing as a
                    # file exclusion pattern (e.g. "tests/**" contains no ":")
                    # but "RULE:glob" has a real rule on the left.
                    if cls._is_path_pattern(rule_id):
                        file_patterns.append(line)  # keep the full original
                    else:
                        entries.append((rule_id, pattern))
                elif cls._is_path_pattern(line):
                    file_patterns.append(line)
                else:
                    entries.append((line, None))
        except Exception as e:
            logger.warning(f"Could not read .codesageignore: {e}")

        return cls(entries, file_patterns)

    def is_suppressed(self, rule_id: str, file_path: Path) -> bool:
        """Check if a finding should be suppressed.

        Args:
            rule_id: The rule identifier (e.g., "SEC001", "PY-BARE-EXCEPT").
            file_path: Path of the file containing the finding.

        Returns:
            True if the finding should be suppressed.
        """
        for entry_rule, entry_pattern in self._entries:
            if entry_rule != rule_id:
                continue
            if entry_pattern is None:
                return True  # Global suppression
            if fnmatch.fnmatch(str(file_path), entry_pattern):
                return True
        return False

    def is_file_excluded(self, file_path) -> bool:
        """Check if an entire file should be excluded from all checks.

        Args:
            file_path: Path (str or Path) of the file to check.

        Returns:
            True if the file matches any exclusion pattern.
        """
        path_str = str(file_path)
        for pattern in self._file_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0 and len(self._file_patterns) == 0


def apply_suppressions(
    findings: List[ReviewFinding],
    file_suppressions: Dict[Path, Dict[int, Set[str]]],
    project_suppressions: Optional[ProjectSuppressions] = None,
) -> List[ReviewFinding]:
    """Apply inline and project-level suppressions to findings.

    Marks matching findings as suppressed=True (does not remove them).

    Args:
        findings: List of findings to filter.
        file_suppressions: Per-file inline suppressions from SuppressionParser.
        project_suppressions: Project-wide suppressions from .codesageignore.

    Returns:
        The same list, with .suppressed set on matching findings.
    """
    for finding in findings:
        # Check project-level suppression
        if project_suppressions and finding.rule_id:
            if project_suppressions.is_suppressed(finding.rule_id, finding.file):
                finding.suppressed = True
                continue

        # Check inline suppression
        file_supps = file_suppressions.get(finding.file)
        if not file_supps or finding.line is None:
            continue

        line_supps = file_supps.get(finding.line)
        if line_supps is None:
            continue

        if len(line_supps) == 0:
            # Empty set = suppress all rules on this line
            finding.suppressed = True
        elif finding.rule_id and finding.rule_id in line_supps:
            finding.suppressed = True

    return findings
