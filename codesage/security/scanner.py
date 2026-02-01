"""Security scanner for detecting vulnerabilities in code.

Provides static pattern matching for security analysis.
"""

import subprocess
import time
from pathlib import Path
from typing import List, Optional, Set

from codesage.security.models import SecurityFinding, SecurityReport, SecurityRule, Severity
from codesage.security.rules import ALL_RULES, get_enabled_rules
from codesage.utils.logging import get_logger

logger = get_logger(__name__)


class SecurityScanner:
    """Multi-layer security scanner for code analysis."""

    SCANNABLE_EXTENSIONS: Set[str] = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".rb", ".php",
        ".yml", ".yaml", ".json", ".toml",
        ".env", ".sh", ".bash", ".sql",
    }

    SKIP_FILES: Set[str] = {
        "package-lock.json", "yarn.lock", "poetry.lock",
        "Pipfile.lock", "Cargo.lock", "go.sum",
    }

    # Files that contain security rule definitions (avoid false positives)
    SKIP_RULE_FILES: Set[str] = {
        "injection.py", "secrets.py", "xss.py", "crypto.py",
        "deserialization.py", "config.py",
    }

    # Test file patterns to skip by default
    TEST_FILE_PREFIXES = ("test_", "tests_")
    TEST_FILE_SUFFIXES = ("_test.py", "_tests.py", "_spec.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts")
    TEST_FILE_NAMES = {"conftest.py", "setup.py"}

    def __init__(
        self,
        rules: Optional[List[SecurityRule]] = None,
        severity_threshold: Severity = Severity.LOW,
        context_lines: int = 2,
        include_tests: bool = False,
    ):
        """Initialize the scanner.

        Args:
            rules: Custom security rules (defaults to all enabled rules)
            severity_threshold: Minimum severity to report
            context_lines: Number of context lines to include
            include_tests: Include test files in scanning (default: False)
        """
        self.rules = rules or get_enabled_rules()
        self.severity_threshold = severity_threshold
        self.context_lines = context_lines
        self.include_tests = include_tests

    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single file for security issues."""
        findings = []

        if not self._should_scan_file(file_path):
            return findings

        content = self._read_file(file_path)
        if content is None:
            return findings

        lines = content.splitlines()

        for rule in self.rules:
            if rule.severity < self.severity_threshold:
                continue

            for match in rule.matches(content):
                finding = self._create_finding(rule, file_path, content, lines, match)
                findings.append(finding)

        return findings

    def scan_files(self, files: List[Path]) -> SecurityReport:
        """Scan multiple files for security issues."""
        start_time = time.time()
        all_findings = []
        files_scanned = 0

        for file_path in files:
            if file_path.is_file():
                all_findings.extend(self.scan_file(file_path))
                files_scanned += 1

        return SecurityReport(
            findings=all_findings,
            files_scanned=files_scanned,
            scan_duration_ms=(time.time() - start_time) * 1000,
            severity_threshold=self.severity_threshold,
        )

    def scan_directory(
        self,
        directory: Path,
        exclude_dirs: Optional[Set[str]] = None,
    ) -> SecurityReport:
        """Scan all files in a directory recursively."""
        exclude_dirs = exclude_dirs or self._default_exclude_dirs()
        files_to_scan = self._collect_files(directory, exclude_dirs)
        return self.scan_files(files_to_scan)

    def scan_staged_files(self, repo_path: Optional[Path] = None) -> SecurityReport:
        """Scan git staged files for security issues."""
        repo_path = repo_path or Path.cwd()
        files = self._get_git_staged_files(repo_path)
        return self.scan_files(files) if files else SecurityReport(files_scanned=0)

    def scan_uncommitted_changes(self, repo_path: Optional[Path] = None) -> SecurityReport:
        """Scan all uncommitted changes (staged + unstaged)."""
        repo_path = repo_path or Path.cwd()
        files = self._get_git_changed_files(repo_path)
        return self.scan_files(files) if files else SecurityReport(files_scanned=0)

    # Private methods

    def _should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned."""
        if file_path.suffix.lower() not in self.SCANNABLE_EXTENSIONS:
            return False
        if file_path.name in self.SKIP_FILES:
            return False

        # Skip security rule definition files (avoid false positives from rule patterns)
        if self._is_security_rule_file(file_path):
            return False

        # Skip test files unless explicitly included
        if not self.include_tests:
            if self._is_test_file(file_path):
                return False

        return True

    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        name = file_path.name.lower()

        # Check exact names
        if name in self.TEST_FILE_NAMES:
            return True

        # Check prefixes/suffixes
        if name.startswith(self.TEST_FILE_PREFIXES):
            return True
        if name.endswith(self.TEST_FILE_SUFFIXES):
            return True

        return False

    def _is_security_rule_file(self, file_path: Path) -> bool:
        """Check if file is a security rule definition file.

        These files contain patterns that would trigger false positives
        since they define the security rules themselves.
        """
        # Check if file is in a 'rules' directory under 'security'
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part == "security" and i + 1 < len(parts) and parts[i + 1] == "rules":
                return True

        return False

    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding fallback."""
        for encoding in ("utf-8", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                return None
        return None

    def _create_finding(
        self,
        rule: SecurityRule,
        file_path: Path,
        content: str,
        lines: List[str],
        match,
    ) -> SecurityFinding:
        """Create a SecurityFinding from a regex match."""
        line_number = content[:match.start()].count("\n") + 1
        line_content = lines[line_number - 1] if 0 < line_number <= len(lines) else match.group(0)

        line_start = content.rfind("\n", 0, match.start()) + 1
        column_start = match.start() - line_start
        column_end = column_start + len(match.group(0))

        context_before = self._get_context_lines(lines, line_number, before=True)
        context_after = self._get_context_lines(lines, line_number, before=False)

        return SecurityFinding(
            rule=rule,
            file=file_path,
            line_number=line_number,
            line_content=line_content,
            match_text=match.group(0),
            column_start=column_start,
            column_end=column_end,
            context_before=context_before,
            context_after=context_after,
        )

    def _get_context_lines(self, lines: List[str], line_number: int, before: bool) -> List[str]:
        """Get context lines before or after the match."""
        if before:
            start = max(0, line_number - 1 - self.context_lines)
            end = line_number - 1
        else:
            start = line_number
            end = min(len(lines), line_number + self.context_lines)

        return lines[start:end]

    def _default_exclude_dirs(self) -> Set[str]:
        """Default directories to exclude from scanning."""
        return {
            # Version control
            ".git", ".svn", ".hg",
            # Dependencies
            "node_modules", "venv", ".venv", "env", ".env",
            "vendor", "third_party",
            # Build/cache
            "__pycache__", ".pytest_cache", ".mypy_cache",
            "dist", "build", ".codesage", ".tox", ".nox",
            # Test directories (often contain intentional security patterns)
            "tests", "test", "testing", "__tests__",
            "spec", "specs", "fixtures",
        }

    def _collect_files(self, directory: Path, exclude_dirs: Set[str]) -> List[Path]:
        """Collect files to scan from directory."""
        files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and not self._is_excluded(file_path, exclude_dirs):
                files.append(file_path)
        return files

    def _is_excluded(self, file_path: Path, exclude_dirs: Set[str]) -> bool:
        """Check if file is in excluded directory."""
        return any(parent.name in exclude_dirs for parent in file_path.parents)

    def _get_git_staged_files(self, repo_path: Path) -> List[Path]:
        """Get list of staged files from git."""
        return self._run_git_command(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            repo_path,
        )

    def _get_git_changed_files(self, repo_path: Path) -> List[Path]:
        """Get list of all changed files (staged + unstaged + untracked)."""
        staged = self._run_git_command(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            repo_path,
        )
        unstaged = self._run_git_command(
            ["git", "diff", "--name-only", "--diff-filter=ACM"],
            repo_path,
        )
        untracked = self._run_git_command(
            ["git", "ls-files", "--others", "--exclude-standard"],
            repo_path,
        )

        all_files = set(staged + unstaged + untracked)
        return list(all_files)

    def _run_git_command(self, cmd: List[str], repo_path: Path) -> List[Path]:
        """Run git command and return list of file paths."""
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return [
                repo_path / f.strip()
                for f in result.stdout.strip().split("\n")
                if f.strip()
            ]
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Git command failed: {e}")
            return []
