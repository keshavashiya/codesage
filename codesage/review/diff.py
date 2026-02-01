"""Git diff operations.

Handles extraction of git diffs and file changes.
"""

import subprocess
from pathlib import Path
from typing import List, Optional

from codesage.review.models import FileChange
from codesage.utils.logging import get_logger

logger = get_logger(__name__)


class DiffExtractor:
    """Extracts git diffs and file change information."""

    # Maximum file size to include in diff (100KB) - prevents memory issues
    MAX_FILE_SIZE = 100 * 1024

    # Maximum diff content length per file (32KB)
    MAX_DIFF_CONTENT = 32 * 1024

    # File extensions to skip (binary/non-code files)
    SKIP_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.webp',
        '.woff', '.woff2', '.ttf', '.eot', '.otf',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.pyc', '.pyo', '.so', '.dylib', '.dll',
        '.exe', '.bin', '.dat', '.db', '.sqlite',
        '.mp3', '.mp4', '.wav', '.avi', '.mov',
    }

    def __init__(self, repo_path: Path):
        """Initialize with repository path."""
        self.repo_path = repo_path

    def get_staged_changes(self) -> List[FileChange]:
        """Get list of staged file changes."""
        return self._get_changes(staged=True)

    def get_unstaged_changes(self) -> List[FileChange]:
        """Get list of unstaged file changes."""
        return self._get_changes(staged=False)

    def get_all_changes(self) -> List[FileChange]:
        """Get all uncommitted changes (staged + unstaged + untracked)."""
        staged = self._get_changes(staged=True)
        unstaged = self._get_changes(staged=False)
        untracked = self._get_untracked_files()

        staged_paths = {f.path for f in staged}
        all_changes = staged + [f for f in unstaged if f.path not in staged_paths]

        # Add untracked files that aren't already in the list
        existing_paths = {f.path for f in all_changes}
        all_changes.extend([f for f in untracked if f.path not in existing_paths])

        return all_changes

    def _get_untracked_files(self) -> List[FileChange]:
        """Get list of untracked files (new files not yet added to git)."""
        cmd = ["git", "ls-files", "--others", "--exclude-standard"]
        result = self._run_git(cmd)
        if not result:
            return []

        changes = []
        for line in result.strip().split("\n"):
            if not line.strip():
                continue

            path = Path(line.strip())
            full_path = self.repo_path / path

            # Skip directories and non-existent files
            if not full_path.exists() or full_path.is_dir():
                continue

            # Skip binary/non-code files
            if path.suffix.lower() in self.SKIP_EXTENSIONS:
                continue

            # Skip large files to prevent memory issues
            try:
                file_size = full_path.stat().st_size
                if file_size > self.MAX_FILE_SIZE:
                    logger.info(f"Skipping large file {path} ({file_size} bytes)")
                    changes.append(FileChange(
                        path=path,
                        status="A",
                        additions=0,
                        deletions=0,
                        diff=f"[File too large to review: {file_size} bytes]",
                    ))
                    continue
            except OSError:
                continue

            # Read file content as the "diff" (it's all new content)
            try:
                content = full_path.read_text(errors="replace")

                # Check if file appears to be binary
                if self._is_binary_content(content):
                    continue

                line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

                # Format as a unified diff-like output (with truncation)
                diff = f"+++ {path}\n"
                diff_lines = content.split("\n")
                for text_line in diff_lines:
                    diff += f"+{text_line}\n"
                    # Truncate diff if too large
                    if len(diff) > self.MAX_DIFF_CONTENT:
                        diff += "\n... (truncated)\n"
                        break

                changes.append(FileChange(
                    path=path,
                    status="A",  # Added/new file
                    additions=line_count,
                    deletions=0,
                    diff=diff,
                ))
            except Exception as e:
                logger.warning(f"Could not read untracked file {path}: {e}")
                continue

        return changes

    def _is_binary_content(self, content: str) -> bool:
        """Check if content appears to be binary."""
        # Check for null bytes or high ratio of non-printable characters
        if '\x00' in content[:1024]:
            return True
        # Check for mostly non-printable characters
        sample = content[:1024]
        non_printable = sum(1 for c in sample if ord(c) < 32 and c not in '\n\r\t')
        return non_printable > len(sample) * 0.3  # More than 30% non-printable

    def _get_changes(self, staged: bool = True) -> List[FileChange]:
        """Get file changes from git."""
        file_stats = self._get_file_stats(staged)
        file_statuses = self._get_file_statuses(staged)

        changes = []
        for path, (status, old_path) in file_statuses.items():
            adds, dels = file_stats.get(path, (0, 0))
            diff = self._get_file_diff(path, staged)

            changes.append(FileChange(
                path=Path(path),
                status=status,
                additions=adds,
                deletions=dels,
                diff=diff,
                old_path=Path(old_path) if old_path else None,
            ))

        return changes

    def _get_file_stats(self, staged: bool) -> dict:
        """Get addition/deletion stats per file."""
        cmd = ["git", "diff", "--numstat", "--diff-filter=ACDMR"]
        if staged:
            cmd.insert(2, "--cached")

        result = self._run_git(cmd)
        if not result:
            return {}

        stats = {}
        for line in result.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                adds = int(parts[0]) if parts[0] != "-" else 0
                dels = int(parts[1]) if parts[1] != "-" else 0
                path = parts[2]
                stats[path] = (adds, dels)

        return stats

    def _get_file_statuses(self, staged: bool) -> dict:
        """Get status (A/M/D/R) per file."""
        cmd = ["git", "diff", "--name-status", "--diff-filter=ACDMR"]
        if staged:
            cmd.insert(2, "--cached")

        result = self._run_git(cmd)
        if not result:
            return {}

        statuses = {}
        for line in result.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            status = parts[0][0]
            path = parts[-1]
            old_path = parts[1] if len(parts) > 2 else None
            statuses[path] = (status, old_path)

        return statuses

    def _get_file_diff(self, path: str, staged: bool) -> str:
        """Get diff content for a specific file."""
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        cmd.extend(["--", path])

        return self._run_git(cmd) or ""

    def _run_git(self, cmd: List[str]) -> Optional[str]:
        """Run git command and return stdout."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return None
        except FileNotFoundError:
            logger.error("Git is not installed or not in PATH")
            return None
