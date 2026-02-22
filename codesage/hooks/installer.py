"""Git hook installation and management.

Provides functionality to install, uninstall, and manage git hooks.
"""

import os
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from codesage.utils.logging import get_logger

logger = get_logger(__name__)

# Hook marker for identification
HOOK_MARKER = "# CodeSage Pre-Commit Hook"

# Path to hook templates
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class HookStatus:
    """Status of a git hook."""

    installed: bool
    hook_path: Path
    is_codesage_hook: bool
    has_other_hook: bool
    backup_path: Optional[Path] = None


class HookInstaller:
    """Manages git hook installation for CodeSage."""

    def __init__(self, repo_path: Optional[Path] = None):
        """Initialize the hook installer."""
        self.repo_path = repo_path or Path.cwd()
        self._git_dir: Optional[Path] = None

    @property
    def git_dir(self) -> Path:
        """Get the .git directory path."""
        if self._git_dir is None:
            self._git_dir = self._find_git_dir()
        return self._git_dir

    @property
    def hooks_dir(self) -> Path:
        """Get the hooks directory path."""
        return self.git_dir / "hooks"

    @property
    def pre_commit_path(self) -> Path:
        """Get the pre-commit hook path."""
        return self.hooks_dir / "pre-commit"

    def get_status(self) -> HookStatus:
        """Get the current status of the pre-commit hook."""
        hook_path = self.pre_commit_path
        installed = hook_path.exists()
        is_codesage = False
        has_other = False
        backup_path = None

        if installed:
            try:
                content = hook_path.read_text()
                is_codesage = HOOK_MARKER in content
                has_other = not is_codesage
            except Exception:
                has_other = True

        backup = self.pre_commit_path.with_suffix(".pre-codesage")
        if backup.exists():
            backup_path = backup

        return HookStatus(
            installed=installed and is_codesage,
            hook_path=hook_path,
            is_codesage_hook=is_codesage,
            has_other_hook=has_other,
            backup_path=backup_path,
        )

    def install(
        self,
        severity: str = "high",
        mode: str = "fast",
        force: bool = False,
    ) -> bool:
        """Install the pre-commit hook."""
        self.hooks_dir.mkdir(parents=True, exist_ok=True)

        status = self.get_status()

        if status.has_other_hook and not status.is_codesage_hook:
            if not force:
                raise ValueError(
                    f"Pre-commit hook already exists at {self.pre_commit_path}. "
                    "Use --force to backup and replace, or integrate manually."
                )
            self._backup_existing_hook()

        hook_content = self._generate_hook_content(severity=severity, mode=mode)
        self._write_hook(hook_content)

        logger.info(f"Installed pre-commit hook at {self.pre_commit_path}")
        return True

    def uninstall(self, restore_backup: bool = True) -> bool:
        """Uninstall the pre-commit hook."""
        status = self.get_status()

        if not status.installed and not status.is_codesage_hook:
            logger.info("No CodeSage hook installed")
            return True

        if status.is_codesage_hook:
            self.pre_commit_path.unlink()
            logger.info(f"Removed CodeSage hook from {self.pre_commit_path}")

            if restore_backup and status.backup_path and status.backup_path.exists():
                status.backup_path.rename(self.pre_commit_path)
                logger.info(f"Restored backup hook from {status.backup_path}")

        return True

    def update(self, severity: str = "high", mode: str = "fast") -> bool:
        """Update the pre-commit hook with new settings."""
        status = self.get_status()

        if not status.is_codesage_hook:
            logger.warning("No CodeSage hook installed to update")
            return False

        self.pre_commit_path.unlink()
        return self.install(severity=severity, mode=mode)

    # Private methods

    def _find_git_dir(self) -> Path:
        """Find the .git directory for the repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            git_dir = result.stdout.strip()
            if os.path.isabs(git_dir):
                return Path(git_dir)
            return self.repo_path / git_dir
        except subprocess.CalledProcessError:
            raise ValueError(f"Not a git repository: {self.repo_path}")
        except FileNotFoundError:
            raise ValueError("Git is not installed or not in PATH")

    def _backup_existing_hook(self) -> None:
        """Backup existing hook before replacement."""
        backup_path = self.pre_commit_path.with_suffix(".pre-codesage")
        logger.info(f"Backing up existing hook to {backup_path}")
        self.pre_commit_path.rename(backup_path)

    # Embedded fallback template — used when the package was installed without
    # the templates/ data files (e.g. older pipx install before package-data fix).
    _HOOK_TEMPLATE = """\
#!/bin/sh
# CodeSage Pre-Commit Hook
# Installed: {installed_at}
# Mode: {mode} | Severity threshold: {severity}
#
# Blocks commits with issues at or above the severity threshold.
# To bypass (use sparingly): git commit --no-verify

# Check if codesage is available
if ! command -v codesage >/dev/null 2>&1; then
    echo "[codesage] Not installed or not in PATH — skipping review."
    echo "           Install: pipx install pycodesage"
    exit 0
fi

echo "[codesage] Reviewing staged changes ({mode} mode, blocking on {severity}+)..."

# Run review on staged changes only
if codesage review --staged --mode {mode} --severity {severity}; then
    echo "[codesage] Review passed."
    exit 0
else
    exit_code=$?
    if [ "$exit_code" = "1" ]; then
        echo ""
        echo "[codesage] Commit blocked: fix the issues above."
        echo "           To bypass: git commit --no-verify"
        exit 1
    else
        # Non-1 exit = unexpected error; warn but don't block
        echo "[codesage] Review error (exit $exit_code) — allowing commit."
        exit 0
    fi
fi
"""

    def _generate_hook_content(self, severity: str = "high", mode: str = "fast") -> str:
        """Generate hook content from template.

        Falls back to the embedded template string if the file is not found
        (e.g. when installed via pipx without the package-data declaration).
        """
        template_path = TEMPLATES_DIR / "pre-commit.sh"
        if template_path.exists():
            template = template_path.read_text()
        else:
            logger.debug(
                f"Template file not found at {template_path}, using embedded fallback"
            )
            template = self._HOOK_TEMPLATE

        return template.format(
            installed_at=datetime.now().isoformat(),
            severity=severity,
            mode=mode,
        )

    def _write_hook(self, content: str) -> None:
        """Write hook file and make executable."""
        self.pre_commit_path.write_text(content)
        self.pre_commit_path.chmod(
            self.pre_commit_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
