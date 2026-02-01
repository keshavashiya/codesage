"""Code review analyzer.

AI-powered code review analysis.
"""

from pathlib import Path
from typing import List, Optional

from codesage.llm.provider import LLMProvider
from codesage.review.models import FileChange, IssueSeverity, ReviewIssue, ReviewResult
from codesage.review.diff import DiffExtractor
from codesage.review.prompts import REVIEW_SYSTEM_PROMPT, REVIEW_FILE_PROMPT, PR_DESCRIPTION_PROMPT
from codesage.utils.config import Config
from codesage.utils.logging import get_logger

logger = get_logger(__name__)


class ReviewAnalyzer:
    """Analyzes code changes and provides AI-powered review feedback."""

    MAX_DIFF_LENGTH = 8000

    def __init__(
        self,
        config: Optional[Config] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the review analyzer."""
        self.config = config
        self.repo_path = repo_path or Path.cwd()
        self._llm: Optional[LLMProvider] = None
        self._diff_extractor = DiffExtractor(self.repo_path)

    @property
    def llm(self) -> LLMProvider:
        """Get or create the LLM provider."""
        if self._llm is None:
            if self.config:
                self._llm = LLMProvider(self.config.llm)
            else:
                from codesage.utils.config import LLMConfig
                self._llm = LLMProvider(LLMConfig())
        return self._llm

    def get_staged_changes(self) -> List[FileChange]:
        """Get list of staged file changes."""
        return self._diff_extractor.get_staged_changes()

    def get_all_changes(self) -> List[FileChange]:
        """Get all uncommitted changes."""
        return self._diff_extractor.get_all_changes()

    def review_changes(
        self,
        changes: Optional[List[FileChange]] = None,
        generate_pr_description: bool = False,
    ) -> ReviewResult:
        """Review code changes using AI."""
        if changes is None:
            changes = self.get_staged_changes()

        if not changes:
            return ReviewResult(summary="No changes to review")

        result = ReviewResult(files_changed=changes)

        # Review each file
        for change in changes:
            if change.status == "D" or not change.diff:
                continue
            issues = self._review_file(change)
            result.issues.extend(issues)

        result.summary = self._generate_summary(result)

        if generate_pr_description:
            result.pr_description = self._generate_pr_description(changes)

        return result

    def _review_file(self, change: FileChange) -> List[ReviewIssue]:
        """Review a single file change."""
        diff = self._truncate_diff(change.diff)

        prompt = REVIEW_FILE_PROMPT.format(
            diff=diff,
            file_path=change.path,
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=REVIEW_SYSTEM_PROMPT,
                temperature=0.3,
            )
            return self._parse_response(response, change.path)

        except Exception as e:
            logger.error(f"Failed to review {change.path}: {e}")
            return []

    def _truncate_diff(self, diff: str) -> str:
        """Truncate diff if too long."""
        if len(diff) > self.MAX_DIFF_LENGTH:
            return diff[:self.MAX_DIFF_LENGTH] + "\n... (truncated)"
        return diff

    def _parse_response(self, response: str, file_path: Path) -> List[ReviewIssue]:
        """Parse LLM response into structured issues."""
        issues = []
        current_issue = None
        current_suggestion = []

        for line in response.strip().split("\n"):
            line = line.strip()

            for severity in IssueSeverity:
                marker = f"[{severity.value.upper()}]"
                if line.upper().startswith(marker):
                    if current_issue:
                        if current_suggestion:
                            current_issue.suggestion = " ".join(current_suggestion)
                        issues.append(current_issue)

                    rest = line[len(marker):].strip()
                    line_num = self._extract_line_number(rest)
                    message = self._extract_message(rest)

                    current_issue = ReviewIssue(
                        severity=severity,
                        file=file_path,
                        line=line_num,
                        message=message,
                    )
                    current_suggestion = []
                    break
            else:
                if line.lower().startswith("suggestion:"):
                    current_suggestion.append(line[11:].strip())
                elif current_suggestion and line:
                    current_suggestion.append(line)
                elif current_issue and line:
                    current_issue.message += " " + line

        if current_issue:
            if current_suggestion:
                current_issue.suggestion = " ".join(current_suggestion)
            issues.append(current_issue)

        return issues

    def _extract_line_number(self, text: str) -> Optional[int]:
        """Extract line number from text like 'Line 42: ...'"""
        if text.lower().startswith("line"):
            parts = text.split(":", 1)
            try:
                return int(parts[0].lower().replace("line", "").strip())
            except ValueError:
                pass
        return None

    def _extract_message(self, text: str) -> str:
        """Extract message from text, removing line number prefix."""
        if text.lower().startswith("line"):
            parts = text.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()
        return text

    def _generate_summary(self, result: ReviewResult) -> str:
        """Generate a summary of the review."""
        parts = [
            f"{len(result.files_changed)} file{'s' if len(result.files_changed) != 1 else ''} changed",
            f"+{result.total_additions} -{result.total_deletions}",
        ]

        if result.critical_count:
            parts.append(f"{result.critical_count} critical")
        if result.warning_count:
            parts.append(f"{result.warning_count} warning{'s' if result.warning_count != 1 else ''}")
        if result.suggestion_count:
            parts.append(f"{result.suggestion_count} suggestion{'s' if result.suggestion_count != 1 else ''}")

        return " | ".join(parts)

    def _generate_pr_description(self, changes: List[FileChange]) -> str:
        """Generate a PR description from changes."""
        files_summary = "\n".join([
            f"- {c.path} ({c.status}: +{c.additions} -{c.deletions})"
            for c in changes
        ])

        diff_summary = ""
        for c in changes:
            if c.diff:
                diff_summary += f"\n### {c.path}\n{c.diff[:500]}\n"

        if len(diff_summary) > 4000:
            diff_summary = diff_summary[:4000] + "\n..."

        prompt = PR_DESCRIPTION_PROMPT.format(
            files_summary=files_summary,
            additions=sum(c.additions for c in changes),
            deletions=sum(c.deletions for c in changes),
            diff_summary=diff_summary,
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that writes clear, concise PR descriptions.",
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate PR description: {e}")
            return ""
