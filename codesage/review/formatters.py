"""Review output formatters.

Provides formatting for code review results.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codesage.review.models import IssueSeverity, ReviewResult


class ReviewFormatter:
    """Format review results for terminal output."""

    SEVERITY_COLORS = {
        IssueSeverity.CRITICAL: "red bold",
        IssueSeverity.WARNING: "yellow",
        IssueSeverity.SUGGESTION: "blue",
        IssueSeverity.PRAISE: "green",
    }

    SEVERITY_ICONS = {
        IssueSeverity.CRITICAL: "[X]",
        IssueSeverity.WARNING: "[!]",
        IssueSeverity.SUGGESTION: "[?]",
        IssueSeverity.PRAISE: "[+]",
    }

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_result(self, result: ReviewResult, verbose: bool = False) -> None:
        """Print the review result to console."""
        self.console.print()
        self._print_summary(result)
        self._print_files_table(result)
        self._print_issues(result)
        self._print_pr_description(result)
        self.console.print()

    def _print_summary(self, result: ReviewResult) -> None:
        """Print summary panel."""
        if result.has_blocking_issues:
            style = "red"
            title = "Code Review - ISSUES FOUND"
        elif result.warning_count > 0:
            style = "yellow"
            title = "Code Review - Warnings"
        else:
            style = "green"
            title = "Code Review - Passed"

        summary_text = Text()
        summary_text.append(f"Files: {len(result.files_changed)}\n")
        summary_text.append(f"Changes: +{result.total_additions} -{result.total_deletions}\n\n")

        if result.critical_count:
            summary_text.append(f"Critical: {result.critical_count}\n", style="red bold")
        if result.warning_count:
            summary_text.append(f"Warnings: {result.warning_count}\n", style="yellow")
        if result.suggestion_count:
            summary_text.append(f"Suggestions: {result.suggestion_count}\n", style="blue")
        if result.praise_count:
            summary_text.append(f"Good practices: {result.praise_count}\n", style="green")

        self.console.print(Panel(summary_text, title=title, border_style=style))

    def _print_files_table(self, result: ReviewResult) -> None:
        """Print files changed table."""
        if not result.files_changed:
            return

        self.console.print()
        table = Table(title="Files Changed")
        table.add_column("Status", width=8)
        table.add_column("File")
        table.add_column("Changes", justify="right")

        status_styles = {"A": "green", "M": "yellow", "D": "red", "R": "cyan"}
        status_names = {"A": "Added", "M": "Modified", "D": "Deleted", "R": "Renamed"}

        for fc in result.files_changed:
            table.add_row(
                Text(status_names.get(fc.status, fc.status), style=status_styles.get(fc.status, "white")),
                str(fc.path),
                f"+{fc.additions} -{fc.deletions}",
            )

        self.console.print(table)

    def _print_issues(self, result: ReviewResult) -> None:
        """Print review issues."""
        if not result.issues:
            return

        self.console.print()
        self.console.print("[bold]Review Findings:[/bold]")

        for severity in IssueSeverity:
            issues = [i for i in result.issues if i.severity == severity]
            if not issues:
                continue

            self.console.print()
            self.console.print(f"[{self.SEVERITY_COLORS[severity]}]{severity.value.upper()}[/]")

            for issue in issues:
                icon = self.SEVERITY_ICONS[severity]
                location = f"{issue.file}"
                if issue.line:
                    location += f":{issue.line}"

                self.console.print(f"  {icon} [{self.SEVERITY_COLORS[severity]}]{location}[/]")
                self.console.print(f"      {issue.message}")
                if issue.suggestion:
                    self.console.print(f"      [dim]Fix: {issue.suggestion}[/dim]")

    def _print_pr_description(self, result: ReviewResult) -> None:
        """Print generated PR description."""
        if not result.pr_description:
            return

        self.console.print()
        self.console.print(Panel(
            result.pr_description,
            title="Generated PR Description",
            border_style="cyan",
        ))
