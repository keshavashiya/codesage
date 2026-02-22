"""Rich terminal output for unified review results.

Formats UnifiedReviewResult into beautiful, scannable terminal output
with severity-colored findings grouped by file.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codesage.review.models import ReviewFinding, UnifiedReviewResult

_SEVERITY_STYLES = {
    "critical": "red bold",
    "high": "red",
    "warning": "yellow",
    "suggestion": "blue",
    "praise": "green",
}

_SEVERITY_ICONS = {
    "critical": "X",
    "high": "!",
    "warning": "^",
    "suggestion": ">",
    "praise": "+",
}

_SEVERITY_ORDER = ["critical", "high", "warning", "suggestion", "praise"]

_STATUS_STYLES = {"A": "green", "M": "yellow", "D": "red", "R": "cyan"}


class RichReviewOutput:
    """Rich terminal formatter for unified review results."""

    def __init__(self, console: Optional[Console] = None, verbose: bool = False):
        self.console = console or Console()
        self.verbose = verbose

    def print_result(
        self,
        result: UnifiedReviewResult,
        severity_threshold: str = "critical",
    ) -> None:
        """Print the complete review result."""
        self.console.print()
        self._print_header(result, severity_threshold)
        self._print_files(result)
        self._print_findings(result)
        self._print_footer(result)
        self.console.print()

    def _print_header(self, result: UnifiedReviewResult, threshold: str) -> None:
        """Print summary header panel."""
        is_blocking = result.has_blocking_issues(threshold)

        if is_blocking:
            border = "red"
            title = "Review -- BLOCKED"
        elif result.high_count or result.warning_count:
            border = "yellow"
            title = "Review -- Warnings"
        else:
            border = "green"
            title = "Review -- Passed"

        text = Text()

        # Stats line
        n_files = len(result.files_changed)
        text.append(f"{n_files} file{'s' if n_files != 1 else ''}")
        text.append(f"  +{result.total_additions} -{result.total_deletions}")
        text.append(f"  {result.mode} mode")
        text.append(f"  {result.duration_ms:.0f}ms\n\n")

        # Severity counts
        counts = []
        if result.critical_count:
            counts.append(("critical", result.critical_count, "red bold"))
        if result.high_count:
            counts.append(("high", result.high_count, "red"))
        if result.warning_count:
            counts.append(("warning", result.warning_count, "yellow"))
        if result.suggestion_count:
            counts.append(("suggestion", result.suggestion_count, "blue"))
        if result.praise_count:
            counts.append(("good practice", result.praise_count, "green"))

        if counts:
            for i, (label, count, style) in enumerate(counts):
                if i > 0:
                    text.append("  ")
                text.append(f"{count} {label}", style=style)
            text.append("\n")
        else:
            text.append("No issues found\n", style="green")

        if result.suppressed_count:
            text.append(f"({result.suppressed_count} suppressed)", style="dim")

        self.console.print(
            Panel(text, title=f"[bold]{title}[/bold]", border_style=border)
        )

    def _print_files(self, result: UnifiedReviewResult) -> None:
        """Print compact files changed table."""
        if not result.files_changed:
            return

        self.console.print()
        table = Table(
            show_header=True, header_style="bold", box=None, padding=(0, 1)
        )
        table.add_column("", width=1)
        table.add_column("File")
        table.add_column("Changes", justify="right")
        table.add_column("Issues", justify="right")

        # Count findings per file
        file_counts: Dict[str, int] = defaultdict(int)
        for f in result.active_findings:
            file_counts[str(f.file)] += 1

        for fc in result.files_changed:
            status_char = fc.status[0] if fc.status else "?"
            style = _STATUS_STYLES.get(status_char, "white")
            issue_count = file_counts.get(str(fc.path), 0)
            issue_text = str(issue_count) if issue_count > 0 else ""
            issue_style = "red" if issue_count > 0 else "dim"

            table.add_row(
                Text(status_char, style=style),
                str(fc.path),
                f"+{fc.additions} -{fc.deletions}",
                Text(issue_text, style=issue_style),
            )

        self.console.print(table)

    def _print_findings(self, result: UnifiedReviewResult) -> None:
        """Print findings grouped by file, sorted by severity."""
        findings = result.active_findings
        if not findings:
            return

        self.console.print()
        self.console.print("[bold]Findings:[/bold]")

        # Group by file
        by_file: Dict[str, List[ReviewFinding]] = defaultdict(list)
        for f in findings:
            by_file[str(f.file)].append(f)

        for file_path in sorted(by_file.keys()):
            file_findings = by_file[file_path]
            self.console.print()
            self.console.print(f"  [bold]{file_path}[/bold]")

            # Sort by severity (critical first) then by line number
            file_findings.sort(
                key=lambda f: (
                    _SEVERITY_ORDER.index(f.severity)
                    if f.severity in _SEVERITY_ORDER
                    else 99,
                    f.line or 0,
                )
            )

            for finding in file_findings:
                icon = _SEVERITY_ICONS.get(finding.severity, "?")
                style = _SEVERITY_STYLES.get(finding.severity, "white")

                # Location
                loc = ""
                if finding.line:
                    loc = f":{finding.line}"

                # Rule tag
                rule = ""
                if finding.rule_id:
                    rule = f" [dim]{finding.rule_id}[/dim]"

                self.console.print(
                    f"    [{style}]{icon}[/{style}]{loc}{rule}  {finding.message}"
                )

                if finding.suggestion:
                    self.console.print(
                        f"      [dim]> {finding.suggestion}[/dim]"
                    )

        # Suppressed details in verbose mode
        if self.verbose and result.suppressed_count:
            suppressed = [f for f in result.findings if f.suppressed]
            self.console.print()
            self.console.print(
                f"  [dim]{len(suppressed)} finding(s) suppressed "
                f"by inline/project rules[/dim]"
            )

    def _print_footer(self, result: UnifiedReviewResult) -> None:
        """Print timing and mode info."""
        if not self.verbose:
            return

        self.console.print()
        parts = []

        if result.stage_timings:
            timing_parts = []
            for stage, ms in sorted(result.stage_timings.items()):
                timing_parts.append(f"{stage}={ms:.0f}ms")
            parts.append(" ".join(timing_parts))

        if result.stages_skipped:
            parts.append(f"skipped: {', '.join(result.stages_skipped)}")

        if parts:
            self.console.print(f"  [dim]{' | '.join(parts)}[/dim]")
