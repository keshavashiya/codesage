"""Security report formatters.

Provides formatting for security scan results in various output formats.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codesage.security.models import SecurityFinding, SecurityReport, Severity


class ReportFormatter:
    """Format security reports for terminal output."""

    SEVERITY_COLORS = {
        Severity.CRITICAL: "red bold",
        Severity.HIGH: "red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "blue",
        Severity.INFO: "dim",
    }

    SEVERITY_ICONS = {
        Severity.CRITICAL: "[!!!]",
        Severity.HIGH: "[!!]",
        Severity.MEDIUM: "[!]",
        Severity.LOW: "[i]",
        Severity.INFO: "[.]",
    }

    def __init__(self, console: Optional[Console] = None):
        """Initialize the formatter."""
        self.console = console or Console()

    def format_summary(self, report: SecurityReport) -> Panel:
        """Format the report summary as a Rich panel."""
        if report.is_clean:
            content = Text("No security issues found!", style="green bold")
            return Panel(content, title="Security Scan Results", border_style="green")

        lines = []
        lines.append(f"Files scanned: {report.files_scanned}")
        lines.append(f"Total findings: {report.total_count}")
        lines.append("")

        if report.critical_count:
            lines.append(
                Text(f"  CRITICAL: {report.critical_count}", style=self.SEVERITY_COLORS[Severity.CRITICAL])
            )
        if report.high_count:
            lines.append(
                Text(f"  HIGH: {report.high_count}", style=self.SEVERITY_COLORS[Severity.HIGH])
            )
        if report.medium_count:
            lines.append(
                Text(f"  MEDIUM: {report.medium_count}", style=self.SEVERITY_COLORS[Severity.MEDIUM])
            )
        if report.low_count:
            lines.append(
                Text(f"  LOW: {report.low_count}", style=self.SEVERITY_COLORS[Severity.LOW])
            )
        if report.info_count:
            lines.append(
                Text(f"  INFO: {report.info_count}", style=self.SEVERITY_COLORS[Severity.INFO])
            )

        content = Text()
        for line in lines:
            if isinstance(line, Text):
                content.append(line)
            else:
                content.append(line)
            content.append("\n")

        border_style = "red" if report.has_blocking_issues else "yellow"
        title = "Security Scan - BLOCKED" if report.has_blocking_issues else "Security Scan Results"

        return Panel(content, title=title, border_style=border_style)

    def format_findings_table(self, report: SecurityReport) -> Table:
        """Format findings as a Rich table."""
        table = Table(title="Security Findings", show_lines=True)
        table.add_column("Severity", style="bold", width=10)
        table.add_column("Location", style="cyan")
        table.add_column("Rule", style="white")
        table.add_column("Message")

        sorted_findings = sorted(report.findings, key=lambda f: f.severity, reverse=True)

        for finding in sorted_findings:
            severity_text = Text(
                f"{self.SEVERITY_ICONS[finding.severity]} {finding.severity.value.upper()}",
                style=self.SEVERITY_COLORS[finding.severity],
            )

            table.add_row(
                severity_text,
                finding.location,
                f"[{finding.rule.id}] {finding.rule.name}",
                finding.rule.message,
            )

        return table

    def format_finding_detail(self, finding: SecurityFinding) -> Panel:
        """Format a single finding with full details."""
        lines = []

        severity_style = self.SEVERITY_COLORS[finding.severity]
        lines.append(
            Text(
                f"{self.SEVERITY_ICONS[finding.severity]} {finding.rule.name}",
                style=severity_style,
            )
        )
        lines.append(Text(f"Rule: {finding.rule.id}", style="dim"))
        if finding.rule.cwe_id:
            lines.append(Text(f"CWE: {finding.rule.cwe_id}", style="dim"))
        lines.append("")

        lines.append(Text(f"Location: {finding.location}", style="cyan"))
        lines.append("")

        lines.append(Text("Code:", style="bold"))
        for i, ctx_line in enumerate(
            finding.context_before,
            start=finding.line_number - len(finding.context_before)
        ):
            lines.append(Text(f"  {i:4d} | {ctx_line}", style="dim"))

        lines.append(
            Text(f"  {finding.line_number:4d} | {finding.line_content}", style=severity_style)
        )

        if finding.column_start > 0:
            underline = " " * (8 + finding.column_start) + "^" * (finding.column_end - finding.column_start)
            lines.append(Text(underline, style=severity_style))

        for i, ctx_line in enumerate(finding.context_after, start=finding.line_number + 1):
            lines.append(Text(f"  {i:4d} | {ctx_line}", style="dim"))

        lines.append("")
        lines.append(Text(f"Issue: {finding.rule.message}", style="white"))
        if finding.rule.description:
            lines.append(Text(f"       {finding.rule.description}", style="dim"))

        if finding.rule.fix_suggestion:
            lines.append("")
            lines.append(Text(f"Fix: {finding.rule.fix_suggestion}", style="green"))

        content = Text()
        for line in lines:
            content.append(line)
            content.append("\n")

        return Panel(content, border_style=severity_style)

    def print_report(self, report: SecurityReport, verbose: bool = False) -> None:
        """Print the full report to console."""
        self.console.print()
        self.console.print(self.format_summary(report))

        if report.total_count > 0:
            self.console.print()
            self.console.print(self.format_findings_table(report))

            if verbose:
                self.console.print()
                self.console.print("[bold]Detailed Findings:[/bold]")
                for finding in sorted(report.findings, key=lambda f: f.severity, reverse=True):
                    self.console.print()
                    self.console.print(self.format_finding_detail(finding))

        self.console.print()
        self.console.print(f"[dim]Scan completed in {report.scan_duration_ms:.0f}ms[/dim]")
