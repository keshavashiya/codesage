"""Security command group for CodeSage CLI."""

import json
from pathlib import Path

import typer
from rich.table import Table

from codesage.cli.utils.console import get_console
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="Security scanning commands")


@app.command("scan")
@handle_errors
def scan(
    path: str = typer.Argument(".", help="Directory or file to scan"),
    staged: bool = typer.Option(False, "--staged", help="Only scan git staged files"),
    severity: str = typer.Option(
        "low",
        "--severity",
        "-s",
        help="Minimum severity to report (low, medium, high, critical)",
    ),
    include_tests: bool = typer.Option(
        False,
        "--include-tests",
        help="Include test files in scanning (excluded by default)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed findings"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    exit_on_findings: bool = typer.Option(
        False,
        "--exit-on-findings",
        help="Exit with code 1 if any findings (for CI/CD)",
    ),
) -> None:
    """Scan code for security vulnerabilities.

    Detects hardcoded secrets, injection vulnerabilities, and other security issues.
    """
    from codesage.security import ReportFormatter, SecurityScanner, Severity

    console = get_console()

    # Parse severity
    severity_map = {
        "low": Severity.LOW,
        "medium": Severity.MEDIUM,
        "high": Severity.HIGH,
        "critical": Severity.CRITICAL,
    }
    min_severity = severity_map.get(severity.lower(), Severity.LOW)

    scanner = SecurityScanner(severity_threshold=min_severity, include_tests=include_tests)

    if staged:
        console.print("[dim]Scanning staged files...[/dim]")
        report = scanner.scan_staged_files(Path(path).resolve())
    else:
        target = Path(path).resolve()
        if target.is_file():
            console.print(f"[dim]Scanning {target.name}...[/dim]")
            report = scanner.scan_files([target])
        else:
            console.print(f"[dim]Scanning directory {target}...[/dim]")
            report = scanner.scan_directory(target)

    if json_output:
        console.print(json.dumps(report.to_dict(), indent=2))
    else:
        formatter = ReportFormatter(console)
        formatter.print_report(report, verbose=verbose)

    # Exit with code 1 if findings and exit_on_findings is set
    if exit_on_findings and report.total_count > 0:
        raise typer.Exit(1)

    # Always exit with code 1 on critical findings
    if report.has_blocking_issues:
        raise typer.Exit(1)


@app.command("rules")
@handle_errors
def rules(
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
) -> None:
    """List available security rules."""
    from codesage.security.rules import ALL_RULES, get_rules_by_category

    console = get_console()
    rule_list = get_rules_by_category(category) if category else ALL_RULES

    table = Table(title="Security Rules")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Severity")
    table.add_column("Category")
    table.add_column("CWE")

    severity_colors = {
        "critical": "red bold",
        "high": "red",
        "medium": "yellow",
        "low": "blue",
        "info": "dim",
    }

    for rule in rule_list:
        table.add_row(
            rule.id,
            rule.name,
            f"[{severity_colors.get(rule.severity.value, 'white')}]{rule.severity.value}[/]",
            rule.category,
            rule.cwe_id or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(rule_list)} rules[/dim]")
