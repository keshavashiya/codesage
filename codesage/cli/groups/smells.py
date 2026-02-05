"""Code smell detection command group for CodeSage CLI."""

import json
from pathlib import Path

import typer
from rich.table import Table

from codesage.cli.utils.console import get_console, print_error, print_warning
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="Detect pattern-deviation code smells")


@app.command("detect")
@handle_errors
def detect(
    file_path: str = typer.Argument(None, help="Optional file path to analyze"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
    severity: str = typer.Option(
        "warning,error",
        "--severity",
        "-s",
        help="Comma-separated severities to include (info, warning, error)",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    staged_only: bool = typer.Option(
        False,
        "--staged",
        help="Analyze only staged changes when no file is provided",
    ),
) -> None:
    """Detect code smells in a file or current changes."""
    from codesage.utils.config import Config
    from codesage.review.smells import PatternDeviationDetector
    from codesage.review.diff import DiffExtractor

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    if not config.features.code_smell_detection:
        print_warning("Code smell detection is disabled.")
        console.print("  Enable with: [cyan]codesage features enable code_smell_detection[/cyan]")
        raise typer.Exit(1)

    detector = PatternDeviationDetector(config)
    smells = []

    if file_path:
        smells = detector.detect_file(project_path / file_path)
    else:
        diff = DiffExtractor(project_path)
        changes = diff.get_staged_changes() if staged_only else diff.get_all_changes()
        target_files = [c.path for c in changes if c.status != "D"]
        if not target_files:
            print_warning("No changes to analyze")
            return

        for f in target_files:
            smells.extend(detector.detect_file(f))

    # Filter by severity
    allowed = {s.strip().lower() for s in severity.split(",") if s.strip()}
    if allowed:
        smells = [s for s in smells if s.severity.lower() in allowed]

    if json_output:
        console.print(json.dumps([s.to_dict() for s in smells], indent=2))
        return

    if not smells:
        console.print("[green]No smells detected.[/green]")
        return

    # Count by severity
    error_count = sum(1 for s in smells if s.severity.lower() == "error")
    warning_count = sum(1 for s in smells if s.severity.lower() == "warning")
    info_count = sum(1 for s in smells if s.severity.lower() == "info")

    # Show severity summary
    from codesage.cli.utils.formatters import format_count_summary
    summary = format_count_summary(
        total=len(smells),
        errors=error_count,
        warnings=warning_count,
        info=info_count,
        item_name="code smells",
    )
    console.print(f"\n{summary}\n")

    table = Table(title="Detected Code Smells")
    table.add_column("Severity", style="cyan")
    table.add_column("File", style="white")
    table.add_column("Line", style="white")
    table.add_column("Message", style="yellow")

    for s in smells:
        # Color severity based on level
        sev = s.severity.lower()
        if sev == "error":
            sev_display = f"[red]{s.severity}[/red]"
        elif sev == "warning":
            sev_display = f"[yellow]{s.severity}[/yellow]"
        else:
            sev_display = f"[blue]{s.severity}[/blue]"

        table.add_row(
            sev_display,
            s.file,
            str(s.line or ""),
            s.message,
        )

    console.print(table)
