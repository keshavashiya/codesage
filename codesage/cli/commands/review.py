"""Review command for CodeSage CLI.

Runs the unified review pipeline on uncommitted changes:
security, bad practices, complexity, naming, structure, patterns,
duplication, and optional LLM synthesis.
"""

from pathlib import Path

import typer

from codesage.cli.utils.console import get_console, print_error, print_info, print_warning
from codesage.cli.utils.decorators import handle_errors


@handle_errors
def review(
    path: str = typer.Argument(".", help="Project directory"),
    staged_only: bool = typer.Option(
        False,
        "--staged",
        "-s",
        help="Only review staged changes (default: all uncommitted)",
    ),
    mode: str = typer.Option(
        "fast",
        "--mode",
        "-m",
        help="fast = static only (<5s), full = +similarity +LLM",
    ),
    severity: str = typer.Option(
        "high",
        "--severity",
        help="Block threshold: critical, high, warning, suggestion",
    ),
    output_format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="Output: rich (terminal), json, sarif",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show timing breakdown and suppression details",
    ),
) -> None:
    """Review uncommitted changes for issues.

    Scans for security vulnerabilities, bad practices, high complexity,
    naming violations, structural issues, and code duplication.

    \b
    Modes:
      fast  Static analysis only. Runs in under 5 seconds. (default)
      full  Adds semantic similarity search and LLM synthesis.

    \b
    Examples:
      codesage review                  # review all uncommitted changes
      codesage review --staged         # only staged changes (pre-commit)
      codesage review --mode full      # include LLM synthesis
      codesage review --format json    # JSON output for CI/CD
      codesage review --format sarif   # SARIF for GitHub Actions
      codesage review --severity warning  # block on warnings too
    """
    console = get_console()
    project_path = Path(path).resolve()

    # Load config (optional — pipeline works without it for static checks)
    config = None
    try:
        from codesage.utils.config import Config

        config = Config.load(project_path)
    except FileNotFoundError:
        if mode == "full":
            print_warning(
                "Project not initialized. Falling back to fast mode."
            )
            print_info("  Run 'codesage init' for full mode (similarity + LLM).")
            mode = "fast"
    except Exception:
        if mode == "full":
            mode = "fast"

    # Build and run pipeline
    from codesage.review.pipeline import ReviewPipeline

    pipeline = ReviewPipeline(
        repo_path=project_path,
        config=config,
        mode=mode,
    )

    # For machine-readable formats: use stderr for status so stdout stays clean
    machine_readable = output_format in ("json", "sarif")
    if machine_readable:
        from codesage.cli.utils.console import get_stderr_console
        status_console = get_stderr_console()
    else:
        status_console = console

    with status_console.status("[bold green]Reviewing changes...[/bold green]"):
        result = pipeline.run(
            staged_only=staged_only,
            severity_threshold=severity,
        )

    # No changes found
    if not result.files_changed and not result.findings:
        if "Error:" in result.summary:
            print_error(result.summary)
            raise typer.Exit(1)
        (get_stderr_console() if machine_readable else console).print(
            "[yellow]No changes found to review.[/yellow]"
        )
        raise typer.Exit(0)

    # Output — machine-readable formats go to raw stdout, bypassing Rich
    if output_format == "json":
        import sys
        sys.stdout.write(result.to_json() + "\n")
        sys.stdout.flush()
    elif output_format == "sarif":
        import sys
        sys.stdout.write(result.to_sarif_json() + "\n")
        sys.stdout.flush()
    else:
        from codesage.review.output import RichReviewOutput

        formatter = RichReviewOutput(console=console, verbose=verbose)
        formatter.print_result(result, severity_threshold=severity)

    # Exit code based on threshold
    if result.has_blocking_issues(severity):
        raise typer.Exit(1)
