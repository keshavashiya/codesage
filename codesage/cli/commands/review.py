"""Review command for CodeSage CLI."""

import json
from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from codesage.cli.utils.console import get_console, print_error, print_warning, print_success
from codesage.cli.utils.decorators import handle_errors


@handle_errors
def review(
    path: str = typer.Argument(".", help="Repository path"),
    staged_only: bool = typer.Option(
        False,
        "--staged",
        help="Only review staged changes",
    ),
    generate_pr: bool = typer.Option(
        False,
        "--generate-pr-description",
        help="Generate a PR description",
    ),
    max_files: int = typer.Option(
        0,
        "--max-files",
        "-m",
        help="Maximum files to review (0 for unlimited, default: unlimited)",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM synthesis (fast mode - only static analysis and duplicate detection)",
    ),
    legacy: bool = typer.Option(
        False,
        "--legacy",
        help="Use legacy pure-LLM review (slower, doesn't use codebase index)",
    ),
    include_tests: bool = typer.Option(
        False,
        "--include-tests",
        help="Include test files in review (excluded by default to reduce noise)",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Review uncommitted code changes with AI.

    Uses a hybrid approach:
    1. Static security analysis (instant)
    2. Semantic duplicate detection against your indexed codebase
    3. LLM synthesis for additional insights (optional)

    This finds actual duplicates in YOUR codebase and runs much faster
    than pure LLM review.
    """
    from codesage.review import ReviewFormatter
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    # Choose analyzer based on mode
    if legacy:
        from codesage.review import ReviewAnalyzer
        console.print("[dim]Using legacy LLM-only review...[/dim]")
        analyzer = ReviewAnalyzer(config=config, repo_path=project_path)
    else:
        from codesage.review import HybridReviewAnalyzer
        mode = "static analysis only" if no_llm else "hybrid (static + semantic + LLM)"
        console.print(f"[dim]Using {mode} review...[/dim]")
        analyzer = HybridReviewAnalyzer(config=config, repo_path=project_path)

    console.print("[dim]Analyzing changes...[/dim]")

    if staged_only:
        changes = analyzer.get_staged_changes()
    else:
        changes = analyzer.get_all_changes()

    if not changes:
        print_warning("No changes to review")
        return

    # Filter out test files unless explicitly included
    if not include_tests:
        test_patterns = (
            "test_", "_test.py", "tests/", "test/",
            "spec/", "_spec.py", "conftest.py"
        )
        original_count = len(changes)
        changes = [
            c for c in changes
            if not any(p in str(c.path).lower() for p in test_patterns)
        ]
        excluded = original_count - len(changes)
        if excluded > 0:
            console.print(f"[dim]Excluded {excluded} test files (use --include-tests to include)[/dim]")

    # Limit number of files if specified
    total_files = len(changes)
    if max_files > 0 and len(changes) > max_files:
        console.print(f"[yellow]Limiting review to {max_files} of {total_files} files[/yellow]")
        console.print(f"[dim]Use --max-files 0 to review all files[/dim]")
        changes = changes[:max_files]

    # Show what we're doing
    console.print(f"[dim]Reviewing {len(changes)} files...[/dim]")
    if not legacy:
        console.print("[dim]  ├─ Security pattern matching[/dim]")
        console.print("[dim]  ├─ Semantic duplicate detection[/dim]")
        if not no_llm:
            console.print("[dim]  └─ LLM synthesis for insights[/dim]")
        else:
            console.print("[dim]  └─ (LLM synthesis skipped)[/dim]")

    # Run review with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(changes))

        # For hybrid analyzer, pass the use_llm_synthesis flag
        if legacy:
            result = analyzer.review_changes(
                changes=changes,
                generate_pr_description=generate_pr,
            )
        else:
            result = analyzer.review_changes(
                changes=changes,
                generate_pr_description=generate_pr,
                use_llm_synthesis=not no_llm,
            )

        progress.update(task, completed=len(changes))

    # Output results
    if json_output:
        console.print(json.dumps(result.to_dict(), indent=2))
    else:
        formatter = ReviewFormatter(console)
        formatter.print_result(result)

        # Show stats if using hybrid mode
        if not legacy and not json_output:
            console.print()
            if result.issues:
                security_issues = len([i for i in result.issues if "security" in i.message.lower() or "SEC" in str(i.message)])
                duplicates = len([i for i in result.issues if "duplicate" in i.message.lower() or "similar" in i.message.lower()])
                other = len(result.issues) - security_issues - duplicates

                console.print("[dim]Analysis breakdown:[/dim]")
                if security_issues:
                    console.print(f"  [red]• {security_issues} security issues (pattern matching)[/red]")
                if duplicates:
                    console.print(f"  [yellow]• {duplicates} duplicate/similar code (semantic search)[/yellow]")
                if other:
                    console.print(f"  [blue]• {other} other issues (LLM insights)[/blue]")
            else:
                print_success("No issues found!")

    if result.has_blocking_issues:
        raise typer.Exit(1)
