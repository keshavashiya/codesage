"""Review command for CodeSage CLI.

Unified code review that consolidates:
- Code changes review
- Security analysis (--security)
- Code smell detection (--smells)
"""

import json
from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from codesage.cli.utils.console import get_console, print_error, print_warning, print_success
from codesage.cli.utils.decorators import handle_errors


@handle_errors
def review(
    path: str = typer.Argument(".", help="Repository or file path"),
    staged_only: bool = typer.Option(
        False,
        "--staged",
        help="Only review staged changes",
    ),
    # Consolidated analysis options
    security: bool = typer.Option(
        True,
        "--security/--no-security",
        help="Include security analysis (default: enabled)",
    ),
    smells: bool = typer.Option(
        False,
        "--smells",
        "-S",
        help="Include code smell/pattern deviation detection",
    ),
    # Output options
    generate_pr: bool = typer.Option(
        False,
        "--generate-pr-description",
        help="Generate a PR description",
    ),
    severity: str = typer.Option(
        "warning",
        "--severity",
        "-s",
        help="Minimum severity: info, warning, error, critical",
    ),
    max_files: int = typer.Option(
        0,
        "--max-files",
        "-m",
        help="Maximum files to review (0 for unlimited)",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM synthesis (fast mode - static analysis only)",
    ),
    legacy: bool = typer.Option(
        False,
        "--legacy",
        help="Use legacy pure-LLM review",
    ),
    include_tests: bool = typer.Option(
        False,
        "--include-tests",
        help="Include test files in review",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Minimal output (for git hooks)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Review code changes with comprehensive analysis.

    Combines multiple analysis types:
    - Static security analysis (pattern matching)
    - Semantic duplicate detection (against indexed codebase)
    - Code smell detection (pattern deviations)
    - LLM synthesis for insights (optional)

    Examples:
      codesage review                    # Review all uncommitted changes
      codesage review --staged           # Review staged changes only
      codesage review --security         # Focus on security issues
      codesage review --smells           # Include code smell detection
      codesage review src/auth.py        # Review specific file
    """
    from codesage.review import ReviewFormatter
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    # Check if path is a file
    review_file = None
    if project_path.is_file():
        review_file = project_path
        project_path = project_path.parent

    # Find project root
    while project_path != project_path.parent:
        if (project_path / ".codesage").exists():
            break
        project_path = project_path.parent

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    # Validate severity
    valid_severities = {"info", "warning", "error", "critical"}
    if severity not in valid_severities:
        print_error(f"Invalid severity. Choose: {', '.join(valid_severities)}")
        raise typer.Exit(1)

    # Choose analyzer
    if legacy:
        from codesage.review import ReviewAnalyzer
        if not quiet:
            console.print("[dim]Using legacy LLM-only review...[/dim]")
        analyzer = ReviewAnalyzer(config=config, repo_path=project_path)
    else:
        from codesage.review import HybridReviewAnalyzer
        mode_parts = []
        if security:
            mode_parts.append("security")
        mode_parts.append("semantic")
        if not no_llm:
            mode_parts.append("LLM")
        if smells:
            mode_parts.append("smells")

        if not quiet:
            console.print(f"[dim]Review mode: {' + '.join(mode_parts)}[/dim]")
        analyzer = HybridReviewAnalyzer(config=config, repo_path=project_path)

    if not quiet:
        console.print("[dim]Analyzing changes...[/dim]")

    # Get changes
    if review_file:
        # Single file review
        from codesage.review.models import FileChange
        changes = [FileChange(path=review_file, status="M", content=review_file.read_text())]
    elif staged_only:
        changes = analyzer.get_staged_changes()
    else:
        changes = analyzer.get_all_changes()

    if not changes:
        if not quiet:
            print_warning("No changes to review")
        return

    # Filter test files
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
        if excluded > 0 and not quiet:
            console.print(f"[dim]Excluded {excluded} test files[/dim]")

    # Limit files
    total_files = len(changes)
    if max_files > 0 and len(changes) > max_files:
        if not quiet:
            console.print(f"[yellow]Limiting to {max_files} of {total_files} files[/yellow]")
        changes = changes[:max_files]

    if not quiet:
        console.print(f"[dim]Reviewing {len(changes)} files...[/dim]")

    # Run review
    if quiet:
        # Silent mode for git hooks
        if legacy:
            result = analyzer.review_changes(changes=changes, generate_pr_description=generate_pr)
        else:
            result = analyzer.review_changes(
                changes=changes,
                generate_pr_description=generate_pr,
                use_llm_synthesis=not no_llm,
            )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing...", total=len(changes))

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

    # Code smell detection (if enabled)
    if smells and not legacy:
        try:
            from codesage.review.smells import PatternDeviationDetector
            from codesage.review.models import IssueSeverity, ReviewIssue

            if not quiet:
                console.print("[dim]Running smell detection...[/dim]")

            detector = PatternDeviationDetector(config)
            smell_issues = []

            for change in changes:
                if change.status == "D":
                    continue
                try:
                    found_smells = detector.detect_file(change.path)
                    for s in found_smells:
                        smell_issues.append(
                            ReviewIssue(
                                severity=IssueSeverity.WARNING,
                                file=change.path,
                                line=s.line,
                                message=f"[Smell] {s.message}",
                                suggestion=s.suggestion,
                            )
                        )
                except Exception:
                    pass

            result.issues.extend(smell_issues)

            if not quiet and smell_issues:
                console.print(f"[yellow]Found {len(smell_issues)} code smells[/yellow]")

        except ImportError:
            if not quiet:
                console.print("[dim]Smell detection not available[/dim]")
        except Exception as e:
            if not quiet:
                console.print(f"[yellow]Smell detection error: {e}[/yellow]")

    # Also check for code smell detection feature flag (backwards compat)
    if config.features.code_smell_detection and not smells:
        try:
            from codesage.review.smells import PatternDeviationDetector
            from codesage.review.models import IssueSeverity, ReviewIssue

            detector = PatternDeviationDetector(config)
            smell_issues = []
            for change in changes:
                if change.status == "D":
                    continue
                try:
                    found_smells = detector.detect_file(change.path)
                    for s in found_smells:
                        smell_issues.append(
                            ReviewIssue(
                                severity=IssueSeverity.WARNING,
                                file=change.path,
                                line=s.line,
                                message=f"[Smell] {s.message}",
                                suggestion=s.suggestion,
                            )
                        )
                except Exception:
                    pass
            result.issues.extend(smell_issues)
        except Exception:
            pass

    # Filter by severity
    severity_order = {"info": 0, "warning": 1, "error": 2, "critical": 3}
    min_severity = severity_order.get(severity, 1)

    if min_severity > 0:
        original_issues = len(result.issues)
        result.issues = [
            i for i in result.issues
            if severity_order.get(str(i.severity.value).lower() if hasattr(i.severity, 'value') else str(i.severity).lower(), 0) >= min_severity
        ]
        filtered = original_issues - len(result.issues)
        if filtered > 0 and not quiet:
            console.print(f"[dim]Filtered {filtered} issues below {severity} severity[/dim]")

    # Output results
    if json_output:
        console.print(json.dumps(result.to_dict(), indent=2))
    elif quiet:
        # Minimal output for hooks
        if result.issues:
            for issue in result.issues[:5]:
                console.print(f"{issue.severity.value}: {issue.file}:{issue.line} - {issue.message}")
            if len(result.issues) > 5:
                console.print(f"... and {len(result.issues) - 5} more issues")
    else:
        formatter = ReviewFormatter(console)
        formatter.print_result(result)

        # Show analysis breakdown
        if not legacy:
            console.print()
            if result.issues:
                security_issues = len([i for i in result.issues if "security" in i.message.lower() or "SEC" in str(i.message)])
                smell_issues = len([i for i in result.issues if "[smell]" in i.message.lower()])
                duplicates = len([i for i in result.issues if "duplicate" in i.message.lower() or "similar" in i.message.lower()])
                other = len(result.issues) - security_issues - duplicates - smell_issues

                console.print("[dim]Analysis breakdown:[/dim]")
                if security_issues:
                    console.print(f"  [red]• {security_issues} security issues[/red]")
                if smell_issues:
                    console.print(f"  [yellow]• {smell_issues} code smells[/yellow]")
                if duplicates:
                    console.print(f"  [cyan]• {duplicates} duplicates/similar code[/cyan]")
                if other:
                    console.print(f"  [blue]• {other} other issues[/blue]")
            else:
                print_success("No issues found!")

    if result.has_blocking_issues:
        raise typer.Exit(1)
