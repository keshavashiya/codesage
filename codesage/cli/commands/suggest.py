"""Suggest command for CodeSage CLI."""

from pathlib import Path

import typer
from rich.syntax import Syntax

from codesage.cli.utils.console import get_console, print_error
from codesage.cli.utils.decorators import handle_errors
from codesage.cli.utils.options import MAX_QUERY_LENGTH


@handle_errors
def suggest(
    query: str = typer.Argument(..., help="What are you looking for?"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of suggestions"),
    min_similarity: float = typer.Option(
        0.2,
        "--min-similarity",
        "-s",
        help="Minimum similarity threshold (0-1)",
    ),
    no_explain: bool = typer.Option(
        False,
        "--no-explain",
        help="Skip LLM explanations (faster)",
    ),
    with_patterns: bool = typer.Option(
        False,
        "--with-patterns",
        "-P",
        help="Include matching patterns from developer memory",
    ),
    cross_project: bool = typer.Option(
        False,
        "--cross-project",
        help="Include cross-project pattern recommendations (opt-in)",
    ),
    show_context: bool = typer.Option(
        False,
        "--context",
        "-c",
        help="Show caller/callee context from graph",
    ),
    deep: bool = typer.Option(
        False,
        "--deep",
        "-d",
        help="Enable multi-agent deep analysis (semantic + graph + patterns + security)",
    ),
) -> None:
    """Get code suggestions based on natural language query.

    Uses semantic search to find relevant code.
    With --deep flag, runs parallel multi-agent analysis including
    graph traversal, pattern matching, and security scanning.
    """
    from codesage.core.suggester import Suggester
    from codesage.utils.config import Config

    console = get_console()

    # Validate query
    query = query.strip()
    if not query:
        print_error("Query cannot be empty")
        raise typer.Exit(1)
    if len(query) > MAX_QUERY_LENGTH:
        print_error(f"Query too long (max {MAX_QUERY_LENGTH} chars)")
        raise typer.Exit(1)

    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    # Handle deep analysis mode
    if deep:
        _run_deep_analysis(query, config, console, limit)
        return

    console.print(f"\n[dim]Searching for:[/dim] {query}\n")

    suggester = Suggester(config)

    # Choose search method based on flags
    matching_patterns = []
    recommendations = []
    cross_project_recommendations = []

    if with_patterns or cross_project:
        if cross_project and not config.features.cross_project_recommendations:
            print_error("Cross-project recommendations are disabled.")
            console.print("  Enable with: [cyan]codesage features enable cross_project_recommendations[/cyan]")
            raise typer.Exit(1)
        result = suggester.find_similar_with_patterns(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=not no_explain,
            include_cross_project=cross_project,
        )
        suggestions = result["suggestions"]
        matching_patterns = result.get("matching_patterns", [])
        recommendations = result.get("recommendations", [])
        cross_project_recommendations = result.get("cross_project_recommendations", [])
    else:
        suggestions = suggester.find_similar(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=not no_explain,
            include_graph_context=show_context,
        )

    if not suggestions and not matching_patterns:
        console.print("[yellow]No suggestions found.[/yellow]")
        console.print("\n[dim]Tips:[/dim]")
        console.print("  • Try different search terms")
        console.print("  • Lower --min-similarity threshold")
        console.print("  • Run [cyan]codesage index[/cyan] to update the index\n")
        return

    # Show matching patterns first if any (deduplicated by name)
    if matching_patterns:
        console.print("[bold magenta]Matching Patterns from Memory[/bold magenta]")
        seen_names = set()
        displayed = 0
        for pattern in matching_patterns:
            name = pattern.get('name', 'Unknown')
            if name in seen_names:
                continue
            seen_names.add(name)
            console.print(f"  - [cyan]{name}[/cyan]")
            if pattern.get('description'):
                console.print(f"    [dim]{pattern['description'][:80]}[/dim]")
            displayed += 1
            if displayed >= 3:
                break
        console.print()

    # Show recommendations
    if recommendations:
        console.print("[bold yellow]💡 Recommendations[/bold yellow]")
        for rec in recommendations[:2]:
            pattern = rec.get("pattern", {})
            console.print(f"  • {rec.get('reason', 'Consider this pattern')}")
            if pattern.get("example_code"):
                console.print(f"    [dim]Example: {pattern['example_code'][:50]}...[/dim]")
        console.print()

    if cross_project_recommendations:
        console.print("[bold cyan]📦 Cross-Project Recommendations[/bold cyan]")
        for rec in cross_project_recommendations[:3]:
            name = rec.get("pattern_name") or rec.get("pattern_id", "pattern")
            reason = rec.get("reason", "")
            console.print(f"  • {name} {f'- {reason}' if reason else ''}")
        console.print()

    for i, suggestion in enumerate(suggestions, 1):
        # Header
        console.print(f"[bold blue]{i}. {suggestion.file}:{suggestion.line}[/bold blue]")
        console.print(
            f"[dim]Similarity: {suggestion.similarity:.0%} | "
            f"Type: {suggestion.element_type}"
            f"{' | ' + suggestion.name if suggestion.name else ''}[/dim]"
        )

        # Code with syntax highlighting
        syntax = Syntax(
            suggestion.code,
            suggestion.language,
            theme="monokai",
            line_numbers=True,
            start_line=suggestion.line,
            word_wrap=True,
        )
        console.print(syntax)

        # Show graph context if requested
        if show_context and suggestion.has_graph_context():
            if suggestion.callers:
                caller_names = [c.get("name", "?") for c in suggestion.callers[:3]]
                console.print(f"  [dim]Called by: {', '.join(caller_names)}[/dim]")
            if suggestion.callees:
                callee_names = [c.get("name", "?") for c in suggestion.callees[:3]]
                console.print(f"  [dim]Calls: {', '.join(callee_names)}[/dim]")
            if suggestion.superclasses:
                parent_names = [p.get("name", "?") for p in suggestion.superclasses]
                console.print(f"  [dim]Inherits from: {', '.join(parent_names)}[/dim]")
            if suggestion.dependencies:
                dep_names = [d.get("name", "?") for d in suggestion.dependencies[:3]]
                console.print(f"  [dim]Depends on: {', '.join(dep_names)}[/dim]")
            if suggestion.impact_score is not None:
                console.print(f"  [dim]Impact score: {suggestion.impact_score:.2f}[/dim]")

        # Explanation
        if suggestion.explanation:
            console.print(f"[italic]💡 {suggestion.explanation}[/italic]")

        console.print()

    console.print(f"[dim]Found {len(suggestions)} suggestion(s)[/dim]\n")


def _run_deep_analysis(query: str, config, console, limit: int) -> None:
    """Run deep multi-agent analysis and display results.

    Args:
        query: Search query.
        config: CodeSage configuration.
        console: Rich console for output.
        limit: Result limit.
    """
    from rich.panel import Panel
    from rich.table import Table
    from codesage.core.deep_analyzer import DeepAnalyzer
    from codesage.cli.utils.formatters import format_risk_score, format_impact_score

    console.print(f"\n[bold cyan]Deep Analysis:[/bold cyan] {query}\n")

    with console.status("[bold green]Running parallel analysis..."):
        analyzer = DeepAnalyzer(config)
        # Use medium depth for CLI, thorough for larger limits
        depth = "thorough" if limit > 5 else "medium"
        result = analyzer.analyze_sync(query, depth=depth)

    # Show risk score
    if result.risk_score > 0:
        console.print(f"[bold]Risk Score:[/bold] {format_risk_score(result.risk_score)}\n")

    # Show semantic results
    if result.semantic_results:
        console.print("[bold blue]Matching Code[/bold blue]")
        for i, item in enumerate(result.semantic_results[:limit], 1):
            console.print(
                f"  {i}. [cyan]{item.get('file')}:{item.get('line')}[/cyan] "
                f"({item.get('similarity', 0):.0%} match)"
            )
            if item.get("name"):
                console.print(f"     [dim]{item.get('type', 'element')}: {item.get('name')}[/dim]")
        console.print()

    # Show impact analysis
    impact = result.impact_analysis
    if impact.get("blast_radius"):
        blast = impact["blast_radius"]
        console.print("[bold yellow]Impact Analysis[/bold yellow]")
        console.print(
            f"  Blast radius: {blast.get('direct_callers', 0)} callers, "
            f"{blast.get('dependents', 0)} dependents"
        )
        if impact.get("highest_impact_elements"):
            console.print("  Highest impact elements:")
            for elem in impact["highest_impact_elements"]:
                console.print(f"    - {elem.get('id')}: {format_impact_score(elem.get('score'))}")
        console.print()

    # Show patterns
    if result.patterns:
        console.print("[bold magenta]Matching Patterns[/bold magenta]")
        for pattern in result.patterns[:3]:
            conf = pattern.get("confidence", 0)
            console.print(f"  - [cyan]{pattern.get('name')}[/cyan] (confidence: {conf:.0%})")
            if pattern.get("description"):
                console.print(f"    [dim]{pattern['description'][:80]}[/dim]")
        console.print()

    # Show security issues
    if result.security_issues:
        console.print("[bold red]Security Issues[/bold red]")
        sec_summary = impact.get("security_summary", {})
        console.print(
            f"  Found {sec_summary.get('total_issues', len(result.security_issues))} issues: "
            f"{sec_summary.get('critical', 0)} critical, {sec_summary.get('high', 0)} high"
        )
        for issue in result.security_issues[:5]:
            severity = issue.get("severity", "unknown")
            if isinstance(severity, str):
                sev_str = severity.upper()
            else:
                sev_str = str(severity)
            console.print(
                f"  [{_severity_color(sev_str)}]{sev_str}[/{_severity_color(sev_str)}] "
                f"{issue.get('file')}:{issue.get('line')} - {issue.get('message', '')[:60]}"
            )
        console.print()

    # Show recommendations
    if result.recommendations:
        console.print("[bold green]Recommendations[/bold green]")
        for rec in result.recommendations:
            console.print(f"  - {rec}")
        console.print()

    # Show errors if any
    if result.errors:
        console.print("[dim]Analysis notes:[/dim]")
        for error in result.errors:
            console.print(f"  [dim yellow]{error}[/dim yellow]")
        console.print()


def _severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        "CRITICAL": "red bold",
        "HIGH": "red",
        "MEDIUM": "yellow",
        "LOW": "blue",
        "INFO": "dim",
    }
    return colors.get(severity.upper(), "white")
