"""Search command for CodeSage CLI.

This is the primary search interface (renamed from 'suggest' for clarity).
"""

from pathlib import Path

import typer
from rich.syntax import Syntax

from codesage.cli.utils.console import get_console, print_error
from codesage.cli.utils.decorators import handle_errors
from codesage.cli.utils.options import MAX_QUERY_LENGTH


@handle_errors
def search(
    query: str = typer.Argument(..., help="What are you looking for?"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
    min_similarity: float = typer.Option(
        0.2,
        "--min-similarity",
        "-s",
        help="Minimum similarity threshold (0-1)",
    ),
    depth: str = typer.Option(
        "medium",
        "--depth",
        "-d",
        help="Search depth: quick (semantic only), medium (+graph), thorough (+patterns)",
    ),
    no_explain: bool = typer.Option(
        False,
        "--no-explain",
        help="Skip LLM explanations (faster)",
    ),
    with_patterns: bool = typer.Option(
        False,
        "--patterns",
        "-P",
        help="Include matching patterns from developer memory",
    ),
    show_context: bool = typer.Option(
        False,
        "--context",
        "-c",
        help="Show caller/callee context from graph",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Search codebase using semantic similarity.

    Uses vector embeddings to find relevant code snippets
    based on natural language queries.

    Examples:
      codesage search "authentication logic"
      codesage search "database query" --depth thorough
      codesage search "error handling" --patterns
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

    # Validate depth
    valid_depths = {"quick", "medium", "thorough"}
    if depth not in valid_depths:
        print_error(f"Invalid depth. Choose: {', '.join(valid_depths)}")
        raise typer.Exit(1)

    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    if not json_output:
        console.print(f"\n[dim]Searching for:[/dim] {query}")
        console.print(f"[dim]Depth: {depth}[/dim]\n")

    suggester = Suggester(config)

    # Adjust search based on depth
    include_patterns = with_patterns or depth == "thorough"
    include_graph = show_context or depth in ("medium", "thorough")

    # Choose search method
    if include_patterns:
        result = suggester.find_similar_with_patterns(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=not no_explain,
            include_cross_project=False,
        )
        suggestions = result["suggestions"]
        matching_patterns = result.get("matching_patterns", [])
        recommendations = result.get("recommendations", [])
    else:
        suggestions = suggester.find_similar(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=not no_explain,
            include_graph_context=include_graph,
        )
        matching_patterns = []
        recommendations = []

    # JSON output
    if json_output:
        import json
        output = {
            "query": query,
            "depth": depth,
            "results": [
                {
                    "file": str(s.file),
                    "line": s.line,
                    "name": s.name,
                    "type": s.element_type,
                    "similarity": s.similarity,
                    "code": s.code[:500],
                }
                for s in suggestions
            ],
            "patterns": matching_patterns,
        }
        console.print(json.dumps(output, indent=2))
        return

    if not suggestions and not matching_patterns:
        console.print("[yellow]No results found.[/yellow]")
        console.print("\n[dim]Tips:[/dim]")
        console.print("  • Try different search terms")
        console.print("  • Lower --min-similarity threshold")
        console.print("  • Run [cyan]codesage index[/cyan] to update\n")
        return

    # Show matching patterns first
    if matching_patterns:
        console.print("[bold magenta]🎯 Matching Patterns[/bold magenta]")
        for pattern in matching_patterns[:3]:
            console.print(f"  • [cyan]{pattern.get('name', 'Unknown')}[/cyan]")
            if pattern.get('description'):
                console.print(f"    [dim]{pattern['description'][:80]}[/dim]")
        console.print()

    # Show recommendations
    if recommendations:
        console.print("[bold yellow]💡 Recommendations[/bold yellow]")
        for rec in recommendations[:2]:
            console.print(f"  • {rec.get('reason', 'Consider this pattern')}")
        console.print()

    # Show results
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

        # Graph context
        if include_graph and suggestion.has_graph_context():
            if suggestion.callers:
                caller_names = [c.get("name", "?") for c in suggestion.callers[:3]]
                console.print(f"  [dim]Called by: {', '.join(caller_names)}[/dim]")
            if suggestion.callees:
                callee_names = [c.get("name", "?") for c in suggestion.callees[:3]]
                console.print(f"  [dim]Calls: {', '.join(callee_names)}[/dim]")

        # Explanation
        if suggestion.explanation:
            console.print(f"[italic]💡 {suggestion.explanation}[/italic]")

        console.print()

    console.print(f"[dim]Found {len(suggestions)} result(s)[/dim]\n")
