"""Context command for CodeSage CLI."""

import json
from pathlib import Path

import typer

from codesage.cli.utils.console import get_console, print_error, print_warning
from codesage.cli.utils.decorators import handle_errors


@handle_errors
def context(
    query: str = typer.Argument(..., help="Task description for context assembly"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: json or markdown",
    ),
    cross_project: bool = typer.Option(
        False,
        "--cross-project",
        help="Include cross-project pattern recommendations (opt-in)",
    ),
    thorough: bool = typer.Option(
        False,
        "--thorough",
        "-t",
        help="Enable thorough analysis with security scanning and full impact assessment",
    ),
) -> None:
    """Get an implementation context pack for a task.

    With --thorough flag, includes deep analysis with security scanning,
    full graph traversal, and comprehensive impact assessment.
    """
    from codesage.utils.config import Config
    from codesage.core.context_provider import ContextProvider

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    if not config.features.context_provider_mode:
        print_warning("Context provider mode is disabled.")
        console.print("  Enable with: [cyan]codesage features enable context_provider_mode[/cyan]")
        raise typer.Exit(1)

    if cross_project and not config.features.cross_project_recommendations:
        print_error("Cross-project recommendations are disabled.")
        console.print("  Enable with: [cyan]codesage features enable cross_project_recommendations[/cyan]")
        raise typer.Exit(1)

    # Handle thorough mode with deep analysis
    if thorough:
        _run_thorough_context(query, config, console, format, cross_project)
        return

    provider = ContextProvider(config)
    context_pack = provider.get_implementation_context(
        task_description=query,
        include_cross_project=cross_project,
    )

    fmt = format.lower()
    if fmt == "json":
        console.print(json.dumps(context_pack.to_dict(), indent=2))
    elif fmt == "markdown":
        console.print(provider.to_markdown(context_pack))
    else:
        print_error(f"Unknown format: {format}")
        console.print("  Use: json or markdown")


def _run_thorough_context(
    query: str,
    config,
    console,
    output_format: str,
    cross_project: bool,
) -> None:
    """Run thorough context analysis with deep analysis.

    Args:
        query: Task description.
        config: CodeSage configuration.
        console: Rich console.
        output_format: Output format (json/markdown).
        cross_project: Include cross-project recommendations.
    """
    from codesage.core.context_provider import ContextProvider
    from codesage.core.deep_analyzer import DeepAnalyzer

    with console.status("[bold green]Running thorough analysis..."):
        # Get standard context
        provider = ContextProvider(config)
        context_pack = provider.get_implementation_context(
            task_description=query,
            include_cross_project=cross_project,
        )

        # Enrich with deep analysis
        analyzer = DeepAnalyzer(config)
        deep_result = analyzer.analyze_sync(query, depth="thorough")

    # Merge deep analysis into context
    context_dict = context_pack.to_dict()
    context_dict["deep_analysis"] = {
        "risk_score": deep_result.risk_score,
        "impact_analysis": deep_result.impact_analysis,
        "security_issues": deep_result.security_issues,
        "recommendations": deep_result.recommendations,
    }

    fmt = output_format.lower()
    if fmt == "json":
        console.print(json.dumps(context_dict, indent=2))
    elif fmt == "markdown":
        # Print standard context
        console.print(provider.to_markdown(context_pack))

        # Add deep analysis section
        console.print("\n## Deep Analysis\n")

        if deep_result.risk_score > 0:
            from codesage.cli.utils.formatters import format_risk_score
            console.print(f"**Risk Score:** {format_risk_score(deep_result.risk_score)}\n")

        if deep_result.impact_analysis:
            blast = deep_result.impact_analysis.get("blast_radius", {})
            console.print("### Impact Analysis")
            console.print(
                f"- Blast radius: {blast.get('direct_callers', 0)} callers, "
                f"{blast.get('dependents', 0)} dependents"
            )
            console.print()

        if deep_result.security_issues:
            console.print("### Security Concerns")
            for issue in deep_result.security_issues[:5]:
                console.print(f"- [{issue.get('severity', 'unknown')}] {issue.get('message', '')}")
            console.print()

        if deep_result.recommendations:
            console.print("### Recommendations")
            for rec in deep_result.recommendations:
                console.print(f"- {rec}")
            console.print()
    else:
        print_error(f"Unknown format: {output_format}")
        console.print("  Use: json or markdown")
