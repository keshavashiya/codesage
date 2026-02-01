"""Profile command group for CodeSage CLI.

Provides commands for managing developer profile, preferences,
patterns, and style analysis.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from codesage.cli.utils.console import get_console
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="Developer profile and pattern management")


@app.command("show")
@handle_errors
def show(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show developer profile summary."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    summary = profile.get_profile_summary()

    if json_output:
        console.print(json.dumps(summary, indent=2, default=str))
        return

    # Summary panel
    console.print(Panel.fit(
        "[bold]Developer Profile[/bold]",
        border_style="blue",
    ))

    # Stats
    console.print(f"\n[cyan]Patterns Learned:[/cyan] {summary['total_patterns']}")
    console.print(f"[cyan]Projects Analyzed:[/cyan] {summary['total_projects']}")

    # Patterns by category
    if summary.get("patterns_by_category"):
        console.print("\n[bold]Patterns by Category:[/bold]")
        for cat, count in summary["patterns_by_category"].items():
            console.print(f"  {cat}: {count}")

    # Top patterns
    if verbose and summary.get("top_patterns"):
        console.print("\n[bold]Top Patterns:[/bold]")
        for p in summary["top_patterns"][:5]:
            confidence = p.get("confidence_score", 0)
            bar = "█" * int(confidence * 5)
            console.print(f"  {p['name']} [{bar}] ({confidence:.0%})")

    # Projects
    if summary.get("projects"):
        console.print("\n[bold]Projects:[/bold]")
        for p in summary["projects"][:5]:
            console.print(f"  - {p['name']} ({p['total_elements']} elements)")

    # Interaction stats
    if verbose and summary.get("interaction_stats"):
        stats = summary["interaction_stats"]
        console.print(f"\n[bold]Interactions:[/bold] {stats.get('total', 0)}")
        if stats.get("acceptance_rate"):
            console.print(f"  Acceptance rate: {stats['acceptance_rate']:.0%}")


@app.command("set")
@handle_errors
def set_preference(
    key: str = typer.Argument(..., help="Preference key"),
    value: str = typer.Argument(..., help="Preference value"),
) -> None:
    """Set a profile preference."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    # Try to parse as JSON for complex values
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    profile.set_preference(key, parsed_value)
    console.print(f"[green]✓[/green] Set {key} = {parsed_value}")


@app.command("get")
@handle_errors
def get_preference(
    key: str = typer.Argument(..., help="Preference key"),
) -> None:
    """Get a profile preference."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    value = profile.get_preference(key)
    if value is None:
        console.print(f"[yellow]Preference '{key}' not set[/yellow]")
    else:
        console.print(f"{key} = {value}")


@app.command("reset")
@handle_errors
def reset(
    keep_preferences: bool = typer.Option(
        False,
        "--keep-preferences",
        help="Keep preferences but clear patterns and projects",
    ),
    confirm: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt"
    ),
) -> None:
    """Reset the developer profile."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()

    if not confirm:
        message = (
            "This will clear all learned patterns and projects."
            if keep_preferences
            else "This will clear your entire profile including preferences."
        )
        confirmed = typer.confirm(f"{message} Continue?")
        if not confirmed:
            console.print("[yellow]Aborted[/yellow]")
            raise typer.Exit(0)

    profile = DeveloperProfileManager()
    profile.reset(keep_preferences=keep_preferences)

    if keep_preferences:
        console.print("[green]✓[/green] Reset patterns and projects (kept preferences)")
    else:
        console.print("[green]✓[/green] Reset entire profile")


@app.command("patterns")
@handle_errors
def patterns(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", help="Minimum confidence threshold"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum patterns to show"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """List learned patterns."""
    from codesage.memory import DeveloperProfileManager, PatternCategory

    console = get_console()
    profile = DeveloperProfileManager()

    # Validate category
    valid_categories = [c.value for c in PatternCategory]
    if category and category not in valid_categories:
        console.print(f"[red]Invalid category. Valid: {', '.join(valid_categories)}[/red]")
        raise typer.Exit(1)

    patterns_list = profile.get_patterns(
        category=category,
        min_confidence=min_confidence,
        limit=limit,
    )

    if json_output:
        console.print(json.dumps([p.to_dict() for p in patterns_list], indent=2, default=str))
        return

    if not patterns_list:
        console.print("[yellow]No patterns found[/yellow]")
        return

    table = Table(title="Learned Patterns")
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("Confidence", justify="right")
    table.add_column("Occurrences", justify="right")

    for p in patterns_list:
        confidence_bar = "█" * int(p.confidence_score * 5)
        table.add_row(
            p.id[:8] + "...",
            p.name,
            p.category.value,
            f"[green]{confidence_bar}[/green] {p.confidence_score:.0%}",
            str(p.occurrence_count),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(patterns_list)} patterns[/dim]")


@app.command("pattern")
@handle_errors
def pattern(
    pattern_id: str = typer.Argument(..., help="Pattern ID"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show details for a single pattern."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    context = profile.get_pattern(pattern_id)

    if not context or not context.get("pattern"):
        console.print(f"[red]Pattern '{pattern_id}' not found[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print(json.dumps(context, indent=2, default=str))
        return

    p = context["pattern"]

    console.print(Panel.fit(
        f"[bold]{p['name']}[/bold]\n{p['description']}",
        title="Pattern Details",
        border_style="cyan",
    ))

    console.print(f"\n[bold]Category:[/bold] {p['category']}")
    console.print(f"[bold]Confidence:[/bold] {p['confidence_score']:.0%}")
    console.print(f"[bold]Occurrences:[/bold] {p['occurrence_count']}")

    if p.get("pattern_text"):
        console.print(f"\n[bold]Pattern:[/bold]")
        console.print(f"  [dim]{p['pattern_text']}[/dim]")

    if p.get("examples"):
        console.print(f"\n[bold]Examples:[/bold]")
        for ex in p["examples"][:5]:
            console.print(f"  - [green]{ex}[/green]")

    if context.get("co_occurring_patterns"):
        console.print(f"\n[bold]Co-occurring Patterns:[/bold]")
        for cp in context["co_occurring_patterns"][:5]:
            console.print(f"  - {cp['name']} ({cp['correlation']:.0%})")

    if context.get("source_projects"):
        console.print(f"\n[bold]Learned From:[/bold]")
        for proj in context["source_projects"][:5]:
            console.print(f"  - {proj['name']}")


@app.command("graph")
@handle_errors
def graph(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show pattern relationship graph summary."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    summary = profile.get_graph_summary()

    if json_output:
        console.print(json.dumps(summary, indent=2, default=str))
        return

    console.print(Panel.fit(
        "[bold]Pattern Relationship Graph[/bold]",
        border_style="blue",
    ))

    # Node counts
    if summary.get("node_counts"):
        console.print("\n[bold]Node Counts:[/bold]")
        for node_type, count in summary["node_counts"].items():
            console.print(f"  {node_type}: {count}")

    # Relationship counts
    if summary.get("relationship_counts"):
        console.print("\n[bold]Relationship Counts:[/bold]")
        for rel_type, count in summary["relationship_counts"].items():
            if count > 0:
                console.print(f"  {rel_type}: {count}")

    # Cross-project patterns
    if summary.get("cross_project_patterns"):
        console.print("\n[bold]Cross-Project Patterns:[/bold]")
        for cp in summary["cross_project_patterns"][:5]:
            console.print(
                f"  - {cp['name']} (in {cp.get('project_count', 0)} projects)"
            )


@app.command("export")
@handle_errors
def export(
    output: str = typer.Argument(..., help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json)"),
) -> None:
    """Export pattern graph to file."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    graph_data = profile.export_graph()

    output_path = Path(output)
    with open(output_path, "w") as f:
        json.dump(graph_data, f, indent=2, default=str)

    console.print(f"[green]✓[/green] Exported graph to {output_path}")
    console.print(f"  Patterns: {len(graph_data.get('patterns', []))}")
    console.print(f"  Projects: {len(graph_data.get('projects', []))}")
    console.print(f"  Relationships: {len(graph_data.get('relationships', []))}")


@app.command("similar-patterns")
@handle_errors
def similar_patterns(
    pattern_id: str = typer.Argument(..., help="Pattern ID to find similar patterns for"),
    limit: int = typer.Option(5, "--limit", "-n", help="Maximum results"),
) -> None:
    """Find patterns similar to a given pattern."""
    from codesage.memory import MemoryManager

    console = get_console()
    memory = MemoryManager()

    # Get co-occurring patterns from graph
    try:
        cooccurring = memory.memory_graph.get_cooccurring_patterns(pattern_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not cooccurring:
        console.print("[yellow]No similar patterns found[/yellow]")
        return

    table = Table(title="Co-occurring Patterns")
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("Correlation", justify="right")

    for p in cooccurring[:limit]:
        table.add_row(
            p["id"][:8] + "..." if len(p["id"]) > 8 else p["id"],
            p["name"],
            p.get("category", "-"),
            f"{p.get('correlation', 0):.0%}",
        )

    console.print(table)


@app.command("projects")
@handle_errors
def projects(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """List all tracked projects."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    projects_list = profile.get_projects()

    if json_output:
        console.print(json.dumps([p.to_dict() for p in projects_list], indent=2, default=str))
        return

    if not projects_list:
        console.print("[yellow]No projects tracked[/yellow]")
        return

    table = Table(title="Tracked Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Language")
    table.add_column("Files", justify="right")
    table.add_column("Elements", justify="right")
    table.add_column("Patterns", justify="right")
    table.add_column("Last Indexed")

    for p in projects_list:
        last_indexed = (
            p.last_indexed.strftime("%Y-%m-%d") if p.last_indexed else "-"
        )
        table.add_row(
            p.name,
            p.language,
            str(p.total_files),
            str(p.total_elements),
            str(p.patterns_learned),
            last_indexed,
        )

    console.print(table)


@app.command("project")
@handle_errors
def project(
    name: str = typer.Argument(..., help="Project name"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show details for a single project."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    proj = profile.get_project(name)

    if not proj:
        console.print(f"[red]Project '{name}' not found[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print(json.dumps(proj.to_dict(), indent=2, default=str))
        return

    console.print(Panel.fit(
        f"[bold]{proj.name}[/bold]\n{proj.path}",
        title="Project Details",
        border_style="cyan",
    ))

    console.print(f"\n[bold]Language:[/bold] {proj.language}")
    console.print(f"[bold]Files:[/bold] {proj.total_files}")
    console.print(f"[bold]Elements:[/bold] {proj.total_elements}")
    console.print(f"[bold]Patterns Learned:[/bold] {proj.patterns_learned}")

    if proj.first_indexed:
        console.print(f"[bold]First Indexed:[/bold] {proj.first_indexed.strftime('%Y-%m-%d %H:%M')}")
    if proj.last_indexed:
        console.print(f"[bold]Last Indexed:[/bold] {proj.last_indexed.strftime('%Y-%m-%d %H:%M')}")

    # Show similar projects
    similar = profile.find_similar_projects(name)
    if similar:
        console.print(f"\n[bold]Similar Projects:[/bold]")
        for s in similar[:5]:
            console.print(f"  - {s['name']} ({s.get('similarity', 0):.0%} similar)")


@app.command("similar-projects")
@handle_errors
def similar_projects(
    name: str = typer.Argument(..., help="Project name"),
    limit: int = typer.Option(5, "--limit", "-n", help="Maximum results"),
) -> None:
    """Find projects similar to a given project."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    similar = profile.find_similar_projects(name)

    if not similar:
        console.print("[yellow]No similar projects found[/yellow]")
        return

    table = Table(title=f"Projects Similar to {name}")
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Similarity", justify="right")

    for p in similar[:limit]:
        table.add_row(
            p["name"],
            p.get("path", "-"),
            f"{p.get('similarity', 0):.0%}",
        )

    console.print(table)


@app.command("style-guide")
@handle_errors
def style_guide(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save to file"),
) -> None:
    """Generate a markdown style guide from learned patterns."""
    from codesage.memory import DeveloperProfileManager

    console = get_console()
    profile = DeveloperProfileManager()

    guide = profile.generate_style_guide()

    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            f.write(guide)
        console.print(f"[green]✓[/green] Saved style guide to {output_path}")
    else:
        console.print(guide)


@app.command("learn")
@handle_errors
def learn(
    project_path: str = typer.Argument(".", help="Project directory to learn from"),
) -> None:
    """Learn patterns from an already-indexed project without reindexing.

    Extracts style patterns, structures, and preferences from existing
    indexed code elements in the database.
    """
    from codesage.memory.hooks import MemoryHooks
    from codesage.memory.memory_manager import MemoryManager
    from codesage.storage.database import Database
    from codesage.utils.config import Config
    from codesage.llm.embeddings import EmbeddingService

    console = get_console()
    path = Path(project_path).resolve()

    try:
        config = Config.load(path)
    except FileNotFoundError:
        console.print("[red]Project not initialized. Run 'codesage init' first.[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Learning patterns from {config.project_name}...[/cyan]\n")

    # Load existing elements from database
    db = Database(config.storage.db_path)
    elements = db.get_all_elements()

    if not elements:
        console.print("[yellow]No indexed elements found. Run 'codesage index' first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Found {len(elements)} code elements")

    # Initialize memory hooks with embeddings
    embedder = EmbeddingService(config.llm, config.cache_dir)
    hooks = MemoryHooks(
        embedding_fn=embedder.embedder,
        enabled=True,
    )

    # Convert elements to dicts and learn
    element_dicts = [el.to_dict() for el in elements]
    patterns = hooks.on_elements_indexed(
        element_dicts,
        config.project_name,
        config.project_path,
    )

    # Notify project indexed
    hooks.on_project_indexed(
        project_name=config.project_name,
        project_path=config.project_path,
        total_files=db.get_stats()["files"],
        total_elements=len(elements),
    )

    stats = hooks.get_stats()

    console.print()
    console.print(Panel(
        f"""[bold]Elements processed:[/bold] {stats['elements_processed']}
[bold]Patterns learned:[/bold] {stats['patterns_learned']}
[bold]Files processed:[/bold] {stats['files_processed']}""",
        title="Learning Complete",
        border_style="green",
    ))


@app.command("mine")
@handle_errors
def mine(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed analysis"),
) -> None:
    """Analyze cross-project patterns and generate recommendations.

    Mines patterns across all tracked projects to find:
    - Common patterns used across projects
    - Pattern clusters and relationships
    - Recommendations for pattern adoption
    """
    from codesage.memory.pattern_miner import PatternMiner
    from codesage.memory.memory_manager import MemoryManager

    console = get_console()

    console.print("[cyan]Mining cross-project patterns...[/cyan]\n")

    memory = MemoryManager()
    miner = PatternMiner(memory)

    # Run mining
    analysis = miner.analyze_all_projects()

    if json_output:
        console.print(json.dumps(analysis, indent=2, default=str))
        return

    # Show cross-project patterns
    cross_patterns = analysis.get("cross_project_patterns", [])
    if cross_patterns:
        console.print("[bold]Cross-Project Patterns[/bold]")
        console.print(f"[dim]Patterns that appear in multiple projects[/dim]\n")

        table = Table()
        table.add_column("Pattern", style="cyan")
        table.add_column("Projects", justify="right")
        table.add_column("Confidence", justify="right")

        for p in cross_patterns[:10]:
            table.add_row(
                p.get("name", "Unknown"),
                str(p.get("project_count", 0)),
                f"{p.get('avg_confidence', 0):.0%}",
            )

        console.print(table)
        console.print()

    # Show pattern clusters
    clusters = analysis.get("pattern_clusters", [])
    if clusters and verbose:
        console.print("[bold]Pattern Clusters[/bold]")
        console.print("[dim]Groups of related patterns that often appear together[/dim]\n")

        for i, cluster in enumerate(clusters[:5], 1):
            pattern_names = [p.get("name", "?") for p in cluster.get("patterns", [])]
            console.print(f"[cyan]Cluster {i}:[/cyan] {', '.join(pattern_names[:5])}")

        console.print()

    # Show recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        console.print("[bold]Recommendations[/bold]")
        console.print("[dim]Patterns you might want to adopt[/dim]\n")

        for rec in recommendations[:5]:
            pattern = rec.get("pattern", {})
            console.print(f"  [green]>[/green] {rec.get('reason', 'Consider this pattern')}")
            console.print(f"    Pattern: [cyan]{pattern.get('name', 'Unknown')}[/cyan]")
            if rec.get("source_project"):
                console.print(f"    Source: {rec['source_project']}")
            console.print()

    # Summary
    console.print(Panel(
        f"""[bold]Projects analyzed:[/bold] {analysis.get('project_count', 0)}
[bold]Total patterns:[/bold] {analysis.get('total_patterns', 0)}
[bold]Cross-project patterns:[/bold] {len(cross_patterns)}
[bold]Pattern clusters:[/bold] {len(clusters)}
[bold]Recommendations:[/bold] {len(recommendations)}""",
        title="Mining Summary",
        border_style="blue",
    ))
