"""Stats command for CodeSage CLI."""

from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from codesage.cli.utils.console import get_console, print_error
from codesage.cli.utils.decorators import handle_errors


@handle_errors
def stats(
    path: str = typer.Argument(".", help="Project directory"),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed statistics including memory and graph",
    ),
) -> None:
    """Show index statistics."""
    from codesage.storage.database import Database
    from codesage.storage.manager import StorageManager
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    db = Database(config.storage.db_path)
    db_stats = db.get_stats()

    # Build languages display
    langs_display = ", ".join(config.languages) if config.languages else "python"

    # Basic stats
    basic_stats = f"""[bold]Project:[/bold] {config.project_name}
[bold]Languages:[/bold] {langs_display}
[bold]Files indexed:[/bold] {db_stats['files']}
[bold]Code elements:[/bold] {db_stats['elements']}
[bold]Last indexed:[/bold] {db_stats['last_indexed'] or 'Never'}
[bold]Model:[/bold] {config.llm.model}
[bold]Embedding:[/bold] {config.llm.embedding_model}"""

    console.print(
        Panel(
            basic_stats,
            title="📊 CodeSage Statistics",
            border_style="blue",
        )
    )

    # Language breakdown (always shown if multiple languages)
    lang_stats = db.get_language_stats()
    if lang_stats and len(lang_stats) > 0:
        tree = Tree("📈 [bold]Language Breakdown[/bold]")

        # Sort by element count descending
        sorted_langs = sorted(
            lang_stats.items(),
            key=lambda x: x[1]["elements"],
            reverse=True
        )

        for lang, counts in sorted_langs:
            lang_name = lang.capitalize() if lang else "Unknown"
            tree.add(
                f"{lang_name}: [cyan]{counts['files']}[/cyan] files │ "
                f"[green]{counts['elements']}[/green] elements"
            )

        console.print(tree)
        console.print()

    if detailed:
        # Storage stats
        try:
            from codesage.llm.embeddings import EmbeddingService
            embedder = EmbeddingService(config.llm, config.cache_dir)
            storage = StorageManager(config, embedding_fn=embedder.embedder)
            storage_metrics = storage.get_metrics()

            # Vector store stats
            vector_stats = storage_metrics.get("vector", {})
            console.print(
                Panel(
                    f"""[bold]Backend:[/bold] {config.storage.vector_backend}
[bold]Vectors:[/bold] {vector_stats.get('count', 'N/A')}
[bold]Path:[/bold] {vector_stats.get('persist_dir', 'N/A')}""",
                    title="🔍 Vector Store",
                    border_style="cyan",
                )
            )

            # Graph stats
            if config.storage.use_graph:
                graph_stats = storage_metrics.get("graph", {})
                rel_counts = graph_stats.get("relationship_counts", {})

                rel_text = "\n".join([
                    f"  {k}: {v}" for k, v in rel_counts.items() if v > 0
                ]) or "  No relationships"

                console.print(
                    Panel(
                        f"""[bold]Backend:[/bold] KuzuDB
[bold]Nodes:[/bold] {graph_stats.get('node_count', 0)}
[bold]Relationships:[/bold]
{rel_text}""",
                        title="🕸️ Graph Store",
                        border_style="magenta",
                    )
                )
        except Exception as e:
            console.print(f"[dim]Could not load storage metrics: {e}[/dim]")

        # Memory stats
        if config.memory.enabled:
            try:
                from codesage.memory.memory_manager import MemoryManager
                memory = MemoryManager(global_dir=config.memory.global_dir)
                memory_metrics = memory.get_metrics()

                pref_stats = memory_metrics.get("preference_store", {})
                graph_stats = memory_metrics.get("memory_graph", {})

                console.print(
                    Panel(
                        f"""[bold]Patterns:[/bold] {pref_stats.get('pattern_count', 0)}
[bold]Projects:[/bold] {pref_stats.get('project_count', 0)}
[bold]Interactions:[/bold] {pref_stats.get('interaction_count', 0)}
[bold]Global Dir:[/bold] {memory_metrics.get('global_dir', 'N/A')}""",
                        title="🧠 Developer Memory",
                        border_style="green",
                    )
                )
            except Exception as e:
                console.print(f"[dim]Could not load memory metrics: {e}[/dim]")

