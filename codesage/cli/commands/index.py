"""Index command for CodeSage CLI."""

from pathlib import Path

import typer
from rich.panel import Panel

from codesage.cli.utils.console import get_console, print_error, print_success, print_warning
from codesage.cli.utils.decorators import handle_errors
from codesage.cli.utils.signals import register_cleanup, unregister_cleanup


def _print_parser_availability_hint(console, project_path: Path, files_scanned: int) -> None:
    """Print helpful hints about parser availability for scanned files.

    Detects if JS/TS or other non-Python files were scanned but couldn't be indexed
    due to missing tree-sitter parsers.
    """
    from codesage.parsers import ParserRegistry
    from codesage.utils.language_detector import detect_languages

    # Get registered parsers
    registered_langs = set(ParserRegistry.supported_languages())

    # Detect languages in the project
    try:
        detected = detect_languages(project_path)
        detected_langs = {lang.name for lang in detected}
    except Exception:
        detected_langs = set()

    # Check for languages detected but not supported
    missing_langs = detected_langs - registered_langs

    if missing_langs:
        # Non-Python languages detected but parsers not available
        lang_list = ", ".join(sorted(missing_langs))
        console.print(f"  Detected [yellow]{lang_list}[/yellow] files but parsers are not installed.")
        console.print(
            "  Install multi-language support with:\n"
            "    [cyan]pipx inject pycodesage 'pycodesage[multi-language]'[/cyan]\n"
            "  Or if using pip:\n"
            "    [cyan]pip install 'pycodesage[multi-language]'[/cyan]\n"
        )
    elif files_scanned > 0 and "python" not in detected_langs:
        # Files were scanned but no recognized language
        console.print("  No supported source files found in this project.\n")
        console.print("  Supported languages: Python, JavaScript, TypeScript, Go, Rust\n")
    else:
        console.print("  Check that you have source files in your project.\n")


@handle_errors
def index(
    path: str = typer.Argument(".", help="Project directory to index"),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Only index changed files (default) or full reindex",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear existing index before indexing",
    ),
    learn: bool = typer.Option(
        None,
        "--learn/--no-learn",
        help="Enable/disable pattern learning (overrides config)",
    ),
) -> None:
    """Index the codebase for semantic search.

    Parses code files and generates embeddings.
    """
    from codesage.core.indexer import Indexer
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]📂 Indexing {config.project_name}[/bold cyan]\n")

    if not incremental:
        console.print("[dim]Running full reindex...[/dim]")

    indexer = Indexer(config)

    # Override memory learning if explicitly set
    if learn is not None:
        indexer.set_memory_learning(learn)
        if learn:
            console.print("[dim]Memory learning enabled[/dim]")
        else:
            console.print("[dim]Memory learning disabled[/dim]")

    # Register cleanup handler for graceful shutdown
    def _cleanup_indexer() -> None:
        try:
            indexer.db.close()
        except Exception:
            pass

    register_cleanup(_cleanup_indexer)

    if clear:
        console.print("[yellow]Clearing existing index...[/yellow]")
        indexer.clear_index()

    stats = indexer.index_repository(incremental=incremental)

    # Unregister cleanup since we completed successfully
    unregister_cleanup(_cleanup_indexer)

    # Build stats display
    stats_text = f"""[bold]Files scanned:[/bold] {stats['files_scanned']}
[bold]Files indexed:[/bold] {stats['files_indexed']}
[bold]Files skipped:[/bold] {stats['files_skipped']} (unchanged)
[bold]Code elements:[/bold] {stats['elements_found']}
[bold]Graph nodes:[/bold] {stats['nodes_added']}
[bold]Relationships:[/bold] {stats['relationships_added']}
[bold]Errors:[/bold] {stats['errors']}"""

    # Add memory stats if available
    memory_stats = indexer.get_memory_stats()
    if memory_stats:
        stats_text += f"""
[bold]Patterns learned:[/bold] {memory_stats.get('patterns_learned', 0)}"""

    console.print()
    console.print(
        Panel(
            stats_text,
            title="📊 Indexing Complete",
            border_style="green",
        )
    )

    if stats["elements_found"] > 0:
        print_success("Ready for suggestions!")
        console.print("  Try: [cyan]codesage suggest 'your query'[/cyan]\n")
    else:
        print_warning("No code elements found.")
        # Check if non-Python files were scanned but no parser is available
        _print_parser_availability_hint(console, project_path, stats["files_scanned"])
