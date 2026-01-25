"""CodeSage CLI - Main entry point."""

# Suppress urllib3 SSL warning on macOS with LibreSSL
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import signal
import sys
import atexit
from typing import Callable, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from pathlib import Path

from codesage import __version__

# Input validation constants
MAX_QUERY_LENGTH = 2000
MAX_PATH_LENGTH = 4096

# Graceful shutdown infrastructure
_cleanup_handlers: List[Callable[[], None]] = []
_shutdown_in_progress = False


def register_cleanup(handler: Callable[[], None]) -> None:
    """Register a cleanup handler to be called on shutdown.

    Args:
        handler: Function to call during cleanup
    """
    if handler not in _cleanup_handlers:
        _cleanup_handlers.append(handler)


def unregister_cleanup(handler: Callable[[], None]) -> None:
    """Unregister a cleanup handler.

    Args:
        handler: Function to remove from cleanup list
    """
    if handler in _cleanup_handlers:
        _cleanup_handlers.remove(handler)


def _run_cleanup() -> None:
    """Run all registered cleanup handlers."""
    for handler in reversed(_cleanup_handlers):
        try:
            handler()
        except Exception as e:
            # Don't let cleanup errors prevent other cleanups
            console.print(f"[dim]Cleanup warning: {e}[/dim]")


def _shutdown_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully.

    Args:
        signum: Signal number received
        frame: Current stack frame
    """
    global _shutdown_in_progress

    if _shutdown_in_progress:
        # Force exit if already shutting down (user pressed Ctrl+C twice)
        console.print("\n[red]Force shutdown...[/red]")
        sys.exit(1)

    _shutdown_in_progress = True
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    console.print(f"\n[yellow]⏳ Received {signal_name}, shutting down gracefully...[/yellow]")

    _run_cleanup()

    console.print("[green]✓ Shutdown complete[/green]")
    sys.exit(0)


def _setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    # Handle SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Also register cleanup with atexit for normal exits
    atexit.register(_run_cleanup)


# Initialize signal handlers on module import
_setup_signal_handlers()

app = typer.Typer(
    name="codesage",
    help="Local-first code intelligence CLI with LangChain-powered RAG",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def init(
    path: str = typer.Argument(".", help="Project directory to initialize"),
    model: str = typer.Option(
        "qwen2.5-coder:7b",
        "--model", "-m",
        help="Ollama model to use for analysis"
    ),
    embedding_model: str = typer.Option(
        "mxbai-embed-large",
        "--embedding-model", "-e",
        help="Model to use for embeddings (mxbai-embed-large recommended for code)"
    ),
) -> None:
    """Initialize CodeSage in a project directory.

    Creates .codesage/ directory with configuration.
    """
    from codesage.utils.config import initialize_project

    console.print(f"\n[bold blue]🚀 Initializing CodeSage[/bold blue]\n")

    try:
        project_path = Path(path).resolve()
        config = initialize_project(project_path, model, embedding_model)

        console.print(f"[green]✓[/green] Created .codesage directory")
        console.print(f"[green]✓[/green] Configuration saved")

        console.print(Panel(
            f"""[bold]Project:[/bold] {config.project_name}
[bold]Model:[/bold] {config.llm.model}
[bold]Embedding:[/bold] {config.llm.embedding_model}
[bold]Language:[/bold] {config.language}""",
            title="📋 Configuration",
            border_style="blue",
        ))

        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Ensure Ollama is running: [cyan]ollama serve[/cyan]")
        console.print("  2. Pull required models:")
        console.print(f"     [cyan]ollama pull {model}[/cyan]")
        console.print(f"     [cyan]ollama pull {embedding_model}[/cyan]")
        console.print("  3. Index your codebase: [cyan]codesage index[/cyan]")
        console.print("  4. Search for code: [cyan]codesage suggest 'your query'[/cyan]\n")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def index(
    path: str = typer.Argument(".", help="Project directory to index"),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Only index changed files (default) or full reindex"
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear existing index before indexing"
    ),
) -> None:
    """Index the codebase for semantic search.

    Parses code files and generates embeddings.
    """
    from codesage.utils.config import Config
    from codesage.core.indexer import Indexer

    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        console.print("[red]✗[/red] Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]📂 Indexing {config.project_name}[/bold cyan]\n")

    if not incremental:
        console.print("[dim]Running full reindex...[/dim]")

    try:
        indexer = Indexer(config)

        # Register cleanup handler for graceful shutdown
        def _cleanup_indexer():
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

        console.print()
        console.print(Panel(
            f"""[bold]Files scanned:[/bold] {stats['files_scanned']}
[bold]Files indexed:[/bold] {stats['files_indexed']}
[bold]Files skipped:[/bold] {stats['files_skipped']} (unchanged)
[bold]Code elements:[/bold] {stats['elements_found']}
[bold]Errors:[/bold] {stats['errors']}""",
            title="📊 Indexing Complete",
            border_style="green",
        ))

        if stats['elements_found'] > 0:
            console.print("\n[green]✓[/green] Ready for suggestions!")
            console.print("  Try: [cyan]codesage suggest 'your query'[/cyan]\n")
        else:
            console.print("\n[yellow]⚠[/yellow] No code elements found.")
            console.print("  Check that you have Python files in your project.\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Indexing interrupted by user.")
        raise typer.Exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def suggest(
    query: str = typer.Argument(..., help="What are you looking for?"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of suggestions"),
    min_similarity: float = typer.Option(
        0.2,
        "--min-similarity", "-s",
        help="Minimum similarity threshold (0-1)"
    ),
    no_explain: bool = typer.Option(
        False,
        "--no-explain",
        help="Skip LLM explanations (faster)"
    ),
) -> None:
    """Get code suggestions based on natural language query.

    Uses semantic search to find relevant code.
    """
    from codesage.utils.config import Config
    from codesage.core.suggester import Suggester

    # Validate query
    query = query.strip()
    if not query:
        console.print("[red]✗[/red] Query cannot be empty")
        raise typer.Exit(1)
    if len(query) > MAX_QUERY_LENGTH:
        console.print(f"[red]✗[/red] Query too long (max {MAX_QUERY_LENGTH} chars)")
        raise typer.Exit(1)

    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        console.print("[red]✗[/red] Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    console.print(f"\n[dim]Searching for:[/dim] {query}\n")

    try:
        suggester = Suggester(config)
        suggestions = suggester.find_similar(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=not no_explain,
        )

        if not suggestions:
            console.print("[yellow]No suggestions found.[/yellow]")
            console.print("\n[dim]Tips:[/dim]")
            console.print("  • Try different search terms")
            console.print("  • Lower --min-similarity threshold")
            console.print("  • Run [cyan]codesage index[/cyan] to update the index\n")
            return

        for i, suggestion in enumerate(suggestions, 1):
            # Header
            console.print(
                f"[bold blue]{i}. {suggestion.file}:{suggestion.line}[/bold blue]"
            )
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

            # Explanation
            if suggestion.explanation:
                console.print(f"[italic]💡 {suggestion.explanation}[/italic]")

            console.print()

        console.print(f"[dim]Found {len(suggestions)} suggestion(s)[/dim]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Search interrupted by user.")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def stats(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """Show index statistics."""
    from codesage.utils.config import Config
    from codesage.storage.database import Database

    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        console.print("[red]✗[/red] Project not initialized.")
        raise typer.Exit(1)

    db = Database(config.storage.db_path)
    stats = db.get_stats()

    console.print(Panel(
        f"""[bold]Project:[/bold] {config.project_name}
[bold]Files indexed:[/bold] {stats['files']}
[bold]Code elements:[/bold] {stats['elements']}
[bold]Last indexed:[/bold] {stats['last_indexed'] or 'Never'}
[bold]Model:[/bold] {config.llm.model}
[bold]Embedding:[/bold] {config.llm.embedding_model}""",
        title="📊 CodeSage Statistics",
        border_style="blue",
    ))


@app.command()
def health(
    path: str = typer.Argument(".", help="Project directory"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Check system health and dependencies.

    Verifies Ollama, database, and vector store are working.
    """
    from codesage.utils.config import Config
    from codesage.utils.health import check_system_health
    import json

    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        if json_output:
            console.print(json.dumps({
                "healthy": False,
                "error": "Project not initialized"
            }))
        else:
            console.print("[red]✗[/red] Project not initialized.")
            console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    status = check_system_health(config)

    if json_output:
        console.print(json.dumps(status.to_dict(), indent=2))
    else:
        console.print("\n[bold]System Health Check[/bold]\n")

        # Ollama status
        if status.ollama_available:
            latency = f" ({status.ollama_latency_ms:.0f}ms)" if status.ollama_latency_ms else ""
            console.print(f"[green]✓[/green] Ollama{latency}")
        else:
            console.print("[red]✗[/red] Ollama")

        # Database status
        if status.database_accessible:
            size = f" ({status.database_size_mb:.1f}MB)" if status.database_size_mb else ""
            console.print(f"[green]✓[/green] Database{size}")
        else:
            console.print("[red]✗[/red] Database")

        # Vector store status
        if status.vector_store_accessible:
            count = f" ({status.vector_count} vectors)" if status.vector_count else ""
            console.print(f"[green]✓[/green] Vector Store{count}")
        else:
            console.print("[red]✗[/red] Vector Store")

        # Disk space
        if status.disk_space_ok:
            console.print("[green]✓[/green] Disk Space")
        else:
            console.print("[yellow]⚠[/yellow] Disk Space")

        # Errors and warnings
        if status.errors:
            console.print("\n[red]Errors:[/red]")
            for error in status.errors:
                console.print(f"  • {error}")

        if status.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in status.warnings:
                console.print(f"  • {warning}")

        # Summary
        console.print()
        if status.is_healthy:
            console.print("[green]✓ System is healthy[/green]")
        else:
            console.print("[red]✗ System has issues[/red]")
            raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show CodeSage version."""
    console.print(f"[bold]CodeSage[/bold] version {__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
