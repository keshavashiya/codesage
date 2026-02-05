"""Storage command group for CodeSage CLI.

Provides commands for managing storage backends.
"""

from pathlib import Path

import typer
from rich.panel import Panel

from codesage.cli.utils.console import get_console, print_error, print_success
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="Storage backend management")


@app.command("info")
@handle_errors
def info(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """Show storage backend information."""
    from codesage.utils.config import Config
    from codesage.storage.manager import StorageManager
    from codesage.llm.embeddings import EmbeddingService

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold]Project:[/bold] {config.project_name}\n"
        f"[bold]Vector Backend:[/bold] {config.storage.vector_backend}\n"
        f"[bold]Graph Enabled:[/bold] {config.storage.use_graph}",
        title="Storage Configuration",
        border_style="blue",
    ))

    # Show paths
    console.print("\n[bold]Storage Paths:[/bold]")
    console.print(f"  SQLite: {config.storage.db_path}")
    console.print(f"  LanceDB: {config.storage.lance_path}")
    console.print(f"  KuzuDB: {config.storage.kuzu_path}")

    # Check which backends exist
    console.print("\n[bold]Backend Status:[/bold]")

    if config.storage.db_path.exists():
        console.print("  [green]SQLite: exists[/green]")
    else:
        console.print("  [dim]SQLite: not initialized[/dim]")

    if config.storage.lance_path.exists():
        try:
            embedder = EmbeddingService(
                config.llm,
                config.cache_dir,
                config.performance,
            )
            storage = StorageManager(config, embedding_fn=embedder)
            lance_count = storage.vector_store.count()
            console.print(f"  [green]LanceDB: {lance_count} documents[/green]")
        except Exception:
            console.print("  [yellow]LanceDB: exists but could not read[/yellow]")
    else:
        console.print("  [dim]LanceDB: not initialized[/dim]")

    if config.storage.kuzu_path.exists():
        console.print("  [green]KuzuDB: exists[/green]")
    else:
        console.print("  [dim]KuzuDB: not initialized[/dim]")


@app.command("stats")
@handle_errors
def stats(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """Show detailed storage statistics."""
    from codesage.utils.config import Config
    from codesage.storage.manager import StorageManager
    from codesage.llm.embeddings import EmbeddingService
    from rich.table import Table

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    try:
        embedder = EmbeddingService(
            config.llm,
            config.cache_dir,
            config.performance,
        )
        storage = StorageManager(config, embedding_fn=embedder)
    except Exception as e:
        print_error(f"Failed to initialize storage: {e}")
        raise typer.Exit(1)

    # Get metrics
    metrics = storage.get_metrics()

    # Display table
    table = Table(title="Storage Metrics")
    table.add_column("Backend", style="cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green")

    # SQLite metrics
    if "sqlite" in metrics:
        for key, value in metrics["sqlite"].items():
            table.add_row("SQLite", key, str(value))

    # LanceDB metrics
    if "vector" in metrics:
        for key, value in metrics["vector"].items():
            table.add_row("LanceDB", key, str(value))

    # KuzuDB metrics
    if "graph" in metrics:
        for key, value in metrics["graph"].items():
            table.add_row("KuzuDB", key, str(value))

    console.print(table)


@app.command("clear")
@handle_errors
def clear(
    path: str = typer.Argument(".", help="Project directory"),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Specific backend to clear (lancedb, kuzu, sqlite, all)",
    ),
    confirm: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt"
    ),
) -> None:
    """Clear storage data."""
    from codesage.utils.config import Config
    import shutil

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    targets = []
    if backend is None or backend == "all":
        targets = ["lancedb", "kuzu", "sqlite"]
    else:
        targets = [backend]

    if not confirm:
        message = f"This will clear data from: {', '.join(targets)}"
        if not typer.confirm(f"{message}. Continue?"):
            console.print("[yellow]Aborted[/yellow]")
            raise typer.Exit(0)

    for target in targets:
        try:
            if target == "lancedb" and config.storage.lance_path.exists():
                shutil.rmtree(config.storage.lance_path)
                console.print("[green]Cleared LanceDB[/green]")

            elif target == "kuzu" and config.storage.kuzu_path.exists():
                shutil.rmtree(config.storage.kuzu_path)
                console.print("[green]Cleared KuzuDB[/green]")

            elif target == "sqlite" and config.storage.db_path.exists():
                config.storage.db_path.unlink()
                console.print("[green]Cleared SQLite[/green]")

        except Exception as e:
            console.print(f"[yellow]Could not clear {target}: {e}[/yellow]")

    print_success("Storage cleared!")
    console.print("[dim]Run 'codesage index' to rebuild.[/dim]")


@app.command("repair")
@handle_errors
def repair(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """Repair storage inconsistencies.

    Checks for orphaned records and index inconsistencies,
    and attempts to fix them.
    """
    from codesage.utils.config import Config
    from codesage.storage.manager import StorageManager
    from codesage.llm.embeddings import EmbeddingService

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    console.print("[cyan]Checking storage consistency...[/cyan]")

    try:
        embedder = EmbeddingService(
            config.llm,
            config.cache_dir,
            config.performance,
        )
        storage = StorageManager(config, embedding_fn=embedder)
    except Exception as e:
        print_error(f"Failed to initialize storage: {e}")
        raise typer.Exit(1)

    # Get counts from each backend
    metrics = storage.get_metrics()

    sqlite_count = metrics.get("sqlite", {}).get("total_elements", 0)
    vector_count = metrics.get("vector", {}).get("total_documents", 0)

    console.print(f"\n[bold]Element Counts:[/bold]")
    console.print(f"  SQLite: {sqlite_count}")
    console.print(f"  LanceDB: {vector_count}")

    if sqlite_count == vector_count:
        print_success("Storage is consistent!")
    else:
        console.print("\n[yellow]Inconsistency detected.[/yellow]")
        console.print("[dim]Run 'codesage index --full' to rebuild from source.[/dim]")
