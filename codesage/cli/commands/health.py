"""Health command for CodeSage CLI."""

import json
from pathlib import Path

import typer

from codesage.cli.utils.console import get_console, print_error
from codesage.cli.utils.decorators import handle_errors


@handle_errors
def health(
    path: str = typer.Argument(".", help="Project directory"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Check system health and dependencies.

    Verifies Ollama, database, and vector store are working.
    """
    from codesage.utils.config import Config
    from codesage.utils.health import check_system_health

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        if json_output:
            console.print(json.dumps({"healthy": False, "error": "Project not initialized"}))
        else:
            print_error("Project not initialized.")
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
