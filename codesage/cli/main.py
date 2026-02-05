"""CodeSage CLI - Main entry point."""

# Suppress urllib3 SSL warning on macOS with LibreSSL
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import typer

# Import commands
from codesage.cli.commands import (
    chat,
    health,
    index,
    init,
    context,
    review,
    search,   # New primary search command
    stats,
    suggest,  # Deprecated - kept for backwards compatibility
    version,
)

# Import command groups
from codesage.cli.groups import (
    config,    # New unified config
    docs,
    features,  # Deprecated - use config features
    hooks,     # Deprecated - use config hooks
    mcp,
    profile,
    security,  # Deprecated - use review --security
    smells,    # Deprecated - use review --smells
    storage,   # Deprecated - use config storage
)

# Import signal handling
from codesage.cli.utils.signals import setup_signal_handlers

# Initialize signal handlers for graceful shutdown
setup_signal_handlers()

# Create main application
app = typer.Typer(
    name="codesage",
    help="Local-first code intelligence CLI with LangChain-powered RAG",
    add_completion=False,
    no_args_is_help=True,
)

# ============================================================================
# Core Commands (12 essential commands as per refactoring plan)
# ============================================================================

# Primary commands
app.command()(init)
app.command()(index)
app.command()(search)  # New: primary search
app.command()(chat)    # Primary: intelligent hub
app.command()(review)  # Consolidated review
app.command()(stats)
app.command()(version)

# ============================================================================
# Command Groups
# ============================================================================

# Primary groups
app.add_typer(mcp.app, name="mcp")
app.add_typer(profile.app, name="profile")
app.add_typer(config.app, name="config")  # New: unified config

# Keep docs under profile as per plan (or standalone)
app.add_typer(docs.app, name="docs")

# ============================================================================
# Deprecated Commands (with backwards compatibility)
# ============================================================================

# suggest -> search (deprecated alias)
@app.command(name="suggest", deprecated=True, hidden=True)
def suggest_deprecated(
    query: str = typer.Argument(...),
    path: str = typer.Option(".", "--path", "-p"),
    limit: int = typer.Option(5, "--limit", "-n"),
) -> None:
    """[Deprecated] Use 'codesage search' instead."""
    from codesage.cli.utils.console import get_console
    console = get_console()
    console.print("[yellow]⚠ 'suggest' is deprecated. Use 'search' instead.[/yellow]")
    # Call the actual suggest function
    suggest(query=query, path=path, limit=limit)


# context -> chat (deprecated)
@app.command(name="context", deprecated=True, hidden=True)
def context_deprecated(
    task: str = typer.Argument(...),
    path: str = typer.Option(".", "--path", "-p"),
) -> None:
    """[Deprecated] Use 'codesage chat' with /plan command instead."""
    from codesage.cli.utils.console import get_console
    console = get_console()
    console.print("[yellow]⚠ 'context' is deprecated. Use 'chat' then '/plan <task>'.[/yellow]")
    context(task=task, path=path)


# health -> stats --health (deprecated)
app.command(name="health", deprecated=True, hidden=True)(health)


# ============================================================================
# Deprecated Command Groups (kept for backwards compatibility)
# ============================================================================

# These groups are now consolidated into 'config' or other commands
# Show deprecation message when used

# Version callback for --version flag
def _version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from codesage import __version__
        from codesage.cli.utils.console import get_console
        get_console().print(f"[bold]CodeSage[/bold] version {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback,
        is_eager=True, help="Show version and exit"
    ),
) -> None:
    """Local-first code intelligence CLI."""
    pass


# Register deprecated groups (keeping old behavior)
app.add_typer(security.app, name="security", deprecated=True, hidden=True)
app.add_typer(smells.app, name="smells", deprecated=True, hidden=True)
app.add_typer(hooks.app, name="hooks", deprecated=True, hidden=True)
app.add_typer(features.app, name="features", deprecated=True, hidden=True)
app.add_typer(storage.app, name="storage", deprecated=True, hidden=True)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
