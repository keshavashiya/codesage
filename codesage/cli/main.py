"""CodeSage CLI - Main entry point."""

# Suppress urllib3 SSL warning on macOS with LibreSSL
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import typer

# Import commands
from codesage.cli.commands import (
    chat,
    index,
    init,
)

# Import command groups
from codesage.cli.groups import mcp

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
# Core Commands
# ============================================================================

app.command()(init)
app.command()(index)
app.command()(chat)

# ============================================================================
# Command Groups
# ============================================================================

app.add_typer(mcp.app, name="mcp")


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


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
