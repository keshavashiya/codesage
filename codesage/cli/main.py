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
    review,
    stats,
    suggest,
    version,
)

# Import command groups
from codesage.cli.groups import hooks, mcp, profile, security, storage

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

# Register individual commands
app.command()(init)
app.command()(index)
app.command()(suggest)
app.command()(stats)
app.command()(health)
app.command()(review)
app.command()(chat)
app.command()(version)

# Register command groups
app.add_typer(security.app, name="security")
app.add_typer(hooks.app, name="hooks")
app.add_typer(profile.app, name="profile")
app.add_typer(storage.app, name="storage")
app.add_typer(mcp.app, name="mcp")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
