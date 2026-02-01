"""Version command for CodeSage CLI."""

from codesage import __version__
from codesage.cli.utils.console import get_console


def version() -> None:
    """Show CodeSage version."""
    get_console().print(f"[bold]CodeSage[/bold] version {__version__}")
