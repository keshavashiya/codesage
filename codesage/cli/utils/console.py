"""Console utilities for CodeSage CLI."""

from functools import lru_cache

from rich.console import Console


@lru_cache(maxsize=1)
def get_console() -> Console:
    """Get singleton console instance.

    Returns:
        Shared Console instance for consistent output formatting.
    """
    return Console()


def print_error(message: str) -> None:
    """Print error message with red cross icon.

    Args:
        message: Error message to display.
    """
    get_console().print(f"[red]✗[/red] {message}")


def print_success(message: str) -> None:
    """Print success message with green checkmark.

    Args:
        message: Success message to display.
    """
    get_console().print(f"[green]✓[/green] {message}")


def print_warning(message: str) -> None:
    """Print warning message with yellow warning icon.

    Args:
        message: Warning message to display.
    """
    get_console().print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message with dim styling.

    Args:
        message: Info message to display.
    """
    get_console().print(f"[dim]{message}[/dim]")
