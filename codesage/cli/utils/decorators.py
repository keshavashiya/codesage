"""Command decorators for CodeSage CLI."""

import functools
from pathlib import Path
from typing import Any, Callable, TypeVar

import typer

from codesage.cli.utils.console import get_console, print_error

F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(func: F) -> F:
    """Decorator to handle command errors gracefully.

    Catches exceptions and keyboard interrupts, printing user-friendly
    error messages and exiting with appropriate codes.

    Args:
        func: Command function to wrap.

    Returns:
        Wrapped function with error handling.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            get_console().print("\n[yellow]⚠[/yellow] Operation cancelled by user.")
            raise typer.Exit(130)
        except typer.Exit:
            raise
        except Exception as e:
            print_error(f"Error: {e}")
            raise typer.Exit(1)

    return wrapper  # type: ignore


def require_project(func: F) -> F:
    """Decorator to ensure project is initialized before running command.

    Loads the project configuration and passes it to the command.
    If project is not initialized, prints an error and exits.

    Args:
        func: Command function to wrap.

    Returns:
        Wrapped function that validates project initialization.
    """
    from codesage.utils.config import Config

    @functools.wraps(func)
    def wrapper(*args: Any, path: str = ".", **kwargs: Any) -> Any:
        project_path = Path(path).resolve()
        try:
            config = Config.load(project_path)
        except FileNotFoundError:
            print_error("Project not initialized.")
            get_console().print("  Run: [cyan]codesage init[/cyan]")
            raise typer.Exit(1)
        return func(*args, path=path, config=config, **kwargs)

    return wrapper  # type: ignore
