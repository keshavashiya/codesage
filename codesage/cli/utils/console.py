"""Console utilities for CodeSage CLI."""

from functools import lru_cache

from rich.console import Console


# Global flag for MCP stdio mode - when True, suppress all stdout output
_mcp_stdio_mode = False


def set_mcp_stdio_mode(enabled: bool) -> None:
    """Enable or disable MCP stdio mode.

    When enabled, this suppresses all console output to stdout
    to avoid corrupting the MCP JSON-RPC protocol.

    Args:
        enabled: True to enable stdio mode (suppress stdout)
    """
    global _mcp_stdio_mode
    _mcp_stdio_mode = enabled


def is_mcp_stdio_mode() -> bool:
    """Check if MCP stdio mode is enabled.

    Returns:
        True if stdout should be suppressed for MCP protocol
    """
    return _mcp_stdio_mode


@lru_cache(maxsize=1)
def get_console() -> Console:
    """Get singleton console instance for stdout.

    Returns:
        Shared Console instance for consistent output formatting.
    """
    return Console()


@lru_cache(maxsize=1)
def get_stderr_console() -> Console:
    """Get singleton console instance for stderr.

    Returns:
        Console instance that writes to stderr.
    """
    return Console(stderr=True)


def print_error(message: str) -> None:
    """Print error message with red cross icon.

    Always writes to stderr to avoid corrupting stdout-based protocols.

    Args:
        message: Error message to display.
    """
    get_stderr_console().print(f"[red]✗[/red] {message}")


def print_success(message: str) -> None:
    """Print success message with green checkmark.

    Args:
        message: Success message to display.
    """
    if _mcp_stdio_mode:
        # In MCP stdio mode, write to stderr to avoid corrupting JSON-RPC
        get_stderr_console().print(f"[green]✓[/green] {message}")
    else:
        get_console().print(f"[green]✓[/green] {message}")


def print_warning(message: str) -> None:
    """Print warning message with yellow warning icon.

    Always writes to stderr to avoid corrupting stdout-based protocols.

    Args:
        message: Warning message to display.
    """
    get_stderr_console().print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message with dim styling.

    Args:
        message: Info message to display.
    """
    if _mcp_stdio_mode:
        # In MCP stdio mode, write to stderr to avoid corrupting JSON-RPC
        get_stderr_console().print(f"[dim]{message}[/dim]")
    else:
        get_console().print(f"[dim]{message}[/dim]")
