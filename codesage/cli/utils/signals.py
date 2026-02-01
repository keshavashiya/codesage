"""Signal handling infrastructure for graceful shutdown."""

import atexit
import signal
import sys
from typing import Callable, List

from codesage.cli.utils.console import get_console

# Cleanup handler registry
_cleanup_handlers: List[Callable[[], None]] = []
_shutdown_in_progress = False


def register_cleanup(handler: Callable[[], None]) -> None:
    """Register a cleanup handler to be called on shutdown.

    Args:
        handler: Function to call during cleanup.
    """
    if handler not in _cleanup_handlers:
        _cleanup_handlers.append(handler)


def unregister_cleanup(handler: Callable[[], None]) -> None:
    """Unregister a cleanup handler.

    Args:
        handler: Function to remove from cleanup list.
    """
    if handler in _cleanup_handlers:
        _cleanup_handlers.remove(handler)


def _run_cleanup() -> None:
    """Run all registered cleanup handlers in reverse order."""
    console = get_console()
    for handler in reversed(_cleanup_handlers):
        try:
            handler()
        except Exception as e:
            console.print(f"[dim]Cleanup warning: {e}[/dim]")


def _shutdown_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully.

    Args:
        signum: Signal number received.
        frame: Current stack frame.
    """
    global _shutdown_in_progress
    console = get_console()

    if _shutdown_in_progress:
        # Force exit if already shutting down (user pressed Ctrl+C twice)
        console.print("\n[red]Force shutdown...[/red]")
        sys.exit(1)

    _shutdown_in_progress = True
    signal_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    console.print(f"\n[yellow]⏳ Received {signal_name}, shutting down gracefully...[/yellow]")

    _run_cleanup()

    console.print("[green]✓ Shutdown complete[/green]")
    sys.exit(0)


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    atexit.register(_run_cleanup)
