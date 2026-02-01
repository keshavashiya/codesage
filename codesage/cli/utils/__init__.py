"""CLI utility modules for CodeSage."""

from codesage.cli.utils.console import (
    get_console,
    get_stderr_console,
    is_mcp_stdio_mode,
    print_error,
    print_info,
    print_success,
    print_warning,
    set_mcp_stdio_mode,
)
from codesage.cli.utils.decorators import handle_errors, require_project
from codesage.cli.utils.options import PathArgument, PathOption

__all__ = [
    "get_console",
    "get_stderr_console",
    "is_mcp_stdio_mode",
    "print_error",
    "print_info",
    "print_success",
    "print_warning",
    "set_mcp_stdio_mode",
    "handle_errors",
    "require_project",
    "PathArgument",
    "PathOption",
]

