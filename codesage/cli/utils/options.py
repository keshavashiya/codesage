"""Shared CLI options and arguments for CodeSage commands."""

import typer

# Common path argument used across multiple commands
PathArgument = typer.Argument(
    ".",
    help="Project directory path",
)

# Path as an option (for commands where path is optional)
PathOption = typer.Option(
    ".",
    "--path",
    "-p",
    help="Project directory path",
)

# Common limit option for search/suggest commands
LimitOption = typer.Option(
    5,
    "--limit",
    "-n",
    help="Maximum number of results to return",
)

# Verbose output option
VerboseOption = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Show detailed output",
)

# JSON output option
JsonOption = typer.Option(
    False,
    "--json",
    "-j",
    help="Output results as JSON",
)

# Input validation constants
MAX_QUERY_LENGTH = 2000
MAX_PATH_LENGTH = 4096
