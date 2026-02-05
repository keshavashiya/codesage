"""Feature flags command group for CodeSage CLI."""

from pathlib import Path

import typer
from rich.table import Table

from codesage.cli.utils.console import get_console, print_error, print_success
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="Manage experimental feature flags")


@app.command("list")
@handle_errors
def list_flags(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """List all feature flags and their current values."""
    from codesage.utils.config import Config
    from codesage.utils.features import FeatureFlags

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    flags = FeatureFlags(config).list()

    table = Table(title="Feature Flags")
    table.add_column("Feature", style="cyan")
    table.add_column("Enabled", style="green")

    for name, enabled in sorted(flags.items()):
        table.add_row(name, "yes" if enabled else "no")

    console.print(table)


@app.command("enable")
@handle_errors
def enable_flag(
    feature: str = typer.Argument(..., help="Feature flag name to enable"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Enable a feature flag."""
    from codesage.utils.config import Config
    from codesage.utils.features import FeatureFlags

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    flags = FeatureFlags(config)
    if not hasattr(config.features, feature):
        print_error(f"Unknown feature: {feature}")
        raise typer.Exit(1)

    flags.enable(feature)
    print_success(f"Enabled: {feature}")
    console.print("[dim]Run 'codesage features list' to verify.[/dim]")


@app.command("disable")
@handle_errors
def disable_flag(
    feature: str = typer.Argument(..., help="Feature flag name to disable"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Disable a feature flag."""
    from codesage.utils.config import Config
    from codesage.utils.features import FeatureFlags

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    flags = FeatureFlags(config)
    if not hasattr(config.features, feature):
        print_error(f"Unknown feature: {feature}")
        raise typer.Exit(1)

    flags.disable(feature)
    print_success(f"Disabled: {feature}")
    console.print("[dim]Run 'codesage features list' to verify.[/dim]")


@app.command("reset")
@handle_errors
def reset_flags(
    path: str = typer.Argument(".", help="Project directory"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Reset all feature flags to defaults."""
    from codesage.utils.config import Config
    from codesage.utils.features import FeatureFlags

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    if not confirm:
        if not typer.confirm("Reset all feature flags to defaults?"):
            console.print("[yellow]Aborted[/yellow]")
            raise typer.Exit(0)

    FeatureFlags(config).reset()
    print_success("Feature flags reset to defaults")
