"""Hooks command group for CodeSage CLI."""

from pathlib import Path

import typer

from codesage.cli.utils.console import get_console, print_error, print_success
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="Git hooks management")


@app.command("install")
@handle_errors
def install(
    path: str = typer.Argument(".", help="Git repository path"),
    severity: str = typer.Option(
        "medium",
        "--severity",
        "-s",
        help="Minimum severity to block commits (low, medium, high, critical)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Replace existing hooks (backup will be created)",
    ),
) -> None:
    """Install CodeSage pre-commit hook.

    The hook will run security scanning on staged files before each commit.
    """
    from codesage.hooks.installer import HookInstaller

    console = get_console()

    try:
        installer = HookInstaller(Path(path).resolve())
        installer.install(severity=severity, force=force)

        print_success("Pre-commit hook installed")
        console.print(f"[dim]Blocking on severity: {severity} and above[/dim]")
        console.print("\n[dim]To bypass: git commit --no-verify[/dim]")

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)


@app.command("uninstall")
@handle_errors
def uninstall(
    path: str = typer.Argument(".", help="Git repository path"),
    no_restore: bool = typer.Option(
        False,
        "--no-restore",
        help="Don't restore backed-up hook",
    ),
) -> None:
    """Uninstall CodeSage pre-commit hook."""
    from codesage.hooks.installer import HookInstaller

    console = get_console()

    try:
        installer = HookInstaller(Path(path).resolve())
        installer.uninstall(restore_backup=not no_restore)
        print_success("Pre-commit hook removed")

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)


@app.command("status")
@handle_errors
def status(
    path: str = typer.Argument(".", help="Git repository path"),
) -> None:
    """Show pre-commit hook status."""
    from codesage.hooks.installer import HookInstaller

    console = get_console()

    try:
        installer = HookInstaller(Path(path).resolve())
        hook_status = installer.get_status()

        if hook_status.is_codesage_hook:
            print_success("CodeSage hook installed")
            console.print(f"[dim]Location: {hook_status.hook_path}[/dim]")
        elif hook_status.has_other_hook:
            console.print("[yellow]![/yellow] Non-CodeSage hook present")
            console.print(f"[dim]Location: {hook_status.hook_path}[/dim]")
            console.print("[dim]Use --force to replace[/dim]")
        else:
            console.print("[dim]No pre-commit hook installed[/dim]")

        if hook_status.backup_path:
            console.print(f"[dim]Backup: {hook_status.backup_path}[/dim]")

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)
