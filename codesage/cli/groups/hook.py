"""Hook command group for CodeSage CLI.

Manages git pre-commit hook installation.
"""

from pathlib import Path

import typer

from codesage.cli.utils.console import (
    get_console,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(
    name="hook",
    help="Manage git pre-commit hook integration.",
    no_args_is_help=True,
)


@app.command("install")
@handle_errors
def install(
    path: str = typer.Argument(".", help="Project directory"),
    severity: str = typer.Option(
        "high",
        "--severity",
        help="Block threshold: critical, high, warning, suggestion",
    ),
    mode: str = typer.Option(
        "fast",
        "--mode",
        "-m",
        help="Review mode: fast (static only) or full (+LLM)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Backup and replace an existing hook",
    ),
) -> None:
    """Install a pre-commit hook that runs 'codesage review --staged'.

    The hook blocks commits when issues at or above the severity threshold
    are found. Requires 'codesage' to be on PATH (installed via pipx).

    \b
    Examples:
      codesage hook install                       # block on high+
      codesage hook install --severity critical   # only block on critical
      codesage hook install --mode full           # include LLM review
      codesage hook install --force               # replace existing hook
    """
    from codesage.hooks.installer import HookInstaller

    console = get_console()
    project_path = Path(path).resolve()

    installer = HookInstaller(project_path)

    try:
        installer.install(severity=severity, mode=mode, force=force)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    status = installer.get_status()
    print_success(f"Pre-commit hook installed at {status.hook_path}")
    console.print(
        f"  Blocks on: [bold]{severity}[/bold]+ severity | "
        f"Mode: [bold]{mode}[/bold]"
    )
    console.print()
    console.print("  Every [bold]git commit[/bold] will now run:")
    console.print(
        f"    [cyan]codesage review --staged --severity {severity} --mode {mode}[/cyan]"
    )
    console.print()
    print_info("To bypass a commit: git commit --no-verify")
    print_info("To uninstall: codesage hook uninstall")


@app.command("uninstall")
@handle_errors
def uninstall(
    path: str = typer.Argument(".", help="Project directory"),
    keep_backup: bool = typer.Option(
        False,
        "--keep-backup",
        help="Do not restore the previously backed-up hook",
    ),
) -> None:
    """Remove the CodeSage pre-commit hook.

    If a previous hook was backed up during install, it will be restored
    unless --keep-backup is set.
    """
    from codesage.hooks.installer import HookInstaller

    project_path = Path(path).resolve()
    installer = HookInstaller(project_path)

    status = installer.get_status()

    if not status.is_codesage_hook:
        if status.installed:
            print_warning("Pre-commit hook exists but is not a CodeSage hook.")
            print_info("  Remove it manually or check its contents first.")
        else:
            print_info("No CodeSage pre-commit hook is installed.")
        raise typer.Exit(0)

    installer.uninstall(restore_backup=not keep_backup)

    if status.backup_path and not keep_backup:
        print_success("CodeSage hook removed. Previous hook restored.")
    else:
        print_success("CodeSage pre-commit hook removed.")


@app.command("status")
@handle_errors
def status(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """Show the current pre-commit hook status."""
    from codesage.hooks.installer import HookInstaller

    console = get_console()
    project_path = Path(path).resolve()

    try:
        installer = HookInstaller(project_path)
        st = installer.get_status()
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    console.print()
    if st.is_codesage_hook:
        print_success("CodeSage pre-commit hook is [green]installed[/green].")
        console.print(f"  Location: [dim]{st.hook_path}[/dim]")
        if st.backup_path:
            console.print(f"  Backup: [dim]{st.backup_path}[/dim]")
        console.print()

        # Show relevant lines from hook (severity/mode config)
        try:
            content = st.hook_path.read_text()
            for line in content.splitlines():
                if "Mode:" in line or "codesage review" in line:
                    console.print(f"  [dim]{line.strip()}[/dim]")
        except Exception:
            pass

    elif st.has_other_hook:
        print_warning("A pre-commit hook exists, but it was [bold]not[/bold] installed by CodeSage.")
        console.print(f"  Location: [dim]{st.hook_path}[/dim]")
        console.print()
        print_info("To install alongside it: codesage hook install --force")

    else:
        console.print("[dim]No pre-commit hook installed.[/dim]")
        console.print()
        print_info("To install: codesage hook install")

    console.print()
