"""Unified config command group for CodeSage CLI.

Consolidates:
- features (enable/disable)
- storage settings
- hooks management
- security rules

This follows the refactoring plan to reduce CLI complexity.
"""

import typer
from pathlib import Path
from typing import Optional

from codesage.cli.utils.console import get_console, print_error, print_success, print_warning
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="Configuration management")


# ============================================================================
# Features Subcommands
# ============================================================================

features_app = typer.Typer(help="Manage feature flags")
app.add_typer(features_app, name="features")


@features_app.command("list")
@handle_errors
def features_list(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """List all feature flags and their status."""
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized. Run 'codesage init' first.")
        raise typer.Exit(1)

    console.print("\n[bold cyan]Feature Flags[/bold cyan]\n")

    features = config.features
    feature_list = [
        ("embeddings", features.embeddings, "Local embeddings for semantic search"),
        ("memory", features.memory, "Developer memory and pattern learning"),
        ("llm_explanations", features.llm_explanations, "LLM-powered explanations"),
        ("graph_storage", features.graph_storage, "Graph-based code relationships"),
        ("context_provider_mode", features.context_provider_mode, "Structured context for AI agents"),
        ("cross_project_recommendations", features.cross_project_recommendations, "Cross-project pattern insights"),
    ]

    for name, enabled, description in feature_list:
        status = "[green]✓ enabled[/green]" if enabled else "[dim]✗ disabled[/dim]"
        console.print(f"  {status}  [bold]{name}[/bold]")
        console.print(f"           [dim]{description}[/dim]")

    console.print("\n[dim]Enable with: codesage config features enable <name>[/dim]")
    console.print("[dim]Disable with: codesage config features disable <name>[/dim]\n")


@features_app.command("enable")
@handle_errors
def features_enable(
    feature: str = typer.Argument(..., help="Feature name to enable"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Enable a feature flag."""
    from codesage.utils.config import Config

    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized. Run 'codesage init' first.")
        raise typer.Exit(1)

    if not hasattr(config.features, feature):
        print_error(f"Unknown feature: {feature}")
        raise typer.Exit(1)

    setattr(config.features, feature, True)
    config.save()
    print_success(f"Feature '{feature}' enabled")


@features_app.command("disable")
@handle_errors
def features_disable(
    feature: str = typer.Argument(..., help="Feature name to disable"),
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Disable a feature flag."""
    from codesage.utils.config import Config

    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized. Run 'codesage init' first.")
        raise typer.Exit(1)

    if not hasattr(config.features, feature):
        print_error(f"Unknown feature: {feature}")
        raise typer.Exit(1)

    setattr(config.features, feature, False)
    config.save()
    print_success(f"Feature '{feature}' disabled")


# ============================================================================
# Storage Subcommands
# ============================================================================

storage_app = typer.Typer(help="Storage management")
app.add_typer(storage_app, name="storage")


@storage_app.command("info")
@handle_errors
def storage_info(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Show storage information and sizes."""
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized. Run 'codesage init' first.")
        raise typer.Exit(1)

    console.print("\n[bold cyan]Storage Information[/bold cyan]\n")

    # Database
    db_path = config.storage.db_path
    if db_path.exists():
        db_size = db_path.stat().st_size / 1024 / 1024
        console.print(f"  [bold]Database:[/bold] {db_path}")
        console.print(f"  [bold]Size:[/bold] {db_size:.2f} MB")
    else:
        console.print("  [dim]Database not found. Run 'codesage index'.[/dim]")

    # Graph store (KuzuDB)
    graph_path = config.storage.kuzu_path
    if graph_path and graph_path.exists():
        # Calculate directory size
        total = sum(f.stat().st_size for f in graph_path.rglob('*') if f.is_file())
        graph_size = total / 1024 / 1024
        console.print(f"\n  [bold]Graph Store:[/bold] {graph_path}")
        console.print(f"  [bold]Size:[/bold] {graph_size:.2f} MB")
    else:
        console.print("\n  [dim]Graph store not found.[/dim]")

    # Embeddings (LanceDB)
    embeddings_path = config.storage.lance_path
    if embeddings_path and embeddings_path.exists():
        total = sum(f.stat().st_size for f in embeddings_path.rglob('*') if f.is_file())
        emb_size = total / 1024 / 1024
        console.print(f"\n  [bold]Embeddings:[/bold] {embeddings_path}")
        console.print(f"  [bold]Size:[/bold] {emb_size:.2f} MB")

    console.print()


@storage_app.command("clear")
@handle_errors
def storage_clear(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Clear all storage data (requires re-indexing)."""
    import shutil
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm("This will delete all indexed data. Continue?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Clear database
    if config.storage.db_path and config.storage.db_path.exists():
        config.storage.db_path.unlink()
        console.print("  ✓ Database cleared")

    # Clear graph store (KuzuDB)
    if config.storage.kuzu_path and config.storage.kuzu_path.exists():
        shutil.rmtree(config.storage.kuzu_path)
        console.print("  ✓ Graph store cleared")

    # Clear embeddings (LanceDB)
    if config.storage.lance_path and config.storage.lance_path.exists():
        shutil.rmtree(config.storage.lance_path)
        console.print("  ✓ Embeddings cleared")

    print_success("Storage cleared. Run 'codesage index' to rebuild.")


@storage_app.command("repair")
@handle_errors
def storage_repair(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Repair storage inconsistencies."""
    from codesage.utils.config import Config
    from codesage.storage.database import Database

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    console.print("Checking storage...\n")

    db = Database(config.storage.db_path)
    stats = db.get_stats()

    console.print(f"  Files: {stats.get('files', 0)}")
    console.print(f"  Elements: {stats.get('elements', 0)}")

    # Run vacuum
    try:
        db._conn.execute("VACUUM")
        print_success("Database optimized")
    except Exception as e:
        print_warning(f"Could not optimize: {e}")


# ============================================================================
# Hooks Subcommands
# ============================================================================

hooks_app = typer.Typer(help="Git hooks management")
app.add_typer(hooks_app, name="hooks")


@hooks_app.command("install")
@handle_errors
def hooks_install(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
    hook: str = typer.Option("pre-commit", "--hook", "-H", help="Hook type"),
) -> None:
    """Install git hooks for automatic review."""
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized. Run 'codesage init' first.")
        raise typer.Exit(1)

    git_hooks_dir = project_path / ".git" / "hooks"
    if not git_hooks_dir.exists():
        print_error("Not a git repository.")
        raise typer.Exit(1)

    hook_path = git_hooks_dir / hook
    hook_content = f"""#!/bin/sh
# CodeSage {hook} hook
codesage review --staged --quiet
"""

    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)
    print_success(f"Installed {hook} hook")


@hooks_app.command("uninstall")
@handle_errors
def hooks_uninstall(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
    hook: str = typer.Option("pre-commit", "--hook", "-H", help="Hook type"),
) -> None:
    """Uninstall git hooks."""
    project_path = Path(path).resolve()

    hook_path = project_path / ".git" / "hooks" / hook
    if hook_path.exists():
        hook_path.unlink()
        print_success(f"Uninstalled {hook} hook")
    else:
        print_warning("Hook not found")


@hooks_app.command("status")
@handle_errors
def hooks_status(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Show installed hooks status."""
    console = get_console()
    project_path = Path(path).resolve()

    git_hooks_dir = project_path / ".git" / "hooks"
    if not git_hooks_dir.exists():
        print_warning("Not a git repository")
        return

    console.print("\n[bold cyan]Git Hooks[/bold cyan]\n")

    hook_types = ["pre-commit", "pre-push", "commit-msg"]
    for hook in hook_types:
        hook_path = git_hooks_dir / hook
        if hook_path.exists():
            # Check if it's a CodeSage hook
            content = hook_path.read_text()
            if "codesage" in content.lower():
                console.print(f"  [green]✓[/green] {hook} [dim](CodeSage)[/dim]")
            else:
                console.print(f"  [yellow]●[/yellow] {hook} [dim](other)[/dim]")
        else:
            console.print(f"  [dim]✗[/dim] {hook}")

    console.print()


# ============================================================================
# Security Rules Subcommands (moved from security group)
# ============================================================================

security_app = typer.Typer(help="Security rules configuration")
app.add_typer(security_app, name="security")


@security_app.command("rules")
@handle_errors
def security_rules(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Show active security rules."""
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    console.print("\n[bold cyan]Security Rules[/bold cyan]\n")

    if hasattr(config, 'security') and hasattr(config.security, 'enabled_rules'):
        for rule in config.security.enabled_rules:
            console.print(f"  [green]✓[/green] {rule}")
    else:
        console.print("  Using default security rules")

    console.print("\n[dim]Security scan: codesage review --security[/dim]\n")


# ============================================================================
# Show all config
# ============================================================================

@app.command("show")
@handle_errors
def show_config(
    path: str = typer.Option(".", "--path", "-p", help="Project directory"),
) -> None:
    """Show all configuration settings."""
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized. Run 'codesage init' first.")
        raise typer.Exit(1)

    console.print("\n[bold cyan]CodeSage Configuration[/bold cyan]\n")

    console.print(f"[bold]Project:[/bold] {config.project_name}")
    console.print(f"[bold]Path:[/bold] {config.project_path}")
    console.print(f"[bold]Language:[/bold] {config.language}")

    console.print("\n[bold]Features:[/bold]")
    console.print(f"  embeddings: {config.features.embeddings}")
    console.print(f"  memory: {config.features.memory}")
    console.print(f"  llm_explanations: {config.features.llm_explanations}")
    console.print(f"  graph_storage: {config.features.graph_storage}")

    console.print("\n[bold]LLM:[/bold]")
    console.print(f"  provider: {config.llm.provider}")
    console.print(f"  model: {config.llm.model}")

    console.print()
