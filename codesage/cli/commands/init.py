"""Init command for CodeSage CLI."""

from pathlib import Path

import typer
from rich.panel import Panel
from rich.tree import Tree

from codesage.cli.utils.console import get_console, print_error, print_success
from codesage.cli.utils.decorators import handle_errors


@handle_errors
def init(
    path: str = typer.Argument(".", help="Project directory to initialize"),
    model: str = typer.Option(
        "qwen2.5-coder:7b",
        "--model",
        "-m",
        help="Ollama model to use for analysis",
    ),
    embedding_model: str = typer.Option(
        "mxbai-embed-large",
        "--embedding-model",
        "-e",
        help="Model to use for embeddings (mxbai-embed-large recommended for code)",
    ),
    no_detect: bool = typer.Option(
        False,
        "--no-detect",
        help="Skip automatic language detection",
    ),
) -> None:
    """Initialize CodeSage in a project directory.

    Creates .codesage/ directory with configuration.
    Automatically detects languages used in the project.
    """
    from codesage.utils.config import initialize_project

    console = get_console()
    console.print("\n[bold blue]🚀 Initializing CodeSage[/bold blue]\n")

    project_path = Path(path).resolve()

    # Show language detection progress
    if not no_detect:
        console.print("[dim]Scanning project for languages...[/dim]")
        try:
            from codesage.utils.language_detector import detect_languages
            detected = detect_languages(project_path)

            if detected:
                tree = Tree("🔍 [bold]Detected Languages[/bold]")
                for lang_info in detected:
                    tree.add(f"{lang_info.name.capitalize()} ({lang_info.file_count} files)")
                console.print(tree)
                console.print()
        except Exception:
            pass  # Silent fallback

    config = initialize_project(
        project_path,
        model,
        embedding_model,
        auto_detect=not no_detect,
    )

    print_success("Created .codesage directory")
    print_success("Configuration saved")

    # Build languages display
    langs_display = ", ".join(config.languages) if config.languages else "python"

    console.print(
        Panel(
            f"""[bold]Project:[/bold] {config.project_name}
[bold]Model:[/bold] {config.llm.model}
[bold]Embedding:[/bold] {config.llm.embedding_model}
[bold]Languages:[/bold] {langs_display}""",
            title="📋 Configuration",
            border_style="blue",
        )
    )

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Ensure Ollama is running: [cyan]ollama serve[/cyan]")
    console.print("  2. Pull required models:")
    console.print(f"     [cyan]ollama pull {model}[/cyan]")
    console.print(f"     [cyan]ollama pull {embedding_model}[/cyan]")
    console.print("  3. Index your codebase: [cyan]codesage index[/cyan]")
    console.print("  4. Search for code: [cyan]codesage suggest 'your query'[/cyan]\n")

