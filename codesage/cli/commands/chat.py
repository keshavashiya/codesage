"""Chat command for CodeSage CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status

from codesage.cli.utils.console import get_console, print_error, print_success
from codesage.cli.utils.decorators import handle_errors

if TYPE_CHECKING:
    from codesage.chat import ChatEngine


@handle_errors
def chat(
    path: str = typer.Argument(".", help="Project directory"),
    no_context: bool = typer.Option(
        False,
        "--no-context",
        help="Disable automatic code context retrieval",
    ),
    max_context: int = typer.Option(
        3,
        "--max-context",
        "-c",
        help="Maximum code snippets to include in context",
    ),
) -> None:
    """Start an interactive chat session about your codebase.

    Ask questions in natural language and get answers with
    relevant code context.

    Commands available in chat:
      /help    - Show available commands
      /search  - Search codebase
      /clear   - Clear conversation
      /exit    - Exit chat
    """
    from codesage.chat import ChatEngine
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    # Load configuration
    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        console.print("  Run: [cyan]codesage init[/cyan]")
        raise typer.Exit(1)

    # Check if index exists
    if not config.storage.db_path.exists():
        print_error("Project not indexed.")
        console.print("  Run: [cyan]codesage index[/cyan]")
        raise typer.Exit(1)

    # Initialize chat engine
    engine = ChatEngine(
        config=config,
        include_context=not no_context,
        max_context_results=max_context,
    )

    # Welcome message
    console.print()
    console.print(
        Panel(
            f"[bold cyan]CodeSage Chat[/bold cyan]\n\n"
            f"Project: [green]{config.project_name}[/green]\n"
            f"Context: {'[green]enabled[/green]' if not no_context else '[yellow]disabled[/yellow]'}\n\n"
            f"[dim]Type your questions or use /help for commands.[/dim]\n"
            f"[dim]Press Ctrl+D or type /exit to quit.[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Main chat loop
    try:
        _run_chat_loop(console, engine)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Chat ended.[/dim]")
    finally:
        console.print()


def _run_chat_loop(console: Console, engine: "ChatEngine") -> None:
    """Run the main chat loop.

    Args:
        console: Rich console instance
        engine: Chat engine instance
    """
    while True:
        # Get user input
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue

        # Process input with spinner for LLM calls
        is_command = user_input.strip().startswith("/")

        if is_command:
            # Commands are fast, no spinner needed
            response, should_continue = engine.process_input(user_input)
        else:
            # LLM calls get a spinner
            with Status("[cyan]Thinking...", spinner="dots", console=console):
                response, should_continue = engine.process_input(user_input)

        # Check if we should exit
        if not should_continue:
            console.print(f"\n[dim]{response}[/dim]")
            break

        # Display response
        _display_response(console, response, is_command)


def _display_response(console: Console, response: str, is_command: bool) -> None:
    """Display the assistant response.

    Args:
        console: Rich console instance
        response: Response text
        is_command: Whether this was a command response
    """
    console.print()

    if is_command:
        # Commands get simple markdown rendering
        console.print(Markdown(response))
    else:
        # Chat responses get a styled panel
        console.print(
            Panel(
                Markdown(response),
                title="[bold green]CodeSage[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    console.print()
