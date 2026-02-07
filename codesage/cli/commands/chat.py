"""Chat command for CodeSage CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from codesage.cli.utils.console import get_console, print_error
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
    """Run the main chat loop with streaming support."""
    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue

        # All input goes through streaming path.
        # process_input_stream() handles both:
        # - LLM commands (/deep, /review, etc.) → streamed token-by-token
        # - Non-LLM commands (/help, /stats, etc.) → single "done" yield
        # - Regular messages → streamed token-by-token
        should_exit = _stream_response(console, engine, user_input)
        if should_exit:
            break


def _stream_response(console: Console, engine: "ChatEngine", user_input: str) -> bool:
    """Stream a chat response with a spinner and formatted final output.

    rich.live.Live cannot handle content taller than the terminal — it
    uses ANSI cursor-up to overwrite, which fails once content scrolls
    off-screen, causing duplicated / frozen panels.

    Instead we:
      1. Print status and context directly (static output).
      2. Show a spinner with a live word-count while tokens arrive.
      3. Print the full Markdown Panel once streaming completes.

    Returns:
        True if the user requested exit, False otherwise.
    """
    console.print()

    collected_tokens: list[str] = []
    exiting = False

    with console.status(
        "[bold green]CodeSage[/bold green] [dim]is thinking…[/dim]",
        spinner="dots",
    ) as status:
        for chunk_type, content in engine.process_input_stream(user_input):
            if chunk_type == "exit":
                exiting = True
                break

            elif chunk_type == "status":
                status.update(
                    f"[bold green]CodeSage[/bold green] [dim]{content}[/dim]"
                )

            elif chunk_type == "context":
                # Temporarily stop spinner so the panel prints cleanly.
                status.stop()
                console.print(
                    Panel(
                        Markdown(content),
                        title="[bold cyan]Context[/bold cyan]",
                        border_style="cyan",
                    )
                )
                status.start()
                status.update(
                    "[bold green]CodeSage[/bold green] [dim]is generating…[/dim]"
                )

            elif chunk_type == "token":
                collected_tokens.append(content)
                words = len("".join(collected_tokens).split())
                status.update(
                    f"[bold green]CodeSage[/bold green] "
                    f"[dim]is generating… ({words} words)[/dim]"
                )

            elif chunk_type == "done":
                if not collected_tokens:
                    # Non-streamed response (commands, clarifications).
                    # Stop spinner, print, then let the with-block clean up.
                    status.stop()
                    console.print(
                        Panel(
                            Markdown(content),
                            title="[bold green]CodeSage[/bold green]",
                            border_style="green",
                            padding=(1, 2),
                        )
                    )

    # Print the full formatted response.
    if collected_tokens:
        console.print(
            Panel(
                Markdown("".join(collected_tokens)),
                title="[bold green]CodeSage[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    if exiting:
        console.print(f"\n[dim]Goodbye![/dim]")
    else:
        console.print()

    return exiting


