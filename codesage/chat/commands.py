"""Command parsing for chat interface."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class ChatCommand(Enum):
    """Available chat commands."""

    HELP = auto()
    SEARCH = auto()
    CONTEXT = auto()
    CLEAR = auto()
    STATS = auto()
    EXIT = auto()
    MESSAGE = auto()  # Regular message (not a command)
    UNKNOWN = auto()  # Unknown command


# Command mapping
COMMAND_MAP = {
    "/help": ChatCommand.HELP,
    "/h": ChatCommand.HELP,
    "/?": ChatCommand.HELP,
    "/search": ChatCommand.SEARCH,
    "/s": ChatCommand.SEARCH,
    "/find": ChatCommand.SEARCH,
    "/context": ChatCommand.CONTEXT,
    "/ctx": ChatCommand.CONTEXT,
    "/clear": ChatCommand.CLEAR,
    "/reset": ChatCommand.CLEAR,
    "/stats": ChatCommand.STATS,
    "/status": ChatCommand.STATS,
    "/exit": ChatCommand.EXIT,
    "/quit": ChatCommand.EXIT,
    "/q": ChatCommand.EXIT,
}


@dataclass
class ParsedCommand:
    """Parsed command result.

    Attributes:
        command: The command type
        args: Optional arguments for the command
        raw_input: The original input string
    """

    command: ChatCommand
    args: Optional[str] = None
    raw_input: str = ""


class CommandParser:
    """Parses user input into commands.

    Handles:
    - Command detection (starting with /)
    - Argument extraction
    - Unknown command handling
    """

    def parse(self, user_input: str) -> ParsedCommand:
        """Parse user input into a command.

        Args:
            user_input: Raw user input string

        Returns:
            ParsedCommand with command type and optional args
        """
        stripped = user_input.strip()

        # Empty input
        if not stripped:
            return ParsedCommand(
                command=ChatCommand.MESSAGE,
                args=None,
                raw_input=user_input,
            )

        # Not a command
        if not stripped.startswith("/"):
            return ParsedCommand(
                command=ChatCommand.MESSAGE,
                args=stripped,
                raw_input=user_input,
            )

        # Parse command and args
        parts = stripped.split(maxsplit=1)
        cmd_str = parts[0].lower()
        args = parts[1] if len(parts) > 1 else None

        # Look up command
        command = COMMAND_MAP.get(cmd_str, ChatCommand.UNKNOWN)

        return ParsedCommand(
            command=command,
            args=args,
            raw_input=user_input,
        )

    def get_command_help(self, command: ChatCommand) -> str:
        """Get help text for a specific command.

        Args:
            command: Command to get help for

        Returns:
            Help text string
        """
        help_texts = {
            ChatCommand.HELP: "Show available commands and usage",
            ChatCommand.SEARCH: "Search codebase: /search <query>",
            ChatCommand.CONTEXT: "Show current code context settings",
            ChatCommand.CLEAR: "Clear conversation history",
            ChatCommand.STATS: "Show index statistics",
            ChatCommand.EXIT: "Exit chat mode",
        }
        return help_texts.get(command, "Unknown command")

    def get_all_commands(self) -> list:
        """Get list of all available commands.

        Returns:
            List of (command_string, description) tuples
        """
        seen = set()
        commands = []

        for cmd_str, cmd_type in COMMAND_MAP.items():
            if cmd_type not in seen and cmd_type != ChatCommand.UNKNOWN:
                seen.add(cmd_type)
                commands.append((cmd_str, self.get_command_help(cmd_type)))

        return commands
