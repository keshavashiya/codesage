"""Command parsing for chat interface."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ChatCommand(Enum):
    """Available chat commands."""

    # Core commands
    HELP = auto()
    SEARCH = auto()
    CONTEXT = auto()
    CLEAR = auto()
    STATS = auto()
    EXIT = auto()
    MESSAGE = auto()  # Regular message (not a command)
    UNKNOWN = auto()  # Unknown command

    # Enhanced commands (Phase 1)
    DEEP = auto()       # Deep multi-agent analysis
    PLAN = auto()       # Implementation planning
    REVIEW = auto()     # Code review
    SECURITY = auto()   # Security analysis
    IMPACT = auto()     # Impact/blast radius analysis
    PATTERNS = auto()   # Show learned patterns
    SIMILAR = auto()    # Find similar code
    MODE = auto()       # Switch interaction mode
    EXPORT = auto()     # Export conversation


# Interaction modes
class ChatMode(Enum):
    """Chat interaction modes."""

    BRAINSTORM = "brainstorm"   # Open-ended exploration
    IMPLEMENT = "implement"     # Task-focused implementation
    REVIEW = "review"           # Change review focus


# Command mapping with aliases
COMMAND_MAP = {
    # Core commands
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

    # Enhanced commands
    "/deep": ChatCommand.DEEP,
    "/d": ChatCommand.DEEP,
    "/analyze": ChatCommand.DEEP,
    "/plan": ChatCommand.PLAN,
    "/p": ChatCommand.PLAN,
    "/implement": ChatCommand.PLAN,
    "/review": ChatCommand.REVIEW,
    "/r": ChatCommand.REVIEW,
    "/security": ChatCommand.SECURITY,
    "/sec": ChatCommand.SECURITY,
    "/impact": ChatCommand.IMPACT,
    "/blast": ChatCommand.IMPACT,
    "/patterns": ChatCommand.PATTERNS,
    "/pat": ChatCommand.PATTERNS,
    "/similar": ChatCommand.SIMILAR,
    "/sim": ChatCommand.SIMILAR,
    "/mode": ChatCommand.MODE,
    "/m": ChatCommand.MODE,
    "/export": ChatCommand.EXPORT,
    "/save": ChatCommand.EXPORT,
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
            # Core commands
            ChatCommand.HELP: "Show available commands and usage",
            ChatCommand.SEARCH: "Search codebase: /search <query>",
            ChatCommand.CONTEXT: "Show current code context settings",
            ChatCommand.CLEAR: "Clear conversation history",
            ChatCommand.STATS: "Show index statistics",
            ChatCommand.EXIT: "Exit chat mode",

            # Enhanced commands
            ChatCommand.DEEP: "Deep multi-agent analysis: /deep <query>",
            ChatCommand.PLAN: "Generate implementation plan: /plan <task>",
            ChatCommand.REVIEW: "Review code changes: /review [file]",
            ChatCommand.SECURITY: "Security analysis: /security [path]",
            ChatCommand.IMPACT: "Impact analysis: /impact <element>",
            ChatCommand.PATTERNS: "Show learned patterns: /patterns [query]",
            ChatCommand.SIMILAR: "Find similar code: /similar <element>",
            ChatCommand.MODE: "Switch mode: /mode <brainstorm|implement|review>",
            ChatCommand.EXPORT: "Export conversation: /export [file]",
        }
        return help_texts.get(command, "Unknown command")

    def get_all_commands(self) -> list:
        """Get list of all available commands.

        Returns:
            List of (command_string, description) tuples
        """
        seen = set()
        commands = []

        # Order matters for display - show primary aliases first
        primary_aliases = [
            "/help", "/search", "/deep", "/plan", "/review",
            "/security", "/impact", "/patterns", "/similar",
            "/mode", "/stats", "/context", "/export", "/clear", "/exit",
        ]

        for cmd_str in primary_aliases:
            cmd_type = COMMAND_MAP.get(cmd_str)
            if cmd_type and cmd_type not in seen:
                seen.add(cmd_type)
                commands.append((cmd_str, self.get_command_help(cmd_type)))

        return commands

    def get_commands_by_category(self) -> dict:
        """Get commands organized by category.

        Returns:
            Dictionary with category -> list of (command, description) tuples
        """
        return {
            "Search & Analysis": [
                ("/search <query>", "Semantic code search"),
                ("/deep <query>", "Deep multi-agent analysis"),
                ("/similar <element>", "Find similar code"),
                ("/patterns [query]", "Show learned patterns"),
            ],
            "Planning & Review": [
                ("/plan <task>", "Generate implementation plan"),
                ("/review [file]", "Review code changes"),
                ("/security [path]", "Security analysis"),
                ("/impact <element>", "Impact/blast radius analysis"),
            ],
            "Session": [
                ("/mode <mode>", "Switch mode (brainstorm/implement/review)"),
                ("/context", "Show context settings"),
                ("/stats", "Show index statistics"),
                ("/export [file]", "Export conversation"),
                ("/clear", "Clear history"),
                ("/help", "Show this help"),
                ("/exit", "Exit chat"),
            ],
        }
