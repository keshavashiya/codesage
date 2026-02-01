"""Chat interface for CodeSage.

Provides an interactive chat mode for asking questions about
the codebase using natural language.
"""

from .engine import ChatEngine
from .models import ChatMessage, ChatSession
from .commands import ChatCommand, CommandParser, ParsedCommand
from .context import CodeContextBuilder

__all__ = [
    "ChatEngine",
    "ChatMessage",
    "ChatSession",
    "ChatCommand",
    "CommandParser",
    "ParsedCommand",
    "CodeContextBuilder",
]
