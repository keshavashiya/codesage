"""Command modules for CodeSage CLI."""

from codesage.cli.commands.chat import chat
from codesage.cli.commands.health import health
from codesage.cli.commands.index import index
from codesage.cli.commands.init import init
from codesage.cli.commands.review import review
from codesage.cli.commands.stats import stats
from codesage.cli.commands.suggest import suggest
from codesage.cli.commands.search import search  # New: replaces suggest
from codesage.cli.commands.version import version
from codesage.cli.commands.context import context

__all__ = [
    "init",
    "index",
    "search",    # Primary search command
    "suggest",   # Deprecated alias for search
    "stats",
    "health",
    "version",
    "review",
    "chat",
    "context",
]
