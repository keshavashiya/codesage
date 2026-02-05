"""Command groups for CodeSage CLI."""

from codesage.cli.groups import (
    config,    # New: unified config
    docs,
    features,  # Deprecated: use config features
    hooks,     # Deprecated: use config hooks
    mcp,
    profile,
    security,  # Deprecated: use review --security
    smells,    # Deprecated: use review --smells
    storage,   # Deprecated: use config storage
)

__all__ = [
    "config",
    "docs",
    "features",
    "hooks",
    "mcp",
    "profile",
    "security",
    "smells",
    "storage",
]
