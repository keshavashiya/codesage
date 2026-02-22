"""MCP (Model Context Protocol) server for CodeSage.

Provides CodeSage capabilities as MCP tools and resources
for integration with Claude Desktop and other MCP clients.

Installation:
    pip install 'pycodesage[mcp]'
    # or with pipx:
    pipx inject pycodesage mcp

Usage:
    codesage mcp serve   # Start MCP server
    codesage mcp setup   # Print IDE configuration snippet
    codesage mcp test    # Smoke-test all tools
"""

from typing import TYPE_CHECKING

# Check if MCP is available
MCP_AVAILABLE = False
try:
    __import__("mcp")
    MCP_AVAILABLE = True
except ImportError:
    pass


def check_mcp_available() -> None:
    """Check if MCP is installed, raise helpful error if not."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP support requires the mcp package.\n"
            "Install with:  pip install 'pycodesage[mcp]'\n"
            "If using pipx: pipx inject pycodesage mcp"
        )


if TYPE_CHECKING or MCP_AVAILABLE:
    from .server import CodeSageMCPServer
    from .global_server import GlobalCodeSageMCPServer


__all__ = ["CodeSageMCPServer", "GlobalCodeSageMCPServer", "MCP_AVAILABLE", "check_mcp_available"]
