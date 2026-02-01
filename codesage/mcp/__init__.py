"""MCP (Model Context Protocol) server for CodeSage.

Provides CodeSage capabilities as MCP tools and resources
for integration with Claude Desktop and other MCP clients.

Installation:
    pipx inject pycodesage mcp (or pip install 'pycodesage[mcp]')

Usage:
    codesage mcp serve  # Start MCP server
    codesage mcp install  # Install for Claude Desktop
"""

from typing import TYPE_CHECKING

# Check if MCP is available
MCP_AVAILABLE = False
try:
    import mcp
    MCP_AVAILABLE = True
except ImportError:
    pass


def check_mcp_available() -> None:
    """Check if MCP is installed, raise helpful error if not."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP support requires the mcp package. "
            "Install with: pipx inject pycodesage mcp (or pip install 'pycodesage[mcp]')"
        )


if TYPE_CHECKING or MCP_AVAILABLE:
    from .server import CodeSageMCPServer
    from .global_server import GlobalCodeSageMCPServer


__all__ = ["CodeSageMCPServer", "GlobalCodeSageMCPServer", "MCP_AVAILABLE", "check_mcp_available"]
