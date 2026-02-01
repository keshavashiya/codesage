"""MCP Server implementation for CodeSage.

Provides CodeSage capabilities as MCP tools for integration
with Claude Desktop and other MCP clients.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from codesage.utils.config import Config
from codesage.utils.logging import get_logger

logger = get_logger("mcp.server")

# Optional MCP import
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolResult,
        TextContent,
        Tool,
        Resource,
        ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None


class CodeSageMCPServer:
    """MCP Server that exposes CodeSage capabilities.

    Tools provided:
        - search_code: Semantic code search
        - get_file_context: Get code from specific file/lines
        - analyze_security: Run security scan
        - review_code: AI code review
        - get_stats: Index statistics

    Resources provided:
        - codesage://codebase: Codebase overview
        - codesage://file/{path}: File content
    """

    def __init__(self, project_path: Path):
        """Initialize the MCP server.

        Args:
            project_path: Path to the project directory
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP support requires the mcp package. "
                "Install with: pipx inject pycodesage mcp (or pip install 'pycodesage[mcp]')"
            )

        self.project_path = project_path.resolve()

        # Load config
        try:
            self.config = Config.load(self.project_path)
        except FileNotFoundError:
            raise ValueError(
                f"Project not initialized at {self.project_path}. "
                "Run 'codesage init' first."
            )

        # Lazy-loaded services
        self._suggester = None
        self._scanner = None
        self._analyzer = None
        self._db = None

        # Create MCP server
        self.server = Server("codesage")

        # Register tools and resources
        self._register_tools()
        self._register_resources()

        logger.info(f"CodeSage MCP Server initialized for {self.config.project_name}")

    @property
    def suggester(self):
        """Lazy-load suggester."""
        if self._suggester is None:
            from codesage.core.suggester import Suggester
            self._suggester = Suggester(self.config)
        return self._suggester

    @property
    def db(self):
        """Lazy-load database."""
        if self._db is None:
            from codesage.storage.database import Database
            self._db = Database(self.config.storage.db_path)
        return self._db

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_code",
                    description="Search the codebase for relevant code using semantic search. Returns matching code snippets with file locations and similarity scores.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query describing what code you're looking for",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)",
                                "default": 5,
                            },
                            "min_similarity": {
                                "type": "number",
                                "description": "Minimum similarity threshold 0-1 (default: 0.2)",
                                "default": 0.2,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_file_context",
                    description="Get the content of a specific file or code element with surrounding context.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file (relative to project root)",
                            },
                            "line_start": {
                                "type": "integer",
                                "description": "Starting line number (optional)",
                            },
                            "line_end": {
                                "type": "integer",
                                "description": "Ending line number (optional)",
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="analyze_security",
                    description="Run security analysis on the codebase to detect potential vulnerabilities.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Specific path to analyze (default: entire project)",
                                "default": ".",
                            },
                            "severity": {
                                "type": "string",
                                "description": "Minimum severity level: low, medium, high, critical",
                                "default": "low",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_stats",
                    description="Get statistics about the indexed codebase.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "detailed": {
                                "type": "boolean",
                                "description": "Include detailed storage metrics",
                                "default": False,
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                if name == "search_code":
                    result = await self._tool_search_code(arguments)
                elif name == "get_file_context":
                    result = await self._tool_get_file_context(arguments)
                elif name == "analyze_security":
                    result = await self._tool_analyze_security(arguments)
                elif name == "get_stats":
                    result = await self._tool_get_stats(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(result, indent=2, default=str),
                        )
                    ]
                )

            except Exception as e:
                logger.error(f"Tool {name} failed: {e}")
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({"error": str(e)}),
                        )
                    ]
                )

    async def _tool_search_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search_code tool."""
        query = args.get("query", "")
        limit = args.get("limit", 5)
        min_similarity = args.get("min_similarity", 0.2)

        suggestions = self.suggester.find_similar(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=True,
        )

        return {
            "query": query,
            "count": len(suggestions),
            "results": [
                {
                    "file": str(s.file),
                    "line": s.line,
                    "name": s.name,
                    "type": s.element_type,
                    "similarity": round(s.similarity, 3),
                    "code": s.code,
                    "explanation": s.explanation,
                }
                for s in suggestions
            ],
        }

    async def _tool_get_file_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_file_context tool."""
        file_path = args.get("file_path", "")
        line_start = args.get("line_start")
        line_end = args.get("line_end")

        # Resolve path
        full_path = self.project_path / file_path
        if not full_path.exists():
            return {"error": f"File not found: {file_path}"}

        # Read file
        try:
            content = full_path.read_text()
            lines = content.split("\n")
        except Exception as e:
            return {"error": f"Could not read file: {e}"}

        # Extract lines if specified
        if line_start is not None:
            line_start = max(1, line_start) - 1  # Convert to 0-indexed
            line_end = line_end or (line_start + 50)
            line_end = min(line_end, len(lines))
            lines = lines[line_start:line_end]
            content = "\n".join(lines)

        # Detect language
        suffix = full_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
        }
        language = language_map.get(suffix, "text")

        return {
            "file": file_path,
            "language": language,
            "line_start": (line_start or 0) + 1,
            "line_count": len(lines),
            "content": content,
        }

    async def _tool_analyze_security(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analyze_security tool."""
        path = args.get("path", ".")
        severity = args.get("severity", "low")

        try:
            from codesage.security.scanner import SecurityScanner

            scanner = SecurityScanner()
            target_path = self.project_path / path
            report = scanner.scan(target_path, severity_threshold=severity)

            return {
                "files_scanned": report.files_scanned,
                "total_findings": report.total_findings,
                "findings_by_severity": {
                    "critical": len([f for f in report.findings if f.severity == "critical"]),
                    "high": len([f for f in report.findings if f.severity == "high"]),
                    "medium": len([f for f in report.findings if f.severity == "medium"]),
                    "low": len([f for f in report.findings if f.severity == "low"]),
                },
                "findings": [
                    {
                        "rule_id": f.rule_id,
                        "severity": f.severity,
                        "message": f.message,
                        "file": str(f.file_path),
                        "line": f.line_number,
                    }
                    for f in report.findings[:20]  # Limit to first 20
                ],
            }
        except Exception as e:
            return {"error": f"Security scan failed: {e}"}

    async def _tool_get_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_stats tool."""
        detailed = args.get("detailed", False)

        db_stats = self.db.get_stats()

        result = {
            "project": self.config.project_name,
            "files_indexed": db_stats.get("files", 0),
            "code_elements": db_stats.get("elements", 0),
            "last_indexed": db_stats.get("last_indexed"),
            "language": self.config.language,
        }

        if detailed:
            from codesage.storage.manager import StorageManager
            from codesage.llm.embeddings import EmbeddingService

            try:
                embedder = EmbeddingService(self.config.llm, self.config.cache_dir)
                storage = StorageManager(self.config, embedding_fn=embedder.embedder)
                result["storage_metrics"] = storage.get_metrics()
            except Exception as e:
                result["storage_metrics"] = {"error": str(e)}

        return result

    def _register_resources(self) -> None:
        """Register MCP resources."""

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="codesage://codebase",
                    name="Codebase Overview",
                    description=f"Overview of the {self.config.project_name} codebase",
                    mimeType="application/json",
                ),
            ]

        @self.server.list_resource_templates()
        async def list_resource_templates() -> List[ResourceTemplate]:
            """List resource templates."""
            return [
                ResourceTemplate(
                    uriTemplate="codesage://file/{path}",
                    name="Source File",
                    description="Get content of a source file",
                    mimeType="text/plain",
                ),
                ResourceTemplate(
                    uriTemplate="codesage://search/{query}",
                    name="Code Search",
                    description="Search for code matching query",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri) -> str:
            """Read a resource by URI."""
            # Convert AnyUrl to string for comparison
            uri_str = str(uri)
            if uri_str == "codesage://codebase":
                stats = self.db.get_stats()
                return json.dumps({
                    "project": self.config.project_name,
                    "path": str(self.project_path),
                    "language": self.config.language,
                    "files_indexed": stats.get("files", 0),
                    "code_elements": stats.get("elements", 0),
                    "last_indexed": stats.get("last_indexed"),
                }, indent=2)

            if uri_str.startswith("codesage://file/"):
                file_path = uri_str.replace("codesage://file/", "")
                full_path = self.project_path / file_path
                if full_path.exists():
                    return full_path.read_text()
                return f"File not found: {file_path}"

            if uri_str.startswith("codesage://search/"):
                query = uri_str.replace("codesage://search/", "")
                results = await self._tool_search_code({"query": query, "limit": 5})
                return json.dumps(results, indent=2)

            return f"Unknown resource: {uri_str}"

    async def run_stdio(self) -> None:
        """Run the MCP server on stdio (single client, process-based)."""
        logger.info("Starting CodeSage MCP Server (stdio transport)...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    async def run_sse(self, host: str = "localhost", port: int = 8080) -> None:
        """Run the MCP server with HTTP/SSE transport (multi-client, network-based).

        Args:
            host: Host to bind to (default: localhost)
            port: Port to bind to (default: 8080)
        """
        try:
            from mcp.server.sse import sse_server
        except ImportError:
            logger.error(
                "SSE transport requires additional dependencies. "
                "Install with: pip install 'mcp[sse]' or use stdio transport."
            )
            raise

        logger.info(f"Starting CodeSage MCP Server (HTTP/SSE transport) on {host}:{port}")
        logger.info(f"Server endpoint: http://{host}:{port}/sse")
        logger.info("Multiple clients can connect simultaneously")

        async with sse_server(host=host, port=port) as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def run_mcp_server(
    project_path: Path,
    transport: str = "stdio",
    host: str = "localhost",
    port: int = 8080,
) -> None:
    """Run the MCP server with specified transport.

    Args:
        project_path: Path to the project directory
        transport: Transport type - "stdio" or "sse" (default: stdio)
        host: Host for SSE transport (default: localhost)
        port: Port for SSE transport (default: 8080)
    """
    server = CodeSageMCPServer(project_path)

    if transport == "stdio":
        await server.run_stdio()
    elif transport == "sse":
        await server.run_sse(host=host, port=port)
    else:
        raise ValueError(f"Unknown transport: {transport}. Use 'stdio' or 'sse'")

