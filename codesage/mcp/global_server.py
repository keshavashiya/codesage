"""Global MCP server that works with all indexed CodeSage projects.

This server operates at the ~/.codesage/ level and can access any indexed project.
Tools accept a project_name or project_path parameter to target specific projects.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import CallToolResult, Resource, TextContent, Tool

from codesage.utils.config import Config
from codesage.utils.logging import get_logger

logger = get_logger("mcp.global_server")

# Check if MCP is available
try:
    from mcp.server.stdio import stdio_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class GlobalCodeSageMCPServer:
    """Global MCP server for all CodeSage projects.

    Provides tools that work across all indexed projects:
    - search_code: Search any project or all projects
    - list_projects: Get all indexed projects
    - get_file_context: Get code from any project
    - analyze_security: Run security scan on any project
    - get_stats: Get stats for any project or global stats
    """

    def __init__(self, global_dir: Optional[Path] = None):
        """Initialize the global MCP server.

        Args:
            global_dir: Global CodeSage directory (default: ~/.codesage)
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP support requires the mcp package. "
                "Install with: pipx inject pycodesage mcp (or pip install 'pycodesage[mcp]')"
            )

        self.global_dir = global_dir or Path.home() / ".codesage"
        self.global_dir.mkdir(parents=True, exist_ok=True)

        # Track indexed projects
        self._projects: Dict[str, Path] = {}
        self._discover_projects()

        # Lazy-loaded global services
        self._memory_manager = None

        # Create MCP server
        self.server = Server("codesage-global")

        # Register tools and resources
        self._register_tools()
        self._register_resources()

        logger.info(f"Global CodeSage MCP Server initialized ({len(self._projects)} projects)")

    def _discover_projects(self) -> None:
        """Discover all indexed CodeSage projects."""
        # Method 1: Check global memory system
        try:
            from codesage.memory.memory_manager import MemoryManager
            memory = MemoryManager(global_dir=self.global_dir / "developer")
            projects = memory.preference_store.get_all_projects()

            for project in projects:
                if project.path and project.path.exists():
                    config_path = project.path / ".codesage" / "config.yaml"
                    if config_path.exists():
                        self._projects[project.name] = project.path
                        logger.debug(f"Found project: {project.name} at {project.path}")
        except Exception as e:
            logger.debug(f"Could not load projects from memory: {e}")

        logger.info(f"Discovered {len(self._projects)} projects")

    def _get_project_path(self, project_name: Optional[str] = None, project_path: Optional[str] = None) -> Optional[Path]:
        """Get project path from name or path parameter.

        Args:
            project_name: Project name
            project_path: Project path

        Returns:
            Path to project or None if not found
        """
        if project_path:
            path = Path(project_path).resolve()
            if (path / ".codesage").exists():
                return path
            return None

        if project_name:
            return self._projects.get(project_name)

        return None

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="list_projects",
                    description="List all indexed CodeSage projects",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="search_code",
                    description="Search code across one or all projects using semantic search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query",
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Optional: Search in specific project only",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results per project",
                                "default": 5,
                            },
                            "min_similarity": {
                                "type": "number",
                                "description": "Minimum similarity score (0.0-1.0)",
                                "default": 0.2,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_file_context",
                    description="Get content of a specific file with optional line range",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Project name",
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to file (relative to project root)",
                            },
                            "line_start": {
                                "type": "integer",
                                "description": "Optional start line (1-indexed)",
                            },
                            "line_end": {
                                "type": "integer",
                                "description": "Optional end line (1-indexed)",
                            },
                        },
                        "required": ["project_name", "file_path"],
                    },
                ),
                Tool(
                    name="get_stats",
                    description="Get statistics for a project or global stats",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Optional: Get stats for specific project",
                            },
                            "detailed": {
                                "type": "boolean",
                                "description": "Include detailed metrics",
                                "default": False,
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> CallToolResult:
            """Handle tool calls."""
            try:
                if name == "list_projects":
                    result = await self._tool_list_projects(arguments or {})
                elif name == "search_code":
                    result = await self._tool_search_code(arguments or {})
                elif name == "get_file_context":
                    result = await self._tool_get_file_context(arguments or {})
                elif name == "get_stats":
                    result = await self._tool_get_stats(arguments or {})
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

    def _register_resources(self) -> None:
        """Register MCP resources."""

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            resources = [
                Resource(
                    uri="codesage://projects",
                    name="All Projects",
                    description="List of all indexed CodeSage projects",
                    mimeType="application/json",
                )
            ]

            # Add resource for each project
            for project_name in self._projects.keys():
                resources.append(
                    Resource(
                        uri=f"codesage://project/{project_name}",
                        name=f"Project: {project_name}",
                        description=f"Overview of {project_name}",
                        mimeType="application/json",
                    )
                )

            return resources

        @self.server.read_resource()
        async def read_resource(uri) -> str:
            """Read a resource."""
            # Convert AnyUrl to string for comparison
            uri_str = str(uri)
            if uri_str == "codesage://projects":
                projects_info = []
                for name, path in self._projects.items():
                    try:
                        config = Config.load(path)
                        projects_info.append({
                            "name": name,
                            "path": str(path),
                            "language": config.language,
                        })
                    except Exception:
                        projects_info.append({
                            "name": name,
                            "path": str(path),
                        })

                return json.dumps(projects_info, indent=2)

            elif uri_str.startswith("codesage://project/"):
                project_name = uri_str.replace("codesage://project/", "")
                project_path = self._projects.get(project_name)

                if not project_path:
                    return json.dumps({"error": f"Project not found: {project_name}"})

                try:
                    config = Config.load(project_path)
                    from codesage.storage.database import Database
                    db = Database(config.storage.db_path)
                    stats = db.get_stats()

                    return json.dumps({
                        "name": config.project_name,
                        "path": str(project_path),
                        "language": config.language,
                        "stats": stats,
                    }, indent=2)
                except Exception as e:
                    return json.dumps({"error": str(e)})

            return json.dumps({"error": "Unknown resource"})

    async def _tool_list_projects(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all projects tool."""
        projects = []
        for name, path in self._projects.items():
            try:
                config = Config.load(path)
                projects.append({
                    "name": name,
                    "path": str(path),
                    "language": config.language,
                })
            except Exception:
                projects.append({
                    "name": name,
                    "path": str(path),
                })

        return {"projects": projects, "count": len(projects)}

    async def _tool_search_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search code across projects."""
        query = args.get("query", "")
        project_name = args.get("project_name")
        limit = args.get("limit", 5)
        min_similarity = args.get("min_similarity", 0.2)

        if not query:
            return {"error": "Query is required"}

        results = []

        # Search specific project or all projects
        projects_to_search = {}
        if project_name:
            path = self._projects.get(project_name)
            if path:
                projects_to_search[project_name] = path
        else:
            projects_to_search = self._projects

        for name, path in projects_to_search.items():
            try:
                config = Config.load(path)
                from codesage.core.suggester import Suggester
                suggester = Suggester(config)

                project_results = suggester.find_similar(
                    query=query,
                    limit=limit,
                    min_similarity=min_similarity,
                    include_explanations=False,
                )

                for result in project_results:
                    result["project"] = name
                    results.append(result)

            except Exception as e:
                logger.warning(f"Failed to search {name}: {e}")

        # Sort by similarity
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Limit total results
        if limit and len(results) > limit * 3:
            results = results[:limit * 3]

        return {"query": query, "count": len(results), "results": results}

    async def _tool_get_file_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get file context from a project."""
        project_name = args.get("project_name")
        file_path = args.get("file_path")
        line_start = args.get("line_start")
        line_end = args.get("line_end")

        if not project_name or not file_path:
            return {"error": "project_name and file_path are required"}

        project_path = self._projects.get(project_name)
        if not project_path:
            return {"error": f"Project not found: {project_name}"}

        full_path = project_path / file_path
        if not full_path.exists():
            return {"error": f"File not found: {file_path}"}

        try:
            with open(full_path, "r") as f:
                lines = f.readlines()

            if line_start is not None and line_end is not None:
                lines = lines[line_start - 1:line_end]
                content = "".join(lines)
            else:
                content = "".join(lines)

            return {
                "project": project_name,
                "file": file_path,
                "line_start": line_start or 1,
                "line_count": len(lines),
                "content": content,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_get_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get project or global stats."""
        project_name = args.get("project_name")
        detailed = args.get("detailed", False)

        if project_name:
            # Get stats for specific project
            project_path = self._projects.get(project_name)
            if not project_path:
                return {"error": f"Project not found: {project_name}"}

            try:
                config = Config.load(project_path)
                from codesage.storage.database import Database
                db = Database(config.storage.db_path)
                stats = db.get_stats()

                return {
                    "project": project_name,
                    "path": str(project_path),
                    **stats,
                }
            except Exception as e:
                return {"error": str(e)}
        else:
            # Get global stats
            global_stats = {
                "total_projects": len(self._projects),
                "projects": list(self._projects.keys()),
            }

            if detailed:
                per_project = {}
                for name, path in self._projects.items():
                    try:
                        config = Config.load(path)
                        from codesage.storage.database import Database
                        db = Database(config.storage.db_path)
                        per_project[name] = db.get_stats()
                    except Exception:
                        per_project[name] = {"error": "Could not load stats"}

                global_stats["per_project"] = per_project

            return global_stats

    async def run_stdio(self) -> None:
        """Run the MCP server on stdio."""
        logger.info("Starting Global CodeSage MCP Server (stdio transport)...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    async def run_sse(self, host: str = "localhost", port: int = 8080) -> None:
        """Run the MCP server with HTTP/SSE transport."""
        try:
            from mcp.server.sse import sse_server
        except ImportError:
            logger.error(
                "SSE transport requires additional dependencies. "
                "Install with: pipx inject pycodesage 'mcp[sse]' (or pip install 'mcp[sse]') or use stdio transport."
            )
            raise

        logger.info(f"Starting Global CodeSage MCP Server (HTTP/SSE transport) on {host}:{port}")
        logger.info(f"Server endpoint: http://{host}:{port}/sse")
        logger.info(f"Serving {len(self._projects)} projects")

        async with sse_server(host=host, port=port) as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )
