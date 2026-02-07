"""Global MCP server that works with all indexed CodeSage projects.

This server operates at the ~/.codesage/ level and can access any indexed project.
Tools accept a project_name or project_path parameter to target specific projects.
All project-level tools are available here — specify project_name to use them.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import CallToolResult, Resource, ResourceTemplate, TextContent, Tool

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

    Provides all tools across all indexed projects:
    - list_projects: Get all indexed projects
    - search_code: Search any project or all projects
    - get_file_context: Get code from any project
    - get_stats: Get stats for any project or global stats
    - get_developer_profile: Developer preferences and learned patterns
    - review_code: AI code review on any project
    - analyze_security: Security scanning on any project
    - explain_concept: LLM-synthesized concept explanation
    - suggest_approach: Implementation guidance with patterns
    - trace_flow: Call chain tracing through dependency graph
    - find_examples: Usage example search grouped by pattern
    - recommend_pattern: Cross-project pattern recommendations
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

        # Cache project-level MCP servers for delegation
        self._project_servers: Dict[str, Any] = {}

        self._initialize_server()

    @property
    def memory(self):
        """Lazy-load memory manager."""
        if self._memory_manager is None:
            from codesage.memory.memory_manager import MemoryManager
            self._memory_manager = MemoryManager(global_dir=self.global_dir / "developer")
        return self._memory_manager

    def _get_project_server(self, project_name: str):
        """Get or create a cached project-level MCP server.

        Args:
            project_name: Name of the project.

        Returns:
            CodeSageMCPServer instance for the project.

        Raises:
            ValueError: If project not found.
        """
        if project_name in self._project_servers:
            return self._project_servers[project_name]

        project_path = self._projects.get(project_name)
        if not project_path:
            raise ValueError(f"Project not found: {project_name}")

        from codesage.mcp.server import CodeSageMCPServer
        server = CodeSageMCPServer(project_path)
        self._project_servers[project_name] = server
        return server

    def _resolve_project(self, args: Dict[str, Any]) -> str:
        """Resolve project_name from args, raising clear error if missing.

        Args:
            args: Tool arguments dict.

        Returns:
            project_name string.

        Raises:
            ValueError: If project_name not provided or project not found.
        """
        project_name = args.get("project_name")
        if not project_name:
            available = ", ".join(self._projects.keys()) if self._projects else "none"
            raise ValueError(
                f"project_name is required. Available projects: {available}"
            )
        if project_name not in self._projects:
            available = ", ".join(self._projects.keys()) if self._projects else "none"
            raise ValueError(
                f"Project not found: {project_name}. Available projects: {available}"
            )
        return project_name

    def _initialize_server(self):
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
        """Get project path from name or path parameter."""
        if project_path:
            path = Path(project_path).resolve()
            if (path / ".codesage").exists():
                return path
            return None

        if project_name:
            return self._projects.get(project_name)

        return None

    # =========================================================================
    # Tool Registration
    # =========================================================================

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        # Common project_name property for tool schemas
        _project_name_prop = {
            "type": "string",
            "description": "Project name (required for project-specific tools)",
        }

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                # --- Global-only tools ---
                Tool(
                    name="list_projects",
                    description="List all indexed CodeSage projects",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="get_developer_profile",
                    description="Get developer profile: preferences, coding style, and learned patterns.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                # --- Cross-project tools ---
                Tool(
                    name="search_code",
                    description=(
                        "Search code across one or all projects using semantic search. "
                        "Returns matching snippets with file locations, similarity scores, "
                        "and confidence tiers."
                    ),
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
                    description=(
                        "Get content of a specific file with definitions, security issues, "
                        "and related code context."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": _project_name_prop,
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
                # --- Project tools (delegated to project server) ---
                Tool(
                    name="review_code",
                    description=(
                        "Review code changes for bugs, security issues, code smells, "
                        "and improvements. Runs hybrid analysis on staged/uncommitted "
                        "changes or a specific file."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": _project_name_prop,
                            "file_path": {
                                "type": "string",
                                "description": "Specific file to review (optional)",
                            },
                            "staged_only": {
                                "type": "boolean",
                                "description": "Review only staged changes",
                                "default": False,
                            },
                            "use_llm": {
                                "type": "boolean",
                                "description": "Use LLM for deeper insights",
                                "default": True,
                            },
                        },
                        "required": ["project_name"],
                    },
                ),
                Tool(
                    name="analyze_security",
                    description=(
                        "Scan code for security vulnerabilities. Detects injection flaws, "
                        "auth issues, secrets exposure, insecure crypto, and misconfigurations."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": _project_name_prop,
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
                        "required": ["project_name"],
                    },
                ),
                Tool(
                    name="explain_concept",
                    description=(
                        "Understand how a concept, pattern, or feature is implemented "
                        "in a project. Performs semantic search and synthesizes a "
                        "narrative explanation using LLM."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": _project_name_prop,
                            "concept": {
                                "type": "string",
                                "description": "The concept, pattern, or feature to explain",
                            },
                            "depth": {
                                "type": "string",
                                "enum": ["quick", "medium", "thorough"],
                                "default": "medium",
                                "description": "How deeply to analyze the concept",
                            },
                        },
                        "required": ["project_name", "concept"],
                    },
                ),
                Tool(
                    name="suggest_approach",
                    description=(
                        "Get implementation guidance for a coding task. Returns relevant "
                        "code, learned patterns, cross-project recommendations, security "
                        "notes, and suggested files to modify."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": _project_name_prop,
                            "task": {
                                "type": "string",
                                "description": "Description of what you want to implement",
                            },
                            "target_files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional target files to focus on",
                            },
                            "include_cross_project": {
                                "type": "boolean",
                                "description": "Include patterns from other projects",
                                "default": False,
                            },
                        },
                        "required": ["project_name", "task"],
                    },
                ),
                Tool(
                    name="trace_flow",
                    description=(
                        "Trace how code flows through the codebase. Finds a code element "
                        "by name, then traces callers, callees, and transitive call chains "
                        "using the dependency graph."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": _project_name_prop,
                            "element_name": {
                                "type": "string",
                                "description": "Name of the function, method, or class to trace",
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["callers", "callees", "both"],
                                "default": "both",
                                "description": "Direction to trace",
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum depth for transitive tracing",
                                "default": 3,
                            },
                        },
                        "required": ["project_name", "element_name"],
                    },
                ),
                Tool(
                    name="find_examples",
                    description=(
                        "Find usage examples of a pattern, function, or coding style. "
                        "Groups results by directory/pattern variation."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": _project_name_prop,
                            "pattern": {
                                "type": "string",
                                "description": "The pattern, function name, or coding style to find",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum total examples to return",
                                "default": 10,
                            },
                        },
                        "required": ["project_name", "pattern"],
                    },
                ),
                Tool(
                    name="recommend_pattern",
                    description=(
                        "Get pattern recommendations from the developer's learned patterns "
                        "and cross-project memory. Returns matching patterns with code "
                        "examples and confidence scores."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Optional: Get project-specific patterns too",
                            },
                            "context": {
                                "type": "string",
                                "description": "What you're trying to do or the kind of pattern you need",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum patterns to return",
                                "default": 5,
                            },
                        },
                        "required": ["context"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> CallToolResult:
            """Handle tool calls."""
            try:
                args = arguments or {}

                # Global-only tools
                if name == "list_projects":
                    result = await self._tool_list_projects(args)
                elif name == "get_developer_profile":
                    result = await self._tool_get_developer_profile(args)
                # Cross-project tools (own implementation)
                elif name == "search_code":
                    result = await self._tool_search_code(args)
                elif name == "get_stats":
                    result = await self._tool_get_stats(args)
                # Project-delegated tools (with own file context handler)
                elif name == "get_file_context":
                    result = await self._tool_get_file_context(args)
                # Project-delegated tools
                elif name in (
                    "review_code", "analyze_security", "explain_concept",
                    "suggest_approach", "trace_flow", "find_examples",
                    "recommend_pattern",
                ):
                    result = await self._delegate_to_project(name, args)
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

    # =========================================================================
    # Project Delegation
    # =========================================================================

    async def _delegate_to_project(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate a tool call to a project-level MCP server.

        Args:
            tool_name: Name of the tool to call.
            args: Tool arguments (must include project_name).

        Returns:
            Tool result dict.
        """
        project_name = self._resolve_project(args)
        server = self._get_project_server(project_name)

        # Map tool names to project server handler methods
        handler_map = {
            "review_code": server._tool_review_code,
            "analyze_security": server._tool_analyze_security,
            "explain_concept": server._tool_explain_concept,
            "suggest_approach": server._tool_suggest_approach,
            "trace_flow": server._tool_trace_flow,
            "find_examples": server._tool_find_examples,
            "recommend_pattern": server._tool_recommend_pattern,
        }

        handler = handler_map.get(tool_name)
        if not handler:
            return {"error": f"No handler for tool: {tool_name}"}

        # Remove project_name from args before delegating (project server doesn't expect it)
        project_args = {k: v for k, v in args.items() if k != "project_name"}
        return await handler(project_args)

    # =========================================================================
    # Resources
    # =========================================================================

    def _register_resources(self) -> None:
        """Register MCP resources."""

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            # Refresh projects discovery
            self._discover_projects()

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

            # Add developer profile resource
            resources.append(
                Resource(
                    uri="codesage://developer/profile",
                    name="Developer Profile",
                    description="User preferences, coding style, and learned patterns",
                    mimeType="application/json",
                )
            )

            return resources

        @self.server.list_resource_templates()
        async def list_resource_templates() -> List[ResourceTemplate]:
            """List resource templates."""
            from mcp.types import ResourceTemplate

            return [
                ResourceTemplate(
                    uriTemplate="codesage://project/{project_name}/file/{path}",
                    name="Source File (Global)",
                    description="Get content of a file from any project",
                    mimeType="text/plain",
                ),
                ResourceTemplate(
                    uriTemplate="codesage://project/{project_name}/search/{query}",
                    name="Code Search (Global)",
                    description="Search for code in a specific project",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri) -> str:
            """Read a resource."""
            # Refresh projects discovery
            self._discover_projects()

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

            elif uri_str == "codesage://developer/profile":
                profile = await self._tool_get_developer_profile({})
                return json.dumps(profile, indent=2)

            elif uri_str.startswith("codesage://project/"):
                # Handle templates first (longer paths)
                if "/file/" in uri_str:
                    # codesage://project/{project_name}/file/{path}
                    parts = uri_str.replace("codesage://project/", "").split("/file/", 1)
                    if len(parts) != 2:
                        return "Invalid file URI format"

                    project_name, file_path = parts
                    project_path = self._projects.get(project_name)

                    if not project_path:
                        return f"Project not found: {project_name}"

                    full_path = project_path / file_path
                    if full_path.exists():
                        return full_path.read_text()
                    return f"File not found: {file_path} in {project_name}"

                elif "/search/" in uri_str:
                    # codesage://project/{project_name}/search/{query}
                    parts = uri_str.replace("codesage://project/", "").split("/search/", 1)
                    if len(parts) != 2:
                        return json.dumps({"error": "Invalid search URI format"})

                    project_name, query = parts
                    results = await self._tool_search_code({
                        "query": query,
                        "project_name": project_name,
                        "limit": 5
                    })
                    return json.dumps(results, indent=2)

                else:
                    # codesage://project/{name} (Project overview)
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

    # =========================================================================
    # Global-only Tool Handlers
    # =========================================================================

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
        """Search code across projects.

        When project_name is specified, delegates to the project server
        for envelope-format results with confidence scoring.
        When omitted, searches across all projects.
        """
        query = args.get("query", "")
        project_name = args.get("project_name")
        limit = args.get("limit", 5)
        min_similarity = args.get("min_similarity", 0.2)

        if not query:
            return {"error": "Query is required"}

        # Single-project search: delegate to project server for full envelope
        if project_name:
            try:
                server = self._get_project_server(project_name)
                return await server._tool_search_code(args)
            except ValueError as e:
                return {"error": str(e)}

        # Multi-project search
        results = []
        for name, path in self._projects.items():
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

                for suggestion in project_results:
                    result_dict = suggestion.to_dict()
                    result_dict["project"] = name
                    results.append(result_dict)

            except Exception as e:
                logger.warning(f"Failed to search {name}: {e}")

        # Sort by similarity
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Limit total results
        if limit and len(results) > limit * 3:
            results = results[:limit * 3]

        return {"query": query, "count": len(results), "results": results}

    async def _tool_get_file_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get file context from a project.

        Delegates to project server for enriched results (definitions, security).
        """
        project_name = args.get("project_name")
        file_path = args.get("file_path")

        if not project_name or not file_path:
            return {"error": "project_name and file_path are required"}

        try:
            server = self._get_project_server(project_name)
            # Delegate — project server returns envelope with definitions + security
            project_args = {k: v for k, v in args.items() if k != "project_name"}
            return await server._tool_get_file_context(project_args)
        except ValueError as e:
            return {"error": str(e)}

    async def _tool_get_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get project or global stats."""
        project_name = args.get("project_name")
        detailed = args.get("detailed", False)

        if project_name:
            # Delegate to project server for envelope format
            try:
                server = self._get_project_server(project_name)
                return await server._tool_get_stats(args)
            except ValueError as e:
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

    async def _tool_get_developer_profile(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get developer profile (preferences and patterns)."""
        try:
            # 1. Get all preferences
            preferences = self.memory.get_all_preferences(category=None)

            # 2. Get top cross-project patterns
            patterns = self.memory.get_cross_project_patterns(min_projects=2)

            # Format patterns
            formatted_patterns = [
                {
                    "name": p.get("name"),
                    "description": p.get("description"),
                    "occurrences": p.get("occurrences", 0),
                    "projects": p.get("projects", []),
                }
                for p in patterns
            ]

            return {
                "preferences": preferences,
                "learned_patterns": formatted_patterns,
                "interaction_stats": self.memory.get_interaction_stats(),
            }
        except Exception as e:
            return {"error": f"Failed to load profile: {e}"}

    # =========================================================================
    # Server Transports
    # =========================================================================

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
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.responses import Response
            from starlette.routing import Mount, Route
            import uvicorn
        except ImportError:
            logger.error(
                "SSE transport requires additional dependencies. "
                "Install with: pipx inject pycodesage 'mcp[sse]' (or pip install 'mcp[sse]') or use stdio transport."
            )
            raise

        logger.info(f"Starting Global CodeSage MCP Server (HTTP/SSE transport) on {host}:{port}")
        logger.info(f"Server endpoint: http://{host}:{port}/sse")
        logger.info(f"Serving {len(self._projects)} projects")

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self.server.run(
                    streams[0],
                    streams[1],
                    self.server.create_initialization_options(),
                )
            return Response()

        starlette_app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        config = uvicorn.Config(
            starlette_app, host=host, port=port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
