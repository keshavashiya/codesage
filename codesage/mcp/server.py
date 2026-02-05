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
        self._memory = None

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

    @property
    def memory(self):
        """Lazy-load memory manager."""
        if self._memory is None:
            from codesage.memory.memory_manager import MemoryManager
            from codesage.llm.embeddings import EmbeddingService

            # Use embedding service for pattern search
            embedder = EmbeddingService(
                self.config.llm,
                self.config.cache_dir,
                self.config.performance,
            )
            self._memory = MemoryManager(embedding_fn=embedder.embed_batch)
        return self._memory

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_code",
                    description="Search the codebase for relevant code using semantic search. Returns matching code snippets with file locations and similarity scores. Use depth='thorough' for security scanning and full impact analysis.",
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
                            "include_graph": {
                                "type": "boolean",
                                "description": "Include graph context (callers, callees, dependencies)",
                                "default": True,
                            },
                            "depth": {
                                "type": "string",
                                "enum": ["quick", "medium", "thorough"],
                                "default": "medium",
                                "description": "Analysis depth - 'thorough' includes security scanning and full impact analysis",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_file_context",
                    description="Get rich context for a file: content, definitions, security issues, and related code.",
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
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="review_code",
                    description="Review current code changes or a specific file for security, bugs, and improvements.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Specific file to review (optional)",
                            },
                            "staged_only": {
                                "type": "boolean",
                                "description": "Review only staged changes (default: false)",
                                "default": False,
                            },
                            "use_llm": {
                                "type": "boolean",
                                "description": "Use LLM for deeper insights (default: true)",
                                "default": True,
                            }
                        },
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
                Tool(
                    name="get_task_context",
                    description="Get comprehensive context for a coding task: relevant files, learned patterns, and user preferences.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Description of what you want to do (e.g. 'implement user login', 'refactor database connection')",
                            },
                        },
                        "required": ["task_description"],
                    },
                ),
                Tool(
                    name="get_implementation_context",
                    description="Assemble a structured context pack for implementing a task. Use depth='thorough' for comprehensive analysis including security scanning.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Task description",
                            },
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional target files",
                            },
                            "include_cross_project": {
                                "type": "boolean",
                                "description": "Include cross-project recommendations",
                                "default": False,
                            },
                            "depth": {
                                "type": "string",
                                "enum": ["quick", "medium", "thorough"],
                                "default": "medium",
                                "description": "Analysis depth - 'thorough' includes security scanning and full impact assessment",
                            },
                        },
                        "required": ["task"],
                    },
                ),
                Tool(
                    name="detect_code_smells",
                    description="Detect pattern-deviation code smells in a file or current changes.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Optional file path to analyze",
                            },
                            "severity": {
                                "type": "string",
                                "description": "Comma-separated severities to include",
                                "default": "warning,error",
                            },
                        },
                    },
                ),
                # Unified context tool (consolidates get_task_context + get_implementation_context)
                Tool(
                    name="get_context",
                    description="Get comprehensive context for a coding task. Unified interface supporting different interaction modes: brainstorm (exploration), implement (task-focused), or review (code review).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Task or query description",
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["brainstorm", "implement", "review"],
                                "default": "implement",
                                "description": "Context mode: brainstorm (exploration), implement (task-focused), review (code review)",
                            },
                            "target_files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional target files to focus on",
                            },
                            "depth": {
                                "type": "string",
                                "enum": ["quick", "medium", "thorough"],
                                "default": "medium",
                                "description": "Analysis depth",
                            },
                            "include_cross_project": {
                                "type": "boolean",
                                "default": False,
                                "description": "Include cross-project recommendations",
                            },
                        },
                        "required": ["task"],
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
                elif name == "review_code":
                    result = await self._tool_review_code(arguments)
                elif name == "analyze_security":
                    result = await self._tool_analyze_security(arguments)
                elif name == "get_stats":
                    result = await self._tool_get_stats(arguments)
                elif name == "get_task_context":
                    result = await self._tool_get_task_context(arguments)
                elif name == "get_implementation_context":
                    result = await self._tool_get_implementation_context(arguments)
                elif name == "detect_code_smells":
                    result = await self._tool_detect_code_smells(arguments)
                elif name == "get_context":
                    result = await self._tool_get_context(arguments)
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
        include_graph = args.get("include_graph", True)
        depth = args.get("depth", "medium")

        # For thorough depth, use deep analyzer
        if depth == "thorough":
            from codesage.core.deep_analyzer import DeepAnalyzer

            analyzer = DeepAnalyzer(self.config)
            deep_result = analyzer.analyze_sync(query, depth="thorough")

            return {
                "query": query,
                "depth": depth,
                "count": len(deep_result.semantic_results),
                "results": deep_result.semantic_results,
                "security_issues": deep_result.security_issues,
                "impact_analysis": deep_result.impact_analysis,
                "risk_score": deep_result.risk_score,
                "recommendations": deep_result.recommendations,
                "patterns": deep_result.patterns,
            }

        suggestions = self.suggester.find_similar(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=True,
            include_graph_context=include_graph,
        )

        return {
            "query": query,
            "depth": depth,
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
                    "callers": s.callers if include_graph else [],
                    "callees": s.callees if include_graph else [],
                    "dependencies": s.dependencies if include_graph else [],
                    "impact_score": s.impact_score if include_graph else None,
                    "relationship_summary": (
                        {
                            "callers": len(s.callers),
                            "callees": len(s.callees),
                            "dependencies": len(s.dependencies),
                        }
                        if include_graph
                        else {}
                    ),
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

        # Get definitions in file
        definitions = []
        try:
            elements = self.db.get_elements_for_file(file_path)
            for el in elements:
                definitions.append({
                    "name": el.name,
                    "type": el.type,
                    "line": el.line_start,
                    "signature": el.signature
                })
        except Exception:
            pass

        # Run quick security scan on this file
        security_issues = []
        try:
            from codesage.security.scanner import SecurityScanner
            scanner = SecurityScanner()
            findings = scanner.scan_file(full_path)
            for f in findings:
                security_issues.append({
                    "severity": f.rule.severity.value,
                    "line": f.line_number,
                    "message": f.rule.message,
                    "suggestion": f.rule.fix_suggestion
                })
        except Exception:
            pass

        return {
            "file": file_path,
            "language": language,
            "line_start": (line_start or 0) + 1,
            "line_count": len(lines),
            "content": content,
            "definitions": definitions,
            "security_issues": security_issues,
        }

    async def _tool_review_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute review_code tool."""
        file_path = args.get("file_path")
        staged_only = args.get("staged_only", False)
        use_llm = args.get("use_llm", True)

        try:
            from codesage.review.hybrid_analyzer import HybridReviewAnalyzer

            analyzer = HybridReviewAnalyzer(config=self.config, repo_path=self.project_path)

            # If specific file requested, we need to construct a change object or filter
            # But the analyzer works on Diff objects locally.
            # For simplicity, if file_path is provided, we analyze just that file if it changed,
            # OR we fallback to static analysis of that file if no diff exists.

            changes = None
            if file_path:
                # Review specific file path - getting changes might be tricky if not in git
                # For now, let's get all changes and filter
                all_changes = analyzer.get_all_changes()
                changes = [c for c in all_changes if str(c.path) == file_path]

                # If no git changes for this file, we might want to still review it?
                # HybridAnalyzer relies on diffs for context.
                # If no diff, we return early
                if not changes:
                    return {"message": "No uncommitted changes found for this file to review."}
            else:
                 if staged_only:
                     changes = analyzer.get_staged_changes()
                 else:
                     changes = analyzer.get_all_changes()

            result = analyzer.review_changes(
                changes=changes,
                use_llm_synthesis=use_llm
            )

            return {
                "summary": result.summary,
                "stats": {
                    "critical": result.critical_count,
                    "warnings": result.warning_count,
                    "security_issues": len([i for i in result.issues if i.severity.name in ("CRITICAL", "WARNING")]),
                },
                "issues": [
                    {
                        "file": str(i.file),
                        "line": i.line,
                        "severity": i.severity.name,
                        "message": i.message,
                        "suggestion": i.suggestion
                    }
                    for i in result.issues
                ]
            }

        except Exception as e:
            return {"error": f"Review failed: {e}"}

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
                embedder = EmbeddingService(
                    self.config.llm,
                    self.config.cache_dir,
                    self.config.performance,
                )
                storage = StorageManager(self.config, embedding_fn=embedder)
                result["storage_metrics"] = storage.get_metrics()
            except Exception as e:
                result["storage_metrics"] = {"error": str(e)}

        return result

    async def _tool_get_task_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_task_context tool."""
        task = args.get("task_description", "")
        if not task:
            return {"error": "task_description is required"}

        context = {
            "task": task,
            "relevant_code": [],
            "learned_patterns": [],
            "user_preferences": {},
            "related_concepts": [],
        }

        # 1. Search for relevant code
        try:
            search_results = self.suggester.find_similar(
                query=task,
                limit=3,
                min_similarity=0.3,
            )
            context["relevant_code"] = [
                {
                    "file": str(s.file),
                    "name": s.name,
                    "type": s.element_type,
                    "similarity": round(s.similarity, 2),
                    "summary": s.docstring[:100] if s.docstring else "No docstring",
                }
                for s in search_results
            ]
        except Exception as e:
            logger.warning(f"Task search failed: {e}")

        # 2. Get learned patterns and preferences from MemoryManager
        try:
            # Find patterns similar to task (e.g. "auth", "api", "error handling")
            patterns = self.memory.find_similar_patterns(task, limit=3)
            context["learned_patterns"] = [
                {
                    "name": p.get("name"),
                    "description": p.get("description"),
                    "confidence": p.get("confidence", 0),
                    "category": p.get("category"),
                }
                for p in patterns
            ]

            # Get user preferences (general)
            prefs = self.memory.get_all_preferences(category="general")
            context["user_preferences"] = prefs

            # Try to infer specialized preferences
            # e.g. if task mentions "test", add test preferences
            if "test" in task.lower():
                test_prefs = self.memory.get_all_preferences(category="testing")
                context["user_preferences"].update(test_prefs)

        except Exception as e:
            logger.warning(f"Memory lookup failed: {e}")
            context["memory_error"] = str(e)

        # 3. Identify related concepts/files (Cross-referencing)
        # If we found code, let's look for what IT relates to
        if context["relevant_code"]:
            primary_match = context["relevant_code"][0]
            try:
                # Naive impact analysis: what files contain elements similar to this?
                # or simplified: just list what other files are in the same module/directory
                path = Path(primary_match["file"])
                siblings = [p.name for p in path.parent.glob("*") if p.is_file() and p.name != path.name][:5]
                if siblings:
                    context["related_concepts"].append(f"Files in {path.parent}: {', '.join(siblings)}")
            except Exception:
                pass

        return context

    async def _tool_get_implementation_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_implementation_context tool."""
        task = args.get("task", "")
        if not task:
            return {"error": "task is required"}

        if not self.config.features.context_provider_mode:
            return {"error": "context_provider_mode feature is disabled"}

        include_cross_project = args.get("include_cross_project", False)
        if include_cross_project and not self.config.features.cross_project_recommendations:
            return {"error": "cross_project_recommendations feature is disabled"}

        depth = args.get("depth", "medium")

        from codesage.core.context_provider import ContextProvider

        provider = ContextProvider(self.config)
        context = provider.get_implementation_context(
            task_description=task,
            target_files=args.get("files"),
            include_cross_project=include_cross_project,
        )

        result = context.to_dict()
        result["depth"] = depth

        # For thorough depth, enrich with deep analysis
        if depth == "thorough":
            from codesage.core.deep_analyzer import DeepAnalyzer

            analyzer = DeepAnalyzer(self.config)
            deep_result = analyzer.analyze_sync(task, depth="thorough")

            result["deep_analysis"] = {
                "risk_score": deep_result.risk_score,
                "impact_analysis": deep_result.impact_analysis,
                "security_issues": deep_result.security_issues,
                "recommendations": deep_result.recommendations,
            }

        return result

    async def _tool_detect_code_smells(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute detect_code_smells tool."""
        if not self.config.features.code_smell_detection:
            return {"error": "code_smell_detection feature is disabled"}

        from codesage.review.smells import PatternDeviationDetector
        from codesage.review.diff import DiffExtractor

        detector = PatternDeviationDetector(self.config)
        file_path = args.get("file_path")
        severity = args.get("severity", "warning,error")
        allowed = {s.strip().lower() for s in severity.split(",") if s.strip()}

        smells = []
        if file_path:
            smells = detector.detect_file(self.project_path / file_path)
        else:
            diff = DiffExtractor(self.project_path)
            changes = diff.get_all_changes()
            for change in changes:
                if change.status == "D":
                    continue
                smells.extend(detector.detect_file(change.path))

        if allowed:
            smells = [s for s in smells if s.severity.lower() in allowed]

        return {"count": len(smells), "results": [s.to_dict() for s in smells]}

    async def _tool_get_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unified get_context tool.

        Combines task context and implementation context with mode-aware behavior.
        """
        task = args.get("task", "")
        if not task:
            return {"error": "task is required"}

        mode = args.get("mode", "implement")
        depth = args.get("depth", "medium")
        target_files = args.get("target_files", [])
        include_cross_project = args.get("include_cross_project", False)

        result = {
            "task": task,
            "mode": mode,
            "depth": depth,
        }

        # Mode-specific context building
        if mode == "brainstorm":
            # Exploration mode: rich search results + patterns + related concepts
            try:
                # Semantic search with more results
                search_results = self.suggester.find_similar(
                    query=task,
                    limit=7,
                    min_similarity=0.2,
                    include_graph_context=True,
                )
                result["relevant_code"] = [
                    {
                        "file": str(s.file),
                        "name": s.name,
                        "type": s.element_type,
                        "similarity": round(s.similarity, 2),
                        "code_preview": s.code[:200] if s.code else "",
                        "callers": len(s.callers) if s.callers else 0,
                        "callees": len(s.callees) if s.callees else 0,
                    }
                    for s in search_results
                ]

                # Get patterns
                patterns = self.memory.find_similar_patterns(task, limit=5)
                result["patterns"] = [
                    {"name": p.get("name"), "description": p.get("description", "")[:100]}
                    for p in patterns
                ]

                # Get preferences
                result["preferences"] = self.memory.get_all_preferences(category="general")

            except Exception as e:
                result["error"] = str(e)

        elif mode == "implement":
            # Implementation mode: structured context pack
            if not self.config.features.context_provider_mode:
                result["warning"] = "context_provider_mode feature is disabled, returning basic context"

            try:
                from codesage.core.context_provider import ContextProvider

                provider = ContextProvider(self.config)
                context = provider.get_implementation_context(
                    task_description=task,
                    target_files=target_files if target_files else None,
                    include_cross_project=include_cross_project,
                )
                impl_data = context.to_dict()

                result["relevant_code"] = impl_data.get("relevant_code", [])
                result["implementation_plan"] = impl_data.get("implementation_plan", {})
                result["suggested_files"] = impl_data.get("suggested_files", [])
                result["dependencies"] = impl_data.get("dependencies", [])
                result["patterns"] = impl_data.get("patterns", [])
                result["security"] = impl_data.get("security", {})

                # For thorough depth, add deep analysis
                if depth == "thorough":
                    from codesage.core.deep_analyzer import DeepAnalyzer

                    analyzer = DeepAnalyzer(self.config)
                    deep_result = analyzer.analyze_sync(task, depth="thorough")

                    result["deep_analysis"] = {
                        "risk_score": deep_result.risk_score,
                        "impact_analysis": deep_result.impact_analysis,
                        "security_issues": deep_result.security_issues,
                        "recommendations": deep_result.recommendations,
                    }

            except Exception as e:
                result["error"] = str(e)

        elif mode == "review":
            # Review mode: focus on code quality and issues
            try:
                # Search for relevant code
                search_results = self.suggester.find_similar(
                    query=task,
                    limit=5,
                    min_similarity=0.3,
                )
                result["relevant_code"] = [
                    {
                        "file": str(s.file),
                        "name": s.name,
                        "type": s.element_type,
                        "similarity": round(s.similarity, 2),
                    }
                    for s in search_results
                ]

                # Include security analysis if files specified
                if target_files:
                    from codesage.security.scanner import SecurityScanner

                    scanner = SecurityScanner()
                    all_findings = []
                    for f in target_files[:5]:  # Limit
                        path = self.project_path / f
                        if path.exists():
                            findings = scanner.scan_file(path)
                            all_findings.extend(findings)

                    result["security_findings"] = [
                        {
                            "severity": f.severity if hasattr(f, 'severity') else "unknown",
                            "message": f.message if hasattr(f, 'message') else str(f),
                            "file": str(f.file_path) if hasattr(f, 'file_path') else "",
                        }
                        for f in all_findings[:10]
                    ]

                # Include code smell detection if enabled
                if self.config.features.code_smell_detection:
                    from codesage.review.smells import PatternDeviationDetector

                    detector = PatternDeviationDetector(self.config)
                    smells = []
                    for f in target_files[:5]:
                        path = self.project_path / f
                        if path.exists():
                            smells.extend(detector.detect_file(path))

                    result["code_smells"] = [s.to_dict() for s in smells[:10]]

            except Exception as e:
                result["error"] = str(e)

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
