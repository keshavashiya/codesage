"""MCP Server implementation for CodeSage.

Provides CodeSage capabilities as MCP tools for integration
with Claude Desktop and other MCP clients.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from codesage.utils.config import Config
from codesage.utils.logging import get_logger
from codesage.memory.style_analyzer import StyleAnalyzer as _StyleAnalyzer

logger = get_logger("mcp.server")

# Single source of truth for Python-only pattern names.
# Used to filter out irrelevant patterns when the project language is not Python.
_PYTHON_SPECIFIC_PATTERNS: frozenset = _StyleAnalyzer.PYTHON_ONLY_PATTERN_NAMES

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
        - search_code: Semantic code search with confidence scoring
        - get_file_context: File content + definitions + security issues
        - review_code: AI code review (absorbs code smells)
        - analyze_security: Security vulnerability scanning
        - get_stats: Index statistics
        - explain_concept: Semantic search + LLM synthesis for concepts
        - suggest_approach: Implementation context + patterns + suggestions
        - trace_flow: Call chain tracing through dependency graph
        - find_examples: Search + group by pattern variation
        - recommend_pattern: Cross-project pattern recommendations

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
        self._llm_provider = None
        self._confidence_scorer = None

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

            embedder = EmbeddingService(
                self.config.llm,
                self.config.cache_dir,
                self.config.performance,
            )
            self._memory = MemoryManager(
                embedding_fn=embedder.embed_batch,
                vector_dim=embedder.get_dimension(),
            )
        return self._memory

    @property
    def llm_provider(self):
        """Lazy-load LLM provider for tools that need synthesis."""
        if self._llm_provider is None:
            from codesage.llm.provider import LLMProvider

            self._llm_provider = LLMProvider(self.config.llm)
        return self._llm_provider

    @property
    def confidence_scorer(self):
        """Lazy-load confidence scorer."""
        if self._confidence_scorer is None:
            from codesage.core.confidence import ConfidenceScorer

            self._confidence_scorer = ConfidenceScorer(
                graph_store=getattr(self.suggester.storage, "graph_store", None),
                memory_manager=self._memory,
                project_path=str(self.project_path),
            )
        return self._confidence_scorer

    def _build_envelope(
        self,
        results: Any,
        confidence_tier: str = "medium",
        confidence_score: float = 0.5,
        narrative: str = "",
        suggested_followup: Optional[List[str]] = None,
        search_time_ms: Optional[float] = None,
        sources_used: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build a standardized response envelope for tool results."""
        envelope = {
            "confidence": confidence_tier,
            "confidence_score": confidence_score,
            "narrative": narrative,
            "results": results,
            "suggested_followup": suggested_followup or [],
            "metadata": {},
        }
        if search_time_ms is not None:
            envelope["metadata"]["search_time_ms"] = round(search_time_ms, 1)
        if sources_used:
            envelope["metadata"]["sources_used"] = sources_used
        return envelope

    def _calculate_dynamic_threshold(
        self, query: str, base_threshold: float = 0.2
    ) -> float:
        """Calculate dynamic similarity threshold based on query characteristics.

        Args:
            query: Search query string
            base_threshold: Starting threshold (default: 0.2)

        Returns:
            Adjusted threshold between 0.1 and 0.5
        """
        if not query or not query.strip():
            return base_threshold

        query_lower = query.lower().strip()
        words = query_lower.split()
        word_count = len(words)

        # Factor 1: Query length adjustment
        if word_count <= 2:
            length_factor = 0.15  # Short queries need higher precision
        elif word_count <= 5:
            length_factor = 0.0
        else:
            length_factor = -0.05  # Long queries can use lower threshold

        # Factor 2: Technical terminology boost
        technical_terms = {
            "function",
            "class",
            "method",
            "implementation",
            "algorithm",
            "pattern",
            "architecture",
        }
        has_technical = any(term in words for term in technical_terms)
        specificity_boost = 0.05 if has_technical else 0.0

        # Factor 3: Code naming patterns (snake_case or camelCase)
        has_underscore = "_" in query and any(c.isalpha() for c in query.split("_")[0])
        has_camel = (
            any(c.isupper() and query[i].islower() for i, c in enumerate(query[1:], 1))
            if len(query) > 1
            else False
        )

        if has_underscore or has_camel:
            specificity_boost += 0.1

        threshold = base_threshold + length_factor + specificity_boost
        return max(0.1, min(0.5, threshold))

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_code",
                    description=(
                        "Use when you need to find code by meaning. "
                        "Performs semantic search across the indexed codebase and returns "
                        "matching snippets with file locations, similarity scores, confidence "
                        "tiers, and graph context (callers/callees/dependencies)."
                    ),
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
                    description=(
                        "Use when you need to read a source file with its definitions, "
                        "security issues, and related code context. Returns file content, "
                        "language detection, all definitions (functions/classes), and any "
                        "security vulnerabilities found."
                    ),
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
                    description=(
                        "Use when you need to review code changes for bugs, security issues, "
                        "code smells, and improvements. Runs hybrid analysis (static + LLM) "
                        "on staged/uncommitted changes or a specific file. Absorbs code smell "
                        "detection — no separate tool needed."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Specific file to review (optional, reviews all changes if omitted)",
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
                            },
                        },
                    },
                ),
                Tool(
                    name="analyze_security",
                    description=(
                        "Use when you need to scan code for security vulnerabilities. "
                        "Detects injection flaws, auth issues, secrets exposure, insecure "
                        "crypto, and misconfigurations. Returns findings grouped by severity "
                        "with remediation advice."
                    ),
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
                    description=(
                        "Use when you need an overview of the indexed codebase. "
                        "Returns file count, code element count, last indexed time, "
                        "language, and optional detailed storage metrics."
                    ),
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
                    name="explain_concept",
                    description=(
                        "Use when you need to understand how a concept, pattern, or feature "
                        "is implemented in the codebase. Performs semantic search, groups results "
                        "by file/module, and synthesizes a narrative explanation using LLM."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string",
                                "description": "The concept, pattern, or feature to explain (e.g., 'authentication flow', 'error handling strategy')",
                            },
                            "depth": {
                                "type": "string",
                                "enum": ["quick", "medium", "thorough"],
                                "default": "medium",
                                "description": "How deeply to analyze the concept",
                            },
                        },
                        "required": ["concept"],
                    },
                ),
                Tool(
                    name="suggest_approach",
                    description=(
                        "Use when you need implementation guidance for a coding task. "
                        "Returns relevant code, learned patterns, cross-project recommendations, "
                        "security notes, and suggested files to modify. Replaces get_task_context "
                        "and get_implementation_context."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
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
                        "required": ["task"],
                    },
                ),
                Tool(
                    name="trace_flow",
                    description=(
                        "Use when you need to trace how code flows through the codebase. "
                        "Finds a code element by name, then traces callers, callees, and "
                        "transitive call chains using the dependency graph. Returns step-by-step "
                        "paths with file:line locations."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_name": {
                                "type": "string",
                                "description": "Name of the function, method, or class to trace",
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["callers", "callees", "both"],
                                "default": "both",
                                "description": "Direction to trace: who calls it, what it calls, or both",
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum depth for transitive tracing (default: 3)",
                                "default": 3,
                            },
                        },
                        "required": ["element_name"],
                    },
                ),
                Tool(
                    name="find_examples",
                    description=(
                        "Use when you need to find usage examples of a pattern, function, "
                        "or coding style. Performs broad semantic search, groups results by "
                        "directory/pattern variation, and uses LLM to explain what each "
                        "group does differently."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The pattern, function name, or coding style to find examples of",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum total examples to return (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["pattern"],
                    },
                ),
                Tool(
                    name="recommend_pattern",
                    description=(
                        "Use when you need pattern recommendations from the developer's "
                        "learned patterns and cross-project memory. Returns matching patterns "
                        "with code examples, confidence scores, and source project labels."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "What you're trying to do or the kind of pattern you need",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum patterns to return (default: 5)",
                                "default": 5,
                            },
                        },
                        "required": ["context"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                # Removed tools: return migration error
                if name in (
                    "get_task_context",
                    "get_implementation_context",
                    "get_context",
                    "detect_code_smells",
                ):
                    result = {
                        "error": (
                            f"Tool '{name}' has been replaced. "
                            "Use 'suggest_approach' for implementation context "
                            "or 'review_code' for code smells."
                        )
                    }
                elif name == "search_code":
                    result = await self._tool_search_code(arguments)
                elif name == "get_file_context":
                    result = await self._tool_get_file_context(arguments)
                elif name == "review_code":
                    result = await self._tool_review_code(arguments)
                elif name == "analyze_security":
                    result = await self._tool_analyze_security(arguments)
                elif name == "get_stats":
                    result = await self._tool_get_stats(arguments)
                elif name == "explain_concept":
                    result = await self._tool_explain_concept(arguments)
                elif name == "suggest_approach":
                    result = await self._tool_suggest_approach(arguments)
                elif name == "trace_flow":
                    result = await self._tool_trace_flow(arguments)
                elif name == "find_examples":
                    result = await self._tool_find_examples(arguments)
                elif name == "recommend_pattern":
                    result = await self._tool_recommend_pattern(arguments)
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
    # Existing Tool Handlers (updated with response envelope)
    # =========================================================================

    async def _tool_search_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search_code tool with confidence scoring."""
        start = time.time()
        query = args.get("query", "")
        limit = args.get("limit", 5)
        min_similarity = args.get("min_similarity", 0.2)
        include_graph = args.get("include_graph", True)
        depth = args.get("depth", "medium")

        # Apply dynamic threshold if using default
        if min_similarity == 0.2:
            min_similarity = self._calculate_dynamic_threshold(
                query, base_threshold=0.2
            )

        # For thorough depth, use deep analyzer
        if depth == "thorough":
            try:
                from codesage.core.deep_analyzer import DeepAnalyzer

                analyzer = DeepAnalyzer(self.config)
                deep_result = analyzer.analyze_sync(query, depth="thorough")

                return self._build_envelope(
                    results=deep_result.semantic_results,
                    confidence_tier="high",
                    confidence_score=0.8,
                    narrative=f"Deep analysis of '{query}' with security scanning and impact analysis.",
                    suggested_followup=[
                        f"explain_concept(concept='{query}')",
                        "trace_flow(element_name='<top_result_name>')",
                    ],
                    search_time_ms=(time.time() - start) * 1000,
                    sources_used=["deep_analyzer", "security_scanner", "graph_store"],
                )
            except Exception as e:
                logger.warning(
                    f"Deep analysis failed, falling back to standard search: {e}"
                )

        suggestions = self.suggester.find_similar(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=True,
            include_graph_context=include_graph,
        )

        # Build results with confidence
        results = []
        total_confidence = 0.0
        for s in suggestions:
            result_item = {
                "file": str(s.file),
                "line": s.line,
                "name": s.name,
                "type": s.element_type,
                "similarity": round(s.similarity, 3),
                "confidence_score": s.confidence_score,
                "confidence_tier": s.confidence_tier,
                "code": s.code,
                "explanation": s.explanation,
            }
            if include_graph:
                result_item["callers"] = s.callers
                result_item["callees"] = s.callees
                result_item["dependencies"] = s.dependencies
                result_item["impact_score"] = s.impact_score
                result_item["relationship_summary"] = {
                    "callers": len(s.callers),
                    "callees": len(s.callees),
                    "dependencies": len(s.dependencies),
                }
            results.append(result_item)
            total_confidence += s.confidence_score

        avg_confidence = total_confidence / len(results) if results else 0.0
        tier = (
            "high"
            if avg_confidence > 0.7
            else "medium"
            if avg_confidence > 0.4
            else "low"
        )

        if results:
            top = results[0]
            top_name = top.get("name", "?")
            top_file = top.get("file", "?")
            top_sim = top.get("similarity", 0)
            narrative = (
                f"Found {len(results)} result{'s' if len(results) != 1 else ''} for '{query}'. "
                f"Top match: '{top_name}' in {top_file} ({top_sim:.0%} similar)."
            )
            expl = top.get("explanation", "")
            if expl:
                narrative += f" {expl[:150]}"
        else:
            narrative = f"No results found for '{query}'. Try broader terms or run 'codesage index'."

        return self._build_envelope(
            results=results,
            confidence_tier=tier,
            confidence_score=round(avg_confidence, 3),
            narrative=narrative,
            suggested_followup=[
                f"get_file_context(file_path='{results[0].get('file', '<file>')}')",
                f"explain_concept(concept='{query}')",
            ]
            if results
            else [f"suggest_approach(task='{query}')"],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["vector_store", "graph_store"]
            if include_graph
            else ["vector_store"],
        )

    async def _tool_get_file_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_file_context tool."""
        start = time.time()
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
            line_start = max(1, line_start) - 1
            line_end = line_end or (line_start + 50)
            line_end = min(line_end, len(lines))
            lines = lines[line_start:line_end]
            content = "\n".join(lines)

        # Detect language and file type
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

        # Determine if this is a documentation or configuration file
        is_documentation = suffix in {".md", ".txt", ".rst", ".adoc", ".markdown"}
        is_configuration = suffix in {
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
        }

        # Handle documentation and config files specially
        if is_documentation or is_configuration:
            file_type = "documentation" if is_documentation else "configuration"

            file_result = {
                "file": file_path,
                "language": language,
                "line_start": (line_start or 0) + 1,
                "line_count": len(lines),
                "content": content,
                "definitions": [],
                "security_issues": [],
                "file_type": file_type,
            }

            return self._build_envelope(
                results=file_result,
                confidence_tier="high",
                confidence_score=1.0,
                narrative=f"{file_type.capitalize()} file '{file_path}' with {len(lines)} lines.",
                suggested_followup=[
                    f"search_code(query='references to {file_path}')",
                ],
                search_time_ms=(time.time() - start) * 1000,
                sources_used=["filesystem"],
            )

        # Get definitions in file
        definitions = []
        try:
            elements = self.db.get_elements_for_file(file_path)
            for el in elements:
                definitions.append(
                    {
                        "name": el.name,
                        "type": el.type,
                        "line": el.line_start,
                        "signature": el.signature,
                    }
                )
        except Exception:
            pass

        # Run quick security scan on this file
        security_issues = []
        try:
            from codesage.security.scanner import SecurityScanner

            scanner = SecurityScanner()
            findings = scanner.scan_file(full_path)
            for f in findings:
                security_issues.append(
                    {
                        "severity": f.rule.severity.value,
                        "line": f.line_number,
                        "message": f.rule.message,
                        "suggestion": f.rule.fix_suggestion,
                    }
                )
        except Exception:
            pass

        file_result = {
            "file": file_path,
            "language": language,
            "line_start": (line_start or 0) + 1,
            "line_count": len(lines),
            "content": content,
            "definitions": definitions,
            "security_issues": security_issues,
        }

        # Build narrative that's honest about index state
        if definitions:
            defs_summary = f"{len(definitions)} definitions"
            index_note = ""
        else:
            defs_summary = "0 definitions in index"
            index_note = " File may not be indexed yet — run 'codesage index' to add it."

        sec_note = (
            f" Found {len(security_issues)} security issue(s)."
            if security_issues
            else " No security issues."
        )
        narrative = (
            f"File '{file_path}' — {defs_summary}.{sec_note}{index_note}"
        )

        return self._build_envelope(
            results=file_result,
            confidence_tier="high",
            confidence_score=1.0,
            narrative=narrative,
            suggested_followup=[
                f"search_code(query='functions in {file_path}')",
                f"review_code(file_path='{file_path}')",
            ],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["filesystem", "database", "security_scanner"],
        )

    async def _review_diff_content(
        self, diff_content: str, start: float
    ) -> Dict[str, Any]:
        """Run static analysis on a caller-supplied unified diff string.

        Parses the diff, extracts the *added* lines for each supported file,
        and runs the appropriate checkers (Python or generic) so MCP clients
        can review an arbitrary diff (e.g. a PR) rather than the local git
        working-tree state.  Supported extensions: .py, .rs, .go, .js, .jsx,
        .mjs, .ts, .tsx, .mts.
        """
        from pathlib import Path as _Path
        from codesage.review.checks.python_checks import PythonBadPracticeChecker
        from codesage.review.checks.complexity import ComplexityChecker
        from codesage.review.checks.generic_checks import GenericFileChecker
        from codesage.review.checks.treesitter_checks import (
            TreeSitterReviewChecker,
            TREESITTER_AVAILABLE,
        )
        from codesage.security.rules import get_enabled_rules
        from codesage.utils.language_detector import SUPPORTED_EXTENSIONS

        # Parse diff: collect added lines per file
        file_additions: Dict[str, list] = {}
        current_file = None
        for line in diff_content.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
                file_additions.setdefault(current_file, [])
            elif line.startswith("+++ /dev/null"):
                current_file = None
            elif current_file and line.startswith("+") and not line.startswith("+++"):
                file_additions[current_file].append(line[1:])

        py_checker = PythonBadPracticeChecker()
        cx_checker = ComplexityChecker()
        # Prefer tree-sitter for non-Python; fall back to generic text checks
        non_py_checker = (
            TreeSitterReviewChecker() if TREESITTER_AVAILABLE else GenericFileChecker()
        )

        # Security findings (SecurityFinding objects — different schema)
        sec_rules = get_enabled_rules()

        issues = []

        for rel_path, added_lines in file_additions.items():
            ext = _Path(rel_path).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            content = "\n".join(added_lines)
            file_path_obj = _Path(rel_path)

            # --- static / complexity ---
            if ext == ".py":
                static_hits = py_checker.check(file_path_obj, content) + cx_checker.check(file_path_obj, content)
            else:
                static_hits = non_py_checker.check(file_path_obj, content)

            for f in static_hits:
                sev = f.severity.upper() if isinstance(f.severity, str) else str(f.severity)
                issues.append({
                    "file": str(f.file),
                    "line": f.line,
                    "severity": sev,
                    "category": f.category,
                    "rule_id": f.rule_id,
                    "message": f.message,
                    "suggestion": f.suggestion or "",
                })

            # --- security scanner (in-memory, no disk I/O) ---
            for rule in sec_rules:
                if not rule.enabled:
                    continue
                for match in rule.matches(content):
                    line_number = content[: match.start()].count("\n") + 1
                    sev_name = rule.severity.name if hasattr(rule.severity, "name") else str(rule.severity)
                    issues.append({
                        "file": rel_path,
                        "line": line_number,
                        "severity": sev_name,
                        "category": "security",
                        "rule_id": rule.id,
                        "message": rule.message,
                        "suggestion": rule.fix_suggestion or "",
                        "match": match.group(0)[:120],
                        "cwe": rule.cwe_id or "",
                    })

        critical = sum(1 for i in issues if i["severity"] in ("CRITICAL", "HIGH"))
        warnings = sum(1 for i in issues if i["severity"] == "WARNING")
        files_reviewed = len([p for p in file_additions if _Path(p).suffix.lower() in SUPPORTED_EXTENSIONS])

        return self._build_envelope(
            results={
                "summary": f"{files_reviewed} file(s) reviewed in diff | {critical} critical/high | {warnings} warnings",
                "stats": {
                    "total_issues": len(issues),
                    "files_reviewed": files_reviewed,
                    "by_severity": {
                        "critical_high": critical,
                        "warnings": warnings,
                        "suggestions": len(issues) - critical - warnings,
                    },
                },
                "issues": issues,
            },
            confidence_tier="high" if files_reviewed > 0 else "low",
            confidence_score=0.9,
            narrative=f"Static analysis of diff: {len(issues)} issues found across {files_reviewed} file(s). {critical} critical/high severity.",
            suggested_followup=["analyze_security()"] if critical > 0 else ["get_stats()"],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["static_analysis", "security_scanner"],
        )

    async def _tool_review_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute review_code tool.

        If 'diff' is provided as a unified diff string, runs static analysis on
        the modified files extracted from that diff.  Otherwise falls back to
        reviewing the current git working-tree changes.
        """
        start = time.time()
        diff_content = args.get("diff", "").strip()
        file_path = args.get("file_path")
        staged_only = args.get("staged_only", False)
        use_llm = args.get("use_llm", True)

        # If caller supplied an explicit diff, run our static pipeline on it
        if diff_content:
            return await self._review_diff_content(diff_content, start)

        try:
            from codesage.review.hybrid_analyzer import HybridReviewAnalyzer

            analyzer = HybridReviewAnalyzer(
                config=self.config, repo_path=self.project_path
            )

            changes = None
            if file_path:
                all_changes = analyzer.get_all_changes()
                changes = [c for c in all_changes if str(c.path) == file_path]
                if not changes:
                    return self._build_envelope(
                        results={
                            "message": "No uncommitted changes found for this file."
                        },
                        confidence_tier="high",
                        confidence_score=1.0,
                        narrative="No changes to review.",
                        search_time_ms=(time.time() - start) * 1000,
                    )
            else:
                if staged_only:
                    changes = analyzer.get_staged_changes()
                else:
                    changes = analyzer.get_all_changes()

            result = analyzer.review_changes(
                changes=changes,
                use_llm_synthesis=use_llm,
            )

            # Calculate detailed statistics
            issues_by_severity = {
                "CRITICAL": 0,
                "HIGH": 0,
                "MEDIUM": 0,
                "WARNING": 0,
                "LOW": 0,
                "SUGGESTION": 0,
                "INFO": 0,
            }
            issues_by_category = {
                "security": 0,
                "performance": 0,
                "maintainability": 0,
                "style": 0,
                "other": 0,
            }

            for issue in result.issues:
                # Count by severity
                sev = issue.severity.name
                if sev in issues_by_severity:
                    issues_by_severity[sev] += 1

                # Categorize by message content
                msg_lower = issue.message.lower()
                if any(
                    term in msg_lower
                    for term in ["security", "vulnerability", "injection", "xss", "sql"]
                ):
                    issues_by_category["security"] += 1
                elif any(
                    term in msg_lower
                    for term in ["performance", "slow", "inefficient", "optimize"]
                ):
                    issues_by_category["performance"] += 1
                elif issue.severity.name == "WARNING":
                    issues_by_category["maintainability"] += 1
                else:
                    issues_by_category["other"] += 1

            files_affected = len(set(str(i.file) for i in result.issues))

            review_results = {
                "summary": result.summary,
                "stats": {
                    "total_issues": len(result.issues),
                    "files_affected": files_affected,
                    "by_severity": {
                        "critical": issues_by_severity["CRITICAL"],
                        "high": issues_by_severity["HIGH"],
                        "medium": issues_by_severity["MEDIUM"],
                        "low": issues_by_severity["LOW"],
                        "warnings": issues_by_severity["WARNING"],
                    },
                    "by_category": {
                        "security": issues_by_category["security"],
                        "performance": issues_by_category["performance"],
                        "maintainability": issues_by_category["maintainability"],
                        "other": issues_by_category["other"],
                    },
                    "deprecated_fields": {
                        "critical": result.critical_count,
                        "warnings": result.warning_count,
                    },
                },
                "issues": [
                    {
                        "file": str(i.file),
                        "line": i.line,
                        "severity": i.severity.name,
                        "category": "security"
                        if any(
                            term in i.message.lower()
                            for term in ["security", "vulnerability"]
                        )
                        else "maintainability"
                        if i.severity.name == "WARNING"
                        else "other",
                        "message": i.message,
                        "suggestion": i.suggestion,
                    }
                    for i in result.issues
                ],
            }

            issue_count = len(result.issues)
            tier = (
                "high" if issue_count == 0 else "medium" if issue_count < 5 else "low"
            )

            critical_n = issues_by_severity.get("CRITICAL", 0)
            high_n = issues_by_severity.get("HIGH", 0)
            medium_n = issues_by_severity.get("MEDIUM", 0)
            warning_n = issues_by_severity.get("WARNING", 0)
            suggestion_n = issues_by_severity.get("SUGGESTION", 0) + issues_by_severity.get("INFO", 0) + issues_by_severity.get("LOW", 0)
            sev_parts = []
            if critical_n: sev_parts.append(f"{critical_n} critical")
            if high_n: sev_parts.append(f"{high_n} high")
            if medium_n: sev_parts.append(f"{medium_n} medium")
            if warning_n: sev_parts.append(f"{warning_n} warning{'s' if warning_n != 1 else ''}")
            if suggestion_n: sev_parts.append(f"{suggestion_n} suggestion{'s' if suggestion_n != 1 else ''}")
            sev_summary = ", ".join(sev_parts) if sev_parts else "no blocking issues"
            review_narrative = (
                f"Review complete: {issue_count} issue{'s' if issue_count != 1 else ''} "
                f"across {files_affected} file{'s' if files_affected != 1 else ''} "
                f"({sev_summary})."
            )

            return self._build_envelope(
                results=review_results,
                confidence_tier=tier,
                confidence_score=0.8,
                narrative=review_narrative,
                suggested_followup=[
                    "analyze_security()"
                    if critical_n > 0 or high_n > 0 or medium_n > 0
                    else "get_stats()",
                ],
                search_time_ms=(time.time() - start) * 1000,
                sources_used=["hybrid_analyzer", "llm"]
                if use_llm
                else ["hybrid_analyzer"],
            )

        except Exception as e:
            return {"error": f"Review failed: {e}"}

    async def _tool_analyze_security(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analyze_security tool."""
        start = time.time()
        path = args.get("path", ".")

        try:
            from codesage.security.scanner import SecurityScanner

            scanner = SecurityScanner()
            target_path = self.project_path / path

            # Support both file and directory paths
            if target_path.is_file():
                raw_findings = scanner.scan_file(target_path)
                files_scanned = 1
                total_count = len(raw_findings)
            else:
                report = scanner.scan_directory(target_path)
                raw_findings = report.findings
                files_scanned = report.files_scanned
                total_count = report.total_count

            findings_list = [
                {
                    "rule_id": f.rule.id,
                    "severity": f.severity.value,
                    "message": f.rule.message,
                    "file": str(f.file),
                    "line": f.line_number,
                }
                for f in raw_findings[:20]
            ]

            security_results = {
                "files_scanned": files_scanned,
                "total_findings": total_count,
                "findings_by_severity": {
                    "critical": len([f for f in raw_findings if f.severity.value == "critical"]),
                    "high": len([f for f in raw_findings if f.severity.value == "high"]),
                    "medium": len([f for f in raw_findings if f.severity.value == "medium"]),
                    "low": len([f for f in raw_findings if f.severity.value == "low"]),
                },
                "findings": findings_list,
            }

            total = total_count
            tier = "high" if total == 0 else "medium" if total < 10 else "low"

            return self._build_envelope(
                results=security_results,
                confidence_tier=tier,
                confidence_score=0.9,
                narrative=f"Scanned {files_scanned} file{'s' if files_scanned != 1 else ''}, found {total} security issue{'s' if total != 1 else ''}.",
                suggested_followup=[
                    f"get_file_context(file_path='{findings_list[0]['file']}')"
                    if findings_list
                    else "get_stats()",
                ],
                search_time_ms=(time.time() - start) * 1000,
                sources_used=["security_scanner"],
            )
        except Exception as e:
            return {"error": f"Security scan failed: {e}"}

    async def _tool_get_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_stats tool."""
        start = time.time()
        detailed = args.get("detailed", False)

        db_stats = self.db.get_stats()

        stats_result = {
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
                stats_result["storage_metrics"] = storage.get_metrics()
            except Exception as e:
                stats_result["storage_metrics"] = {"error": str(e)}

        return self._build_envelope(
            results=stats_result,
            confidence_tier="high",
            confidence_score=1.0,
            narrative=f"Project '{self.config.project_name}': {db_stats.get('files', 0)} files, {db_stats.get('elements', 0)} elements.",
            suggested_followup=["search_code(query='main entry point')"],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["database"],
        )

    # =========================================================================
    # New Tool Handlers
    # =========================================================================

    async def _tool_explain_concept(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Explain a concept by searching + LLM synthesis."""
        start = time.time()
        concept = args.get("concept", "")
        depth = args.get("depth", "medium")

        if not concept:
            return {"error": "concept is required"}

        limit = {"quick": 3, "medium": 5, "thorough": 10}.get(depth, 5)

        # Calculate dynamic threshold for better search quality
        dynamic_threshold = self._calculate_dynamic_threshold(
            concept, base_threshold=0.2
        )

        # Semantic search for the concept
        suggestions = self.suggester.find_similar(
            query=concept,
            limit=limit,
            min_similarity=dynamic_threshold,
            include_explanations=False,
            include_graph_context=True,
        )

        if not suggestions:
            return self._build_envelope(
                results=[],
                confidence_tier="low",
                confidence_score=0.1,
                narrative=f"No code found related to '{concept}'.",
                suggested_followup=[f"search_code(query='{concept}')"],
                search_time_ms=(time.time() - start) * 1000,
            )

        # Group results by file/module
        by_module: Dict[str, list] = {}
        for s in suggestions:
            module = str(Path(str(s.file)).parent)
            by_module.setdefault(module, []).append(s)

        # Build context for LLM synthesis
        context_parts = [f"Concept: {concept}\n"]
        for module, items in by_module.items():
            context_parts.append(f"\n## Module: {module}")
            for item in items[:3]:
                context_parts.append(
                    f"- **{item.name or item.element_type}** at `{item.file}:{item.line}` "
                    f"(similarity: {item.similarity:.0%})"
                )
                if item.code:
                    context_parts.append(f"```\n{item.code[:300]}\n```")

        # Also check patterns from memory
        patterns = []
        try:
            patterns = self.memory.find_similar_patterns(concept, limit=3)
            if patterns:
                context_parts.append("\n## Relevant Patterns")
                for p in patterns:
                    context_parts.append(
                        f"- **{p.get('name', '?')}**: {p.get('description', '')[:100]}"
                    )
        except Exception:
            pass

        # Guard: skip LLM synthesis when average similarity is too low (high false-positive risk)
        avg_sim_check = sum(s.similarity for s in suggestions) / len(suggestions)
        if avg_sim_check < 0.35:
            narrative = (
                f"Found {len(suggestions)} loosely related code elements "
                f"(avg similarity {avg_sim_check:.0%}) — results may not be directly relevant to '{concept}'. "
                f"Try `search_code(query='{concept}')` for raw results or ensure the project is indexed."
            )
        else:
            # Synthesize narrative with LLM
            narrative = ""
            try:
                context_str = "\n".join(context_parts)
                prompt = (
                    f"You are explaining how a codebase implements a concept.\n\n"
                    f"{context_str}\n\n"
                    f"Provide a clear, concise explanation of how '{concept}' is implemented. "
                    f"Reference specific files and functions. Keep it practical."
                )
                narrative = self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You are a senior developer explaining code architecture concisely.",
                )
            except Exception as e:
                narrative = f"Found {len(suggestions)} relevant code elements across {len(by_module)} modules."
                logger.warning(f"LLM synthesis failed for explain_concept: {e}")

        results = [
            {
                "file": str(s.file),
                "line": s.line,
                "name": s.name,
                "type": s.element_type,
                "similarity": round(s.similarity, 3),
                "confidence_tier": s.confidence_tier,
                "code_preview": s.code[:200] if s.code else "",
            }
            for s in suggestions
        ]

        avg_sim = sum(s.similarity for s in suggestions) / len(suggestions)
        tier = "high" if avg_sim > 0.6 else "medium" if avg_sim > 0.3 else "low"

        return self._build_envelope(
            results=results,
            confidence_tier=tier,
            confidence_score=round(avg_sim, 3),
            narrative=narrative,
            suggested_followup=[
                f"trace_flow(element_name='{suggestions[0].name}')"
                if suggestions[0].name
                else "",
                f"find_examples(pattern='{concept}')",
            ],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["vector_store", "memory", "llm"],
        )

    async def _tool_suggest_approach(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest implementation approach for a task."""
        start = time.time()
        task = args.get("task", "")
        target_files = args.get("target_files", [])
        include_cross_project = args.get("include_cross_project", False)

        if not task:
            return {"error": "task is required"}

        result_data = {
            "task": task,
            "relevant_code": [],
            "patterns": [],
            "suggested_files": [],
            "security_notes": [],
            "approach": "",
            "cross_project_patterns": [],
        }

        # 1. Get implementation context
        try:
            from codesage.core.context_provider import ContextProvider

            provider = ContextProvider(self.config)
            context = provider.get_implementation_context(
                task_description=task,
                target_files=target_files if target_files else None,
                include_cross_project=include_cross_project,
            )
            impl_data = context.to_dict()

            result_data["relevant_code"] = impl_data.get("relevant_code", [])
            result_data["suggested_files"] = impl_data.get("suggested_files", [])
            result_data["security_notes"] = impl_data.get("security", {}).get(
                "requirements", []
            )

            if impl_data.get("implementation_plan", {}).get("steps"):
                result_data["approach"] = "\n".join(
                    f"{i + 1}. {step}"
                    for i, step in enumerate(impl_data["implementation_plan"]["steps"])
                )

        except Exception as e:
            logger.warning(f"Context provider failed, using basic search: {e}")
            # Fallback to basic search
            suggestions = self.suggester.find_similar(
                query=task,
                limit=5,
                min_similarity=0.2,
            )
            result_data["relevant_code"] = [
                {
                    "file": str(s.file),
                    "name": s.name,
                    "type": s.element_type,
                    "similarity": round(s.similarity, 2),
                }
                for s in suggestions
            ]

        # 2. Get patterns from memory
        try:
            patterns = self.memory.find_similar_patterns(task, limit=5)
            # Filter out Python-specific patterns when project language is not Python
            suggest_lang = getattr(self.config, "language", "python").lower()
            all_patterns = [
                {
                    "name": p.get("name", "?"),
                    "description": p.get("description", "")[:150],
                    "confidence": p.get("confidence", 0),
                }
                for p in patterns
            ]
            if suggest_lang != "python":
                # Filter Python-specific patterns; empty is better than wrong-language patterns
                result_data["patterns"] = [
                    p for p in all_patterns
                    if p["name"] not in _PYTHON_SPECIFIC_PATTERNS
                ]
            else:
                result_data["patterns"] = all_patterns
        except Exception:
            pass

        # 3. Cross-project recommendations
        if include_cross_project:
            try:
                from codesage.memory.pattern_miner import PatternMiner

                miner = PatternMiner(self.memory)
                cross_patterns = miner.recommend_patterns(
                    project_name=self.config.project_name,
                    limit=5,
                )
                result_data["cross_project_patterns"] = cross_patterns
            except Exception:
                pass

        # Synthesize narrative
        narrative = ""
        try:
            code_summary = ", ".join(
                r.get("name", r.get("file", "?"))
                for r in result_data["relevant_code"][:3]
            )
            suggest_primary_lang = getattr(self.config, "language", "python")
            lang_context = (
                f"This is a {suggest_primary_lang} project. "
                if suggest_primary_lang != "python"
                else ""
            )
            prompt = (
                f"{lang_context}Task: {task}\n"
                f"Relevant code: {code_summary}\n"
                f"Approach: {result_data.get('approach', 'No structured approach available')}\n\n"
                f"Write a brief (2-3 sentence) summary of the recommended approach. "
                f"Focus on {suggest_primary_lang} idioms and patterns — do NOT mention Python-specific "
                f"conventions (type hints, docstrings, dunder methods) unless this is a Python project."
            )
            narrative = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=f"You are a helpful senior {suggest_primary_lang} developer giving concise implementation guidance.",
            )
        except Exception:
            narrative = f"Found {len(result_data['relevant_code'])} relevant code elements and {len(result_data['patterns'])} patterns."

        return self._build_envelope(
            results=result_data,
            confidence_tier="medium",
            confidence_score=0.6,
            narrative=narrative,
            suggested_followup=[
                f"explain_concept(concept='{task}')",
                f"search_code(query='{task}')",
            ],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["context_provider", "memory", "llm"],
        )

    async def _tool_trace_flow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Trace call flow for a code element with two-phase lookup."""
        start = time.time()
        element_name = args.get("element_name", "")
        direction = args.get("direction", "both")
        max_depth = args.get("max_depth", 3)

        if not element_name:
            return {"error": "element_name is required"}

        # PHASE 1: Try exact name match first for precision
        exact_matches = self.db.find_elements_by_name(element_name, limit=5)

        if exact_matches:
            # Use first exact match
            element = exact_matches[0]

            # Build suggestion-like object
            class ExactMatchTarget:
                def __init__(self, element):
                    self.name = element.name
                    self.file = element.file
                    self.line = element.line_start
                    self.element_type = element.type
                    self.callers = []
                    self.callees = []
                    self.dependencies = []
                    self.dependents = []
                    self.id = element.id

            target = ExactMatchTarget(element)
            logger.info(
                f"Trace flow: Found exact match for '{element_name}' in {target.file}"
            )
        else:
            # PHASE 2: Fall back to semantic search
            suggestions = self.suggester.find_similar(
                query=element_name,
                limit=3,
                min_similarity=0.25,
                include_explanations=False,
                include_graph_context=True,
            )

            if not suggestions:
                return self._build_envelope(
                    results={
                        "element": element_name,
                        "callers": [],
                        "callees": [],
                        "chains": [],
                    },
                    confidence_tier="low",
                    confidence_score=0.1,
                    narrative=f"Could not find element '{element_name}' in the codebase.",
                    suggested_followup=[f"search_code(query='{element_name}')"],
                    search_time_ms=(time.time() - start) * 1000,
                )

            target = suggestions[0]
            logger.info(
                f"Trace flow: Using semantic search for '{element_name}' -> '{target.name}'"
            )
        trace_result = {
            "element": target.name or element_name,
            "file": str(target.file),
            "line": target.line,
            "type": target.element_type,
            "callers": [],
            "callees": [],
            "chains": [],
        }

        # Track whether the element was found in the graph at all
        _node_ids_found = False

        # Use graph store for tracing
        try:
            from codesage.storage.manager import StorageManager

            storage = StorageManager(self.config)

            if storage.graph_store:
                # Resolve element name to graph node ID (hex hash)
                name_to_find = target.name or element_name
                nodes = storage.graph_store.find_nodes_by_name(name_to_find)
                node_ids = [n["id"] for n in nodes] if nodes else []
                _node_ids_found = bool(node_ids)

                if not node_ids:
                    logger.info(f"No graph nodes found for name '{name_to_find}'")

                for node_id in node_ids[
                    :3
                ]:  # Check up to 3 matches (same name in different files)
                    if direction in ("callers", "both"):
                        callers = storage.graph_store.get_callers(node_id)
                        trace_result["callers"].extend(
                            [
                                {
                                    "name": c.get("name", "?"),
                                    "file": c.get("file", "?"),
                                    "call_line": c.get("call_line", 0),
                                }
                                for c in callers[:20]
                            ]
                        )

                        # Transitive callers
                        if max_depth > 1:
                            try:
                                transitive = storage.graph_store.get_transitive_callers(
                                    node_id, max_depth=max_depth
                                )
                                trace_result["chains"].extend(
                                    [
                                        {
                                            "direction": "caller_chain",
                                            "name": t.get("name", "?"),
                                            "file": t.get("file", "?"),
                                        }
                                        for t in transitive[:10]
                                    ]
                                )
                            except Exception:
                                pass

                    if direction in ("callees", "both"):
                        callees = storage.graph_store.get_callees(node_id)
                        trace_result["callees"].extend(
                            [
                                {
                                    "name": c.get("name", "?"),
                                    "file": c.get("file", "?"),
                                    "call_line": c.get("call_line", 0),
                                }
                                for c in callees[:20]
                            ]
                        )

                # Deduplicate by name+file
                seen_callers = set()
                unique_callers = []
                for c in trace_result["callers"]:
                    key = (c["name"], c["file"])
                    if key not in seen_callers:
                        seen_callers.add(key)
                        unique_callers.append(c)
                trace_result["callers"] = unique_callers

                seen_callees = set()
                unique_callees = []
                for c in trace_result["callees"]:
                    key = (c["name"], c["file"])
                    if key not in seen_callees:
                        seen_callees.add(key)
                        unique_callees.append(c)
                trace_result["callees"] = unique_callees

        except Exception as e:
            logger.warning(f"Graph tracing failed: {e}")
            # Fall back to basic caller/callee info from suggestion
            trace_result["callers"] = target.callers[:10] if target.callers else []
            trace_result["callees"] = target.callees[:10] if target.callees else []

        total_connections = len(trace_result["callers"]) + len(trace_result["callees"])
        tier = (
            "high"
            if total_connections > 5
            else "medium"
            if total_connections > 0
            else "low"
        )

        # Build rich narrative
        callers_list = trace_result["callers"]
        callees_list = trace_result["callees"]
        # Detect if the project likely has no graph relationships for its language
        primary_lang = getattr(self.config, "language", "python").lower()
        _graph_coverage_note = ""
        if _node_ids_found and total_connections == 0 and primary_lang != "python":
            _graph_coverage_note = (
                f" (Note: call-graph relationships for {primary_lang} are not yet indexed — "
                "graph tracing currently covers Python only; run 'codesage index' once "
                "Rust/Go/JS support is added.)"
            )
        elif not _node_ids_found and total_connections == 0:
            _graph_coverage_note = " (element not found in call graph)"

        narrative_parts = [
            f"'{trace_result['element']}' in {trace_result.get('file', '?')} "
            f"({trace_result.get('type', 'element')}).\n"
        ]
        if callers_list:
            caller_names = ", ".join(
                f"'{c['name']}'" for c in callers_list[:5]
            )
            narrative_parts.append(f"Called by: {caller_names}.\n")
        else:
            narrative_parts.append(
                f"No callers found{_graph_coverage_note or ' (may be an entry point)'}.\n"
            )
        if callees_list:
            callee_names = ", ".join(
                f"'{c['name']}'" for c in callees_list[:5]
            )
            narrative_parts.append(f"Calls: {callee_names}.\n")
        else:
            narrative_parts.append(
                f"No callees found{_graph_coverage_note or ' in graph'}.\n"
            )
        if trace_result.get("chains"):
            narrative_parts.append(
                f"Transitive chain depth: {len(trace_result['chains'])} nodes."
            )

        return self._build_envelope(
            results=trace_result,
            confidence_tier=tier,
            confidence_score=0.7 if total_connections > 0 else 0.3,
            narrative="".join(narrative_parts),
            suggested_followup=[
                f"get_file_context(file_path='{target.file}')",
                f"explain_concept(concept='{element_name}')",
            ],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["vector_store", "graph_store"],
        )

    async def _tool_find_examples(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find usage examples grouped by pattern variation."""
        start = time.time()
        pattern = args.get("pattern", "")
        limit = args.get("limit", 10)

        if not pattern:
            return {"error": "pattern is required"}

        # Broad search
        suggestions = self.suggester.find_similar(
            query=pattern,
            limit=limit * 2,
            min_similarity=0.15,
            include_explanations=False,
            include_graph_context=False,
        )

        if not suggestions:
            return self._build_envelope(
                results=[],
                confidence_tier="low",
                confidence_score=0.1,
                narrative=f"No examples found for '{pattern}'.",
                suggested_followup=[f"search_code(query='{pattern}')"],
                search_time_ms=(time.time() - start) * 1000,
            )

        # Group by directory
        by_dir: Dict[str, list] = {}
        for s in suggestions[:limit]:
            dir_name = str(Path(str(s.file)).parent)
            by_dir.setdefault(dir_name, []).append(s)

        groups = []
        for dir_name, items in by_dir.items():
            group = {
                "directory": dir_name,
                "count": len(items),
                "examples": [
                    {
                        "file": str(s.file),
                        "line": s.line,
                        "name": s.name,
                        "type": s.element_type,
                        "similarity": round(s.similarity, 3),
                        "code_preview": s.code[:200] if s.code else "",
                    }
                    for s in items[:3]
                ],
            }
            groups.append(group)

        # LLM explanation of groups
        narrative = ""
        try:
            group_summary = "\n".join(
                f"- {g['directory']}: {g['count']} examples ({', '.join(e['name'] or '?' for e in g['examples'][:2])})"
                for g in groups[:5]
            )
            prompt = (
                f"Pattern: {pattern}\n"
                f"Found these groups of examples:\n{group_summary}\n\n"
                f"Briefly explain what each group demonstrates about the pattern."
            )
            narrative = self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a code analyst explaining pattern usage concisely.",
            )
        except Exception:
            narrative = (
                f"Found {len(suggestions)} examples across {len(groups)} directories."
            )

        avg_sim = sum(s.similarity for s in suggestions[:limit]) / min(
            limit, len(suggestions)
        )

        return self._build_envelope(
            results=groups,
            confidence_tier="medium" if avg_sim > 0.3 else "low",
            confidence_score=round(avg_sim, 3),
            narrative=narrative,
            suggested_followup=[
                f"explain_concept(concept='{pattern}')",
                f"recommend_pattern(context='{pattern}')",
            ],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["vector_store", "llm"],
        )

    async def _tool_recommend_pattern(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend patterns from memory and cross-project data."""
        start = time.time()
        context = args.get("context", "")
        limit = args.get("limit", 5)

        if not context:
            return {"error": "context is required"}

        results = []

        # Find patterns from memory
        try:
            patterns = self.memory.find_similar_patterns(context, limit=limit)
            for p in patterns:
                results.append(
                    {
                        "name": p.get("name", "Unknown"),
                        "description": p.get("description", ""),
                        "confidence": p.get("confidence", 0),
                        "category": p.get("category", "general"),
                        "source_project": p.get("project", self.config.project_name),
                        "examples": p.get("examples", [])[:2],
                    }
                )
        except Exception as e:
            logger.warning(f"Pattern lookup failed: {e}")

        # Try cross-project recommendations
        try:
            from codesage.memory.pattern_miner import PatternMiner

            miner = PatternMiner(self.memory)
            cross_patterns = miner.recommend_patterns(
                project_name=self.config.project_name,
                limit=limit,
            )
            for cp in cross_patterns:
                if isinstance(cp, dict):
                    results.append(
                        {
                            "name": cp.get("name", "Unknown"),
                            "description": cp.get("description", ""),
                            "confidence": cp.get("confidence", 0),
                            "category": cp.get("category", "cross-project"),
                            "source_project": cp.get("project", "other"),
                            "examples": cp.get("examples", [])[:2],
                        }
                    )
        except Exception:
            pass

        # Deduplicate by name
        seen = set()
        unique_results = []
        for r in results:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique_results.append(r)
        results = unique_results[:limit]

        # Remove Python-specific patterns when the project's primary language is not Python
        recommend_lang = getattr(self.config, "language", "python").lower()
        if recommend_lang != "python":
            non_python = [r for r in results if r.get("name", "") not in _PYTHON_SPECIFIC_PATTERNS]
            results = non_python  # Use filtered list (may be empty — better than showing wrong language patterns)

        # Filter: if context has architecture keywords but results are only
        # generic style patterns (naming, docstring, type_hints), note the gap.
        _ARCH_KEYWORDS = {
            "async", "cache", "cach", "singleton", "factory", "decorator",
            "pool", "queue", "ttl", "retry", "circuit", "observer", "event",
            "pipeline", "plugin", "middleware", "strategy", "adapter", "proxy",
            "builder", "repository", "service", "dependency", "injection",
        }
        # Generic code style pattern names that are unlikely to match architecture queries
        _STYLE_ONLY_NAMES = _PYTHON_SPECIFIC_PATTERNS | {
            "context_managers", "specific_exceptions", "error_handling",
        }
        arch_note = ""
        context_lower = context.lower()
        if any(kw in context_lower for kw in _ARCH_KEYWORDS):
            naming_cats = {"naming_convention", "naming", "style"}
            # A result is "style-only" if its name is in the style set or its
            # category is a naming/style category
            def _is_style_only(r: dict) -> bool:
                return (
                    r.get("name", "") in _STYLE_ONLY_NAMES
                    or r.get("category", "") in naming_cats
                )

            non_style = [r for r in results if not _is_style_only(r)]
            if not non_style and results:
                arch_note = (
                    "Note: No architecture patterns have been learned yet for this context. "
                    "Run 'codesage index' more and use the tool more to build pattern memory. "
                    "Showing general codebase style patterns as a fallback."
                )
            elif non_style:
                results = non_style

        narrative = ""
        if results:
            names = ", ".join(r["name"] for r in results[:3])
            narrative = f"Found {len(results)} relevant patterns: {names}."
            if arch_note:
                narrative = arch_note + " " + narrative
        else:
            narrative = f"No patterns found for '{context}'. The memory system may need more usage data."

        tier = "high" if len(results) >= 3 else "medium" if results else "low"
        # Downgrade confidence if we're serving a fallback
        if arch_note:
            tier = "low"

        return self._build_envelope(
            results=results,
            confidence_tier=tier,
            confidence_score=round(results[0]["confidence"], 3) if results else 0.0,
            narrative=narrative,
            suggested_followup=[
                f"find_examples(pattern='{results[0]['name']}')"
                if results
                else f"search_code(query='{context}')",
            ],
            search_time_ms=(time.time() - start) * 1000,
            sources_used=["memory", "pattern_miner"],
        )

    # =========================================================================
    # Resources
    # =========================================================================

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
            uri_str = str(uri)
            if uri_str == "codesage://codebase":
                stats = self.db.get_stats()
                return json.dumps(
                    {
                        "project": self.config.project_name,
                        "path": str(self.project_path),
                        "language": self.config.language,
                        "files_indexed": stats.get("files", 0),
                        "code_elements": stats.get("elements", 0),
                        "last_indexed": stats.get("last_indexed"),
                    },
                    indent=2,
                )

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

    # =========================================================================
    # Server Transports
    # =========================================================================

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
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.responses import Response
            from starlette.routing import Mount, Route
            import uvicorn
        except ImportError:
            logger.error(
                "SSE transport requires additional dependencies. "
                "Install with: pip install 'mcp[sse]' or use stdio transport."
            )
            raise

        logger.info(
            f"Starting CodeSage MCP Server (HTTP/SSE transport) on {host}:{port}"
        )
        logger.info(f"Server endpoint: http://{host}:{port}/sse")
        logger.info("Multiple clients can connect simultaneously")

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

        config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


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
