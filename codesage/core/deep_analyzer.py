"""Deep analyzer with parallel multi-agent analysis.

Provides comprehensive code analysis by running semantic, graph,
pattern, and security analysis in parallel with resource management.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from codesage.core.resource_manager import (
    ResourceManager,
    ResourceType,
    get_resource_manager,
)
from codesage.utils.logging import get_logger

if TYPE_CHECKING:
    from codesage.utils.config import Config

logger = get_logger("deep_analyzer")


class AnalysisDepth(str, Enum):
    """Analysis depth levels controlling timeouts and result limits."""

    QUICK = "quick"  # 500ms total, top 3 results
    MEDIUM = "medium"  # 2s total, top 5 results
    THOROUGH = "thorough"  # 5s total, top 10 results + full graph


@dataclass
class DeepAnalysisResult:
    """Result from deep analysis."""

    query: str
    depth: str
    semantic_results: List[Dict[str, Any]] = field(default_factory=list)
    graph_context: Dict[str, Any] = field(default_factory=dict)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    resource_stats: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "depth": self.depth,
            "semantic_results": self.semantic_results,
            "graph_context": self.graph_context,
            "patterns": self.patterns,
            "security_issues": self.security_issues,
            "impact_analysis": self.impact_analysis,
            "recommendations": self.recommendations,
            "risk_score": self.risk_score,
            "errors": self.errors if self.errors else None,
            "resource_stats": self.resource_stats if self.resource_stats else None,
            "execution_time_ms": self.execution_time_ms,
        }


class DeepAnalyzer:
    """Internal multi-agent analyzer for comprehensive code analysis.

    Orchestrates parallel analysis across multiple dimensions:
    - Semantic search for relevant code
    - Graph traversal for relationships and impact
    - Pattern matching from developer memory
    - Security scanning for vulnerabilities

    Not exposed directly to CLI; used via --deep flag on existing commands.
    """

    # Timeout settings per depth level (in seconds)
    TIMEOUTS = {
        AnalysisDepth.QUICK: 0.5,
        AnalysisDepth.MEDIUM: 2.0,
        AnalysisDepth.THOROUGH: 5.0,
    }

    # Result limits per depth level
    LIMITS = {
        AnalysisDepth.QUICK: 3,
        AnalysisDepth.MEDIUM: 5,
        AnalysisDepth.THOROUGH: 10,
    }

    def __init__(
        self,
        config: "Config",
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        """Initialize the deep analyzer.

        Args:
            config: CodeSage configuration.
            resource_manager: Optional resource manager for parallel execution.
        """
        self.config = config
        self._suggester = None
        self._memory = None
        self._scanner = None
        self._storage = None
        self._resource_manager = resource_manager

    @property
    def suggester(self):
        """Lazy-load suggester."""
        if self._suggester is None:
            from codesage.core.suggester import Suggester
            self._suggester = Suggester(self.config)
        return self._suggester

    @property
    def storage(self):
        """Lazy-load storage manager."""
        if self._storage is None:
            self._storage = self.suggester.storage
        return self._storage

    @property
    def memory(self):
        """Lazy-load memory manager."""
        if self._memory is None and self.config.memory.enabled:
            from codesage.memory.memory_manager import MemoryManager
            from codesage.llm.embeddings import EmbeddingService

            embedder = EmbeddingService(
                self.config.llm,
                self.config.cache_dir,
                self.config.performance,
            )
            self._memory = MemoryManager(
                global_dir=self.config.memory.global_dir,
                embedding_fn=embedder.embed_batch,
                vector_dim=embedder.get_dimension(),
            )
        return self._memory

    @property
    def scanner(self):
        """Lazy-load security scanner."""
        if self._scanner is None:
            from codesage.security.scanner import SecurityScanner
            self._scanner = SecurityScanner()
        return self._scanner

    @property
    def resource_manager(self) -> ResourceManager:
        """Get resource manager (lazy-loaded global if not provided)."""
        if self._resource_manager is None:
            self._resource_manager = get_resource_manager()
        return self._resource_manager

    async def analyze(
        self,
        query: str,
        depth: str = "medium",
        target_files: Optional[List[str]] = None,
    ) -> DeepAnalysisResult:
        """Run comprehensive multi-agent analysis with resource management.

        Phase 1: Semantic search (must run first for element IDs)
        Phase 2: Parallel enrichment (graph, patterns, security)

        Uses dedicated thread/process pools with token bucket rate limiting.

        Args:
            query: Search query or task description.
            depth: Analysis depth - "quick", "medium", or "thorough".
            target_files: Optional list of files to focus on.

        Returns:
            DeepAnalysisResult with merged findings and resource stats.
        """
        import time
        start_time = time.monotonic()

        try:
            depth_enum = AnalysisDepth(depth)
        except ValueError:
            depth_enum = AnalysisDepth.MEDIUM

        timeout = self.TIMEOUTS[depth_enum]
        limit = self.LIMITS[depth_enum]

        result = DeepAnalysisResult(query=query, depth=depth)

        try:
            # Phase 1: Semantic search (sequential - needed for element IDs)
            semantic_results = await asyncio.wait_for(
                self._run_semantic(query, limit),
                timeout=timeout,
            )
            result.semantic_results = semantic_results

            # Extract element IDs and files for enrichment
            element_ids = [r.get("id") for r in semantic_results if r.get("id")]
            files = list(set(
                r.get("file") for r in semantic_results if r.get("file")
            ))

            if target_files:
                files = list(set(files + target_files))

            # Phase 2: Parallel enrichment with dedicated resource pools
            tasks = [
                self._run_graph(element_ids, depth_enum),
                self._run_patterns(query, limit),
                self._run_security(files),
            ]

            try:
                enrichment_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )

                graph_result, pattern_result, security_result = enrichment_results

                if not isinstance(graph_result, Exception):
                    result.graph_context = graph_result or {}
                else:
                    result.errors.append(f"Graph analysis error: {graph_result}")

                if not isinstance(pattern_result, Exception):
                    result.patterns = pattern_result or []
                else:
                    result.errors.append(f"Pattern analysis error: {pattern_result}")

                if not isinstance(security_result, Exception):
                    result.security_issues = security_result or []
                else:
                    result.errors.append(f"Security analysis error: {security_result}")

            except asyncio.TimeoutError:
                result.errors.append("Enrichment phase timed out")

        except asyncio.TimeoutError:
            result.errors.append("Semantic search timed out")

        # Record execution time and resource stats
        result.execution_time_ms = round((time.monotonic() - start_time) * 1000, 2)
        result.resource_stats = self.resource_manager.get_stats()

        # Calculate risk score and generate recommendations
        result.risk_score = self._calculate_risk_score(result)
        result.recommendations = self._generate_recommendations(result)
        result.impact_analysis = self._build_impact_analysis(result)

        return result

    def analyze_sync(
        self,
        query: str,
        depth: str = "medium",
        target_files: Optional[List[str]] = None,
    ) -> DeepAnalysisResult:
        """Synchronous wrapper for analyze.

        Args:
            query: Search query or task description.
            depth: Analysis depth - "quick", "medium", or "thorough".
            target_files: Optional list of files to focus on.

        Returns:
            DeepAnalysisResult with merged findings.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.analyze(query, depth, target_files))

    async def _run_semantic(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Run semantic search phase with resource management.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of semantic search results.
        """
        def do_search():
            suggestions = self.suggester.find_similar(
                query=query,
                limit=limit,
                min_similarity=0.2,
                include_explanations=False,
                include_graph_context=False,  # We'll do this in parallel
            )
            return [
                {
                    "id": getattr(s, "id", None) or f"{s.file}:{s.line}:{s.name}",
                    "file": str(s.file),
                    "line": s.line,
                    "name": s.name,
                    "type": s.element_type,
                    "similarity": round(s.similarity, 3),
                    "code": s.code[:400] if len(s.code) > 400 else s.code,
                    "language": s.language,
                }
                for s in suggestions
            ]

        # Use resource manager for embedding operations (semantic search uses embeddings)
        return await self.resource_manager.run(
            ResourceType.EMBEDDING,
            do_search,
            tokens=limit,  # Cost proportional to results
            timeout=5.0,
        )

    async def _run_graph(
        self,
        element_ids: List[str],
        depth: AnalysisDepth,
    ) -> Dict[str, Any]:
        """Run graph analysis phase with resource management.

        Args:
            element_ids: Element IDs to analyze.
            depth: Analysis depth.

        Returns:
            Graph context dictionary.
        """
        if not element_ids or not self.storage.graph_store:
            return {}

        def do_graph_analysis():
            context = {
                "callers": [],
                "callees": [],
                "dependencies": [],
                "dependents": [],
                "impact_scores": {},
            }

            # Only analyze first N elements based on depth
            max_elements = {
                AnalysisDepth.QUICK: 1,
                AnalysisDepth.MEDIUM: 3,
                AnalysisDepth.THOROUGH: len(element_ids),
            }[depth]

            graph = self.storage.graph_store
            for eid in element_ids[:max_elements]:
                callers = graph.get_callers(eid)
                callees = graph.get_callees(eid)
                deps = graph.get_dependencies(eid)
                dependents = graph.get_dependents(eid)

                context["callers"].extend(callers)
                context["callees"].extend(callees)
                context["dependencies"].extend(deps)
                context["dependents"].extend(dependents)

                # Calculate impact score
                if depth == AnalysisDepth.THOROUGH:
                    transitive = graph.get_transitive_callers(eid, max_depth=2)
                    score = min(1.0, (len(callers) + len(transitive)) / 50.0)
                    context["impact_scores"][eid] = round(score, 3)

            # Deduplicate
            seen_ids = set()
            for key in ["callers", "callees", "dependencies", "dependents"]:
                unique = []
                for item in context[key]:
                    item_id = item.get("id", item.get("name"))
                    if item_id and item_id not in seen_ids:
                        seen_ids.add(item_id)
                        unique.append(item)
                context[key] = unique

            return context

        # Use resource manager for graph I/O operations
        return await self.resource_manager.run(
            ResourceType.GRAPH,
            do_graph_analysis,
            tokens=len(element_ids),  # Cost proportional to elements
            timeout=3.0,
        )

    async def _run_patterns(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Run pattern matching phase with resource management.

        Args:
            query: Search query.
            limit: Maximum patterns.

        Returns:
            List of matching patterns.
        """
        if not self.memory:
            return []

        def do_pattern_search():
            patterns = self.memory.find_similar_patterns(query=query, limit=limit)
            # Deduplicate by name
            seen = set()
            unique = []
            for p in patterns:
                name = p.get("name")
                if name and name not in seen:
                    seen.add(name)
                    unique.append({
                        "name": name,
                        "description": p.get("description"),
                        "confidence": p.get("confidence", p.get("confidence_score", 0)),
                        "category": p.get("category"),
                        "example_code": p.get("example_code", "")[:200] if p.get("example_code") else None,
                    })
            return unique

        # Pattern search uses embeddings internally
        return await self.resource_manager.run(
            ResourceType.EMBEDDING,
            do_pattern_search,
            tokens=limit,
            timeout=3.0,
        )

    async def _run_security(
        self,
        files: List[str],
    ) -> List[Dict[str, Any]]:
        """Run security scanning phase with resource management.

        Uses ProcessPoolExecutor for CPU-bound security scanning.

        Args:
            files: Files to scan.

        Returns:
            List of security issues.
        """
        if not files:
            return []

        def do_security_scan():
            issues = []
            for file_str in files[:5]:  # Limit files to scan
                file_path = Path(file_str)
                if file_path.exists():
                    findings = self.scanner.scan_file(file_path)
                    for f in findings:
                        issues.append({
                            "file": str(f.file_path),
                            "line": f.line_number,
                            "severity": f.severity,
                            "rule_id": f.rule_id,
                            "message": f.message,
                            "suggestion": getattr(f.rule, "fix_suggestion", None) if hasattr(f, "rule") else None,
                        })
            return issues

        # Security scanning is CPU-bound - uses dedicated pool
        return await self.resource_manager.run(
            ResourceType.SECURITY,
            do_security_scan,
            tokens=len(files),
            timeout=5.0,
        )

    def _calculate_risk_score(self, result: DeepAnalysisResult) -> float:
        """Calculate overall risk score based on findings.

        Args:
            result: Analysis result to score.

        Returns:
            Risk score between 0 and 1.
        """
        score = 0.0

        # Security issues contribute heavily
        for issue in result.security_issues:
            severity = issue.get("severity", "low")
            if isinstance(severity, str):
                severity = severity.lower()
            if severity in ("critical", "high"):
                score += 0.3
            elif severity == "medium":
                score += 0.15
            else:
                score += 0.05

        # High impact elements add to risk
        for eid, impact in result.graph_context.get("impact_scores", {}).items():
            if impact > 0.7:
                score += 0.1

        # Many dependents increases risk of changes
        dependents = result.graph_context.get("dependents", [])
        if len(dependents) > 10:
            score += 0.2
        elif len(dependents) > 5:
            score += 0.1

        return min(1.0, score)

    def _generate_recommendations(self, result: DeepAnalysisResult) -> List[str]:
        """Generate recommendations based on analysis.

        Args:
            result: Analysis result.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Security recommendations
        if result.security_issues:
            critical = sum(
                1 for i in result.security_issues
                if str(i.get("severity", "")).lower() in ("critical", "high")
            )
            if critical > 0:
                recommendations.append(
                    f"Address {critical} critical/high security issue(s) before proceeding"
                )

        # Impact recommendations
        dependents = result.graph_context.get("dependents", [])
        if len(dependents) > 5:
            recommendations.append(
                f"Changes may affect {len(dependents)} dependent code elements - consider thorough testing"
            )

        # Pattern recommendations
        if result.patterns:
            top_pattern = result.patterns[0]
            conf = top_pattern.get("confidence", 0)
            if conf > 0.7:
                recommendations.append(
                    f"Consider using established pattern: {top_pattern.get('name')}"
                )

        # Default recommendation if none generated
        if not recommendations:
            recommendations.append("No specific recommendations - proceed with standard practices")

        return recommendations

    def _build_impact_analysis(self, result: DeepAnalysisResult) -> Dict[str, Any]:
        """Build impact analysis summary.

        Args:
            result: Analysis result.

        Returns:
            Impact analysis dictionary.
        """
        graph = result.graph_context
        return {
            "blast_radius": {
                "direct_callers": len(graph.get("callers", [])),
                "callees": len(graph.get("callees", [])),
                "dependencies": len(graph.get("dependencies", [])),
                "dependents": len(graph.get("dependents", [])),
            },
            "highest_impact_elements": [
                {"id": eid, "score": score}
                for eid, score in sorted(
                    graph.get("impact_scores", {}).items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
            ],
            "security_summary": {
                "total_issues": len(result.security_issues),
                "critical": sum(
                    1 for i in result.security_issues
                    if str(i.get("severity", "")).lower() == "critical"
                ),
                "high": sum(
                    1 for i in result.security_issues
                    if str(i.get("severity", "")).lower() == "high"
                ),
            },
        }
