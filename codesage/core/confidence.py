"""Multi-factor confidence scoring for search results."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from codesage.utils.logging import get_logger

logger = get_logger("core.confidence")


@dataclass
class ConfidenceResult:
    """Result of multi-factor confidence scoring."""

    score: float  # 0.0-1.0
    tier: str  # "high", "medium", "low"
    factors: Dict[str, float] = field(default_factory=dict)


class ConfidenceScorer:
    """Multi-factor confidence scoring for search results.

    Combines vector similarity, graph centrality, name overlap,
    pattern matching, and file recency into a single confidence score.
    Missing backends contribute 0.0 (graceful degradation).
    """

    WEIGHTS = {
        "vector_similarity": 0.30,
        "graph_centrality": 0.20,
        "name_overlap": 0.20,
        "pattern_match": 0.15,
        "file_recency": 0.15,
    }

    def __init__(
        self,
        graph_store=None,
        memory_manager=None,
        project_path: Optional[str] = None,
    ):
        """Initialize the confidence scorer.

        Args:
            graph_store: Optional KuzuDB graph store for centrality
            memory_manager: Optional MemoryManager for pattern matching
            project_path: Optional project path for file recency
        """
        self._graph_store = graph_store
        self._memory_manager = memory_manager
        self._project_path = project_path

    def score(self, result: dict, query: str) -> ConfidenceResult:
        """Score a search result with multi-factor confidence.

        Args:
            result: Search result dict with keys like similarity, name, file, etc.
            query: The original search query

        Returns:
            ConfidenceResult with score, tier, and factor breakdown
        """
        factors = {}
        factors["vector_similarity"] = result.get("similarity", 0.0)
        factors["graph_centrality"] = self._calc_graph_centrality(result)
        factors["name_overlap"] = self._calc_name_overlap(result, query)
        factors["pattern_match"] = self._calc_pattern_match(result, query)
        factors["file_recency"] = self._calc_file_recency(result)

        total = sum(factors[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        tier = "high" if total > 0.7 else "medium" if total > 0.4 else "low"
        return ConfidenceResult(score=round(total, 3), tier=tier, factors=factors)

    def score_suggestion(self, suggestion, query: str) -> ConfidenceResult:
        """Score a Suggestion object directly.

        Args:
            suggestion: A Suggestion dataclass instance
            query: The original search query

        Returns:
            ConfidenceResult
        """
        result = {
            "similarity": getattr(suggestion, "similarity", 0.0),
            "name": getattr(suggestion, "name", ""),
            "file": str(getattr(suggestion, "file", "")),
            "callers": getattr(suggestion, "callers", []),
            "callees": getattr(suggestion, "callees", []),
            "dependencies": getattr(suggestion, "dependencies", []),
            "dependents": getattr(suggestion, "dependents", []),
        }
        return self.score(result, query)

    def _calc_graph_centrality(self, result: dict) -> float:
        """Calculate graph centrality score based on connections.

        More callers/callees/dependencies = higher centrality.
        """
        callers = len(result.get("callers", []))
        callees = len(result.get("callees", []))
        deps = len(result.get("dependencies", []))
        dependents = len(result.get("dependents", []))

        total_connections = callers + callees + deps + dependents
        if total_connections == 0:
            # Try graph store if available (get_callers/get_callees take hex hash IDs, not names)
            if self._graph_store and result.get("name"):
                try:
                    nodes = self._graph_store.find_nodes_by_name(result["name"])
                    for node in nodes[:1]:
                        node_id = node.get("id")
                        if node_id:
                            callers_list = self._graph_store.get_callers(node_id)
                            callees_list = self._graph_store.get_callees(node_id)
                            total_connections = len(callers_list) + len(callees_list)
                except Exception:
                    pass

        # Normalize: 10+ connections = 1.0
        return min(1.0, total_connections / 10.0)

    def _calc_name_overlap(self, result: dict, query: str) -> float:
        """Calculate name overlap between result and query."""
        name = (result.get("name") or "").lower()
        if not name:
            return 0.0

        query_lower = query.lower()
        query_terms = set(re.findall(r"\b\w+\b", query_lower))

        if not query_terms:
            return 0.0

        # Exact match
        if name in query_lower or query_lower in name:
            return 1.0

        # Term overlap
        name_terms = set(re.findall(r"\b\w+\b", name))
        if not name_terms:
            return 0.0

        overlap = query_terms & name_terms
        return len(overlap) / max(len(query_terms), len(name_terms))

    def _calc_pattern_match(self, result: dict, query: str) -> float:
        """Calculate pattern match score using memory manager."""
        if not self._memory_manager:
            return 0.0

        try:
            patterns = self._memory_manager.find_similar_patterns(query, limit=3)
            if not patterns:
                return 0.0

            # Check if any pattern relates to this result's name/file
            result_name = (result.get("name") or "").lower()
            result_file = (result.get("file") or "").lower()

            for p in patterns:
                p_name = (p.get("name") or "").lower()
                if p_name and (p_name in result_name or p_name in result_file):
                    return min(1.0, p.get("confidence", 0.5) + 0.3)

            # Partial: patterns found but not directly related
            return 0.3
        except Exception:
            return 0.0

    def _calc_file_recency(self, result: dict) -> float:
        """Calculate file recency score based on modification time."""
        file_path = result.get("file")
        if not file_path or not self._project_path:
            return 0.5  # Neutral when unavailable

        try:
            full_path = os.path.join(self._project_path, file_path)
            if not os.path.exists(full_path):
                return 0.5

            mtime = os.path.getmtime(full_path)
            age_days = (time.time() - mtime) / 86400

            # Recent files (< 7 days) = 1.0, old files (> 365 days) = 0.1
            if age_days < 7:
                return 1.0
            elif age_days < 30:
                return 0.8
            elif age_days < 90:
                return 0.6
            elif age_days < 365:
                return 0.4
            else:
                return 0.1
        except Exception:
            return 0.5
