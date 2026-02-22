"""Cross-project pattern mining and analysis.

Analyzes patterns across multiple projects to find:
- Common patterns (used in multiple projects)
- Unique patterns (project-specific)
- Pattern evolution over time
- Style consistency metrics
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

from codesage.utils.logging import get_logger

from .memory_manager import MemoryManager
from .models import LearnedPattern, PatternCategory

logger = get_logger("memory.pattern_miner")


class PatternMiner:
    """Mines patterns across projects for insights.

    Provides analysis capabilities:
        - Cross-project pattern detection
        - Pattern evolution tracking
        - Style consistency scoring
        - Pattern recommendation
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        """Initialize the pattern miner.

        Args:
            memory_manager: Memory manager for data access.
        """
        self._memory = memory_manager

    def get_cross_project_patterns(
        self,
        min_projects: int = 2,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Get patterns that appear across multiple projects.

        Args:
            min_projects: Minimum number of projects.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of cross-project patterns with project info.
        """
        try:
            patterns = self._memory.memory_graph.get_cross_project_patterns(min_projects)

            # Filter by confidence
            filtered = [
                p for p in patterns
                if p.get("confidence", 0) >= min_confidence
            ]

            # Sort by project count
            filtered.sort(key=lambda x: x.get("project_count", 0), reverse=True)

            return filtered
        except Exception as e:
            logger.warning(f"Failed to get cross-project patterns: {e}")
            return []

    def get_unique_patterns(
        self,
        project_name: str,
        min_confidence: float = 0.5,
    ) -> List[LearnedPattern]:
        """Get patterns unique to a specific project.

        Args:
            project_name: Project name.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of patterns unique to the project.
        """
        unique = []

        # Get all patterns for the project
        patterns = self._memory.preference_store.get_patterns(
            min_confidence=min_confidence, limit=1000
        )

        for pattern in patterns:
            if project_name in pattern.source_projects:
                # Check if it appears in other projects
                if len(pattern.source_projects) == 1:
                    unique.append(pattern)

        return unique

    def get_pattern_evolution(
        self,
        pattern_name: str,
        days: int = 90,
    ) -> List[Dict[str, Any]]:
        """Track how a pattern has evolved over time.

        Args:
            pattern_name: Pattern name to track.
            days: Number of days to look back.

        Returns:
            List of evolution events.
        """
        cutoff = datetime.now() - timedelta(days=days)
        evolution = []

        # Search for patterns with similar names
        patterns = self._memory.preference_store.get_patterns(limit=1000)

        matching = [p for p in patterns if pattern_name.lower() in p.name.lower()]
        matching.sort(key=lambda x: x.first_seen or datetime.min)

        for i, pattern in enumerate(matching):
            if pattern.first_seen and pattern.first_seen >= cutoff:
                evolution.append({
                    "pattern_id": pattern.id,
                    "pattern_name": pattern.name,
                    "date": pattern.first_seen.isoformat(),
                    "confidence": pattern.confidence_score,
                    "occurrence_count": pattern.occurrence_count,
                    "is_current": i == len(matching) - 1,
                })

        return evolution

    def calculate_style_consistency(
        self,
        project_name: str,
    ) -> Dict[str, Any]:
        """Calculate style consistency score for a project.

        Higher scores indicate more consistent coding style.

        Args:
            project_name: Project name.

        Returns:
            Dictionary with consistency metrics.
        """
        # Get patterns for the project
        patterns = self._memory.preference_store.get_patterns(limit=1000)
        project_patterns = [
            p for p in patterns
            if project_name in p.source_projects
        ]

        if not project_patterns:
            return {
                "score": 0.0,
                "category_scores": {},
                "message": "No patterns found for project",
            }

        # Calculate per-category consistency
        category_scores: Dict[str, Dict[str, Any]] = {}

        for category in PatternCategory:
            cat_patterns = [p for p in project_patterns if p.category == category]
            if cat_patterns:
                avg_confidence = sum(p.confidence_score for p in cat_patterns) / len(cat_patterns)
                total_occurrences = sum(p.occurrence_count for p in cat_patterns)

                # Consistency is higher when fewer patterns explain more occurrences
                coverage = total_occurrences / (len(cat_patterns) * 10)  # Normalize
                category_scores[category.value] = {
                    "pattern_count": len(cat_patterns),
                    "avg_confidence": avg_confidence,
                    "score": min((avg_confidence + coverage) / 2, 1.0),
                }

        # Overall score is weighted average
        if category_scores:
            total_weight = sum(s["pattern_count"] for s in category_scores.values())
            weighted_sum = sum(
                s["score"] * s["pattern_count"]
                for s in category_scores.values()
            )
            overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall = 0.0

        return {
            "score": overall,
            "category_scores": category_scores,
            "total_patterns": len(project_patterns),
            "message": self._get_consistency_message(overall),
        }

    def _get_consistency_message(self, score: float) -> str:
        """Get a human-readable consistency message.

        Args:
            score: Consistency score.

        Returns:
            Message string.
        """
        if score >= 0.8:
            return "Excellent consistency - very predictable coding style"
        elif score >= 0.6:
            return "Good consistency - mostly predictable patterns"
        elif score >= 0.4:
            return "Moderate consistency - some style variations"
        elif score >= 0.2:
            return "Low consistency - significant style variations"
        else:
            return "Very low consistency - highly variable patterns"

    def recommend_patterns(
        self,
        project_name: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend patterns based on cross-project analysis.

        Suggests patterns that:
            - Are commonly used in similar projects
            - Have high confidence across projects
            - Are not yet detected in this project

        Args:
            project_name: Target project.
            limit: Maximum recommendations.

        Returns:
            List of recommended patterns with reasoning.
        """
        recommendations = []

        # Get patterns for this project
        project_pattern_ids = set()
        patterns = self._memory.preference_store.get_patterns(limit=1000)
        for p in patterns:
            if project_name in p.source_projects:
                project_pattern_ids.add(p.id)

        # Get similar projects
        similar = self._memory.find_similar_projects(project_name, min_similarity=0.3)

        if not similar:
            # Fall back to cross-project patterns
            cross = self.get_cross_project_patterns(min_projects=3)
            for cp in cross[:limit]:
                if cp.get("id") not in project_pattern_ids:
                    recommendations.append({
                        "pattern_id": cp.get("id"),
                        "pattern_name": cp.get("name"),
                        "category": cp.get("category"),
                        "confidence": cp.get("confidence", 0),
                        "reason": f"Used in {cp.get('project_count', 0)} other projects",
                        "source": "cross_project",
                    })
            return recommendations

        # Collect patterns from similar projects
        similar_patterns: Dict[str, Dict[str, Any]] = {}

        for sim_project in similar:
            sim_name = sim_project.get("name", "")
            similarity = sim_project.get("similarity", 0)

            for p in patterns:
                if sim_name in p.source_projects:
                    if p.id not in project_pattern_ids:
                        if p.id not in similar_patterns:
                            similar_patterns[p.id] = {
                                "pattern": p,
                                "similar_projects": [],
                                "total_similarity": 0,
                            }
                        similar_patterns[p.id]["similar_projects"].append(sim_name)
                        similar_patterns[p.id]["total_similarity"] += similarity

        # Score and sort recommendations
        scored = []
        for pattern_id, data in similar_patterns.items():
            pattern = data["pattern"]
            score = (
                pattern.confidence_score * 0.3 +
                len(data["similar_projects"]) * 0.3 +
                data["total_similarity"] * 0.4
            )
            scored.append({
                "pattern_id": pattern_id,
                "pattern_name": pattern.name,
                "category": pattern.category.value,
                "confidence": pattern.confidence_score,
                "reason": f"Used in similar projects: {', '.join(data['similar_projects'][:3])}",
                "score": score,
                "source": "similar_projects",
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def get_pattern_clusters(
        self,
        min_correlation: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Get clusters of co-occurring patterns.

        Args:
            min_correlation: Minimum correlation threshold.

        Returns:
            List of pattern clusters.
        """
        clusters: List[Dict[str, Any]] = []
        visited: set = set()

        patterns = self._memory.preference_store.get_patterns(limit=1000)

        for pattern in patterns:
            if pattern.id in visited:
                continue

            # Get co-occurring patterns
            try:
                cooccurring = self._memory.memory_graph.get_cooccurring_patterns(
                    pattern.id, min_correlation
                )
            except Exception:
                continue

            if cooccurring:
                cluster = {
                    "root": pattern.name,
                    "root_id": pattern.id,
                    "root_category": pattern.category.value,
                    "members": [
                        {
                            "id": p.get("id"),
                            "name": p.get("name"),
                            "correlation": p.get("correlation"),
                        }
                        for p in cooccurring
                    ],
                    "size": len(cooccurring) + 1,
                }
                clusters.append(cluster)
                visited.add(pattern.id)
                visited.update(p.get("id") for p in cooccurring)

        # Sort by cluster size
        clusters.sort(key=lambda x: x["size"], reverse=True)
        return clusters

    def get_category_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by pattern category.

        Returns:
            Dictionary with per-category statistics.
        """
        summary: Dict[str, Dict[str, Any]] = {}

        for category in PatternCategory:
            patterns = self._memory.preference_store.get_patterns(
                category=category, limit=1000
            )

            if not patterns:
                continue

            # Calculate stats
            total_occurrences = sum(p.occurrence_count for p in patterns)
            avg_confidence = sum(p.confidence_score for p in patterns) / len(patterns)

            # Get top patterns
            top = sorted(patterns, key=lambda x: x.occurrence_count, reverse=True)[:3]

            summary[category.value] = {
                "pattern_count": len(patterns),
                "total_occurrences": total_occurrences,
                "avg_confidence": avg_confidence,
                "top_patterns": [
                    {
                        "name": p.name,
                        "occurrences": p.occurrence_count,
                        "confidence": p.confidence_score,
                    }
                    for p in top
                ],
            }

        return summary

    def export_analysis_report(self) -> Dict[str, Any]:
        """Export a comprehensive analysis report.

        Returns:
            Dictionary with full analysis data.
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "category_summary": self.get_category_summary(),
            "cross_project_patterns": self.get_cross_project_patterns(min_projects=2),
            "pattern_clusters": self.get_pattern_clusters(min_correlation=0.5),
            "learning_stats": self._memory.get_metrics(),
        }

    def analyze_all_projects(self) -> Dict[str, Any]:
        """Analyze patterns across all tracked projects.

        Performs comprehensive cross-project analysis including:
            - Cross-project pattern detection
            - Pattern clustering
            - Recommendations for each project

        Returns:
            Dictionary with analysis results.
        """
        projects = self._memory.preference_store.get_all_projects()
        patterns = self._memory.preference_store.get_patterns(limit=10000)

        # Cross-project patterns
        cross_patterns = self.get_cross_project_patterns(min_projects=2)

        # Enhance cross-project patterns with average confidence
        for cp in cross_patterns:
            matching = [
                p for p in patterns
                if p.name == cp.get("name")
            ]
            if matching:
                cp["avg_confidence"] = sum(p.confidence_score for p in matching) / len(matching)

        # Pattern clusters
        clusters = self.get_pattern_clusters(min_correlation=0.4)

        # Format clusters for output
        formatted_clusters = []
        for cluster in clusters:
            formatted_clusters.append({
                "root_pattern": cluster.get("root"),
                "patterns": [
                    {"id": m.get("id"), "name": m.get("name"), "correlation": m.get("correlation")}
                    for m in cluster.get("members", [])
                ],
                "size": cluster.get("size", 0),
            })

        # Generate recommendations for each project
        recommendations = []
        for project in projects[:5]:  # Limit to top 5 projects
            project_recs = self.recommend_patterns(project.name, limit=2)
            for rec in project_recs:
                rec["target_project"] = project.name
                rec["source_project"] = rec.get("reason", "").split(": ")[-1] if rec.get("reason") else None
                rec["pattern"] = {
                    "name": rec.get("pattern_name"),
                    "category": rec.get("category"),
                    "confidence": rec.get("confidence"),
                }
                recommendations.append(rec)

        # Deduplicate recommendations
        seen_patterns = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec.get("pattern_id") not in seen_patterns:
                seen_patterns.add(rec.get("pattern_id"))
                unique_recommendations.append(rec)

        return {
            "project_count": len(projects),
            "total_patterns": len(patterns),
            "cross_project_patterns": cross_patterns,
            "pattern_clusters": formatted_clusters,
            "recommendations": unique_recommendations[:10],
            "category_summary": self.get_category_summary(),
            "generated_at": datetime.now().isoformat(),
        }
