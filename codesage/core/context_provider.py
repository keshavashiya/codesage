"""Context provider for implementation guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from codesage.core.suggester import Suggester
from codesage.llm.embeddings import EmbeddingService
from codesage.memory.memory_manager import MemoryManager
from codesage.memory.pattern_miner import PatternMiner
from codesage.models.context import CodeReference, ImplementationContext, ImplementationPlan
from codesage.utils.config import Config
from codesage.utils.logging import get_logger

logger = get_logger("context.provider")


class ContextProvider:
    """Assemble rich context packs for AI IDEs and developers."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._suggester = Suggester(config)
        self._memory: Optional[MemoryManager] = None

        if config.memory.enabled:
            embedder = EmbeddingService(
                config.llm,
                config.cache_dir,
                config.performance,
            )
            self._memory = MemoryManager(
                global_dir=config.memory.global_dir,
                embedding_fn=embedder.embed_batch,
                vector_dim=embedder.get_dimension(),
            )

    def get_implementation_context(
        self,
        task_description: str,
        target_files: Optional[List[str]] = None,
        include_cross_project: bool = False,
    ) -> ImplementationContext:
        """Provide everything needed to implement a task."""
        suggestions = self._suggester.find_similar(
            query=task_description,
            limit=5,
            min_similarity=0.2,
            include_explanations=False,
            include_graph_context=True,
        )

        relevant_code: List[CodeReference] = []
        dependencies: List[Dict[str, Any]] = []
        dependency_ids = set()

        for suggestion in suggestions:
            snippet = suggestion.code
            if len(snippet) > 400:
                snippet = snippet[:400] + "\n... [truncated]"

            graph = {
                "callers": suggestion.callers,
                "callees": suggestion.callees,
                "dependencies": suggestion.dependencies,
                "dependents": suggestion.dependents,
                "impact_score": suggestion.impact_score,
            }

            relevant_code.append(
                CodeReference(
                    file=str(suggestion.file),
                    line=suggestion.line,
                    name=suggestion.name,
                    element_type=suggestion.element_type,
                    similarity=round(suggestion.similarity, 3),
                    snippet=snippet,
                    graph=graph,
                )
            )

            for dep in suggestion.dependencies:
                dep_id = dep.get("id")
                if dep_id and dep_id not in dependency_ids:
                    dependency_ids.add(dep_id)
                    dependencies.append(dep)

        patterns: List[Dict[str, Any]] = []
        preferences: Dict[str, Any] = {}

        if self._memory:
            try:
                raw_patterns = self._memory.find_similar_patterns(
                    query=task_description,
                    limit=5,
                )
                # Deduplicate patterns by name
                seen: set = set()
                patterns = [
                    p for p in raw_patterns
                    if p.get("name") not in seen and not seen.add(p.get("name"))
                ]
                preferences = self._memory.get_all_preferences()
            except Exception as e:
                logger.debug(f"Memory lookup failed: {e}")

        plan = self._generate_plan(
            task_description,
            patterns=patterns,
            suggested_files=target_files or [r.file for r in relevant_code],
        )

        security = self._derive_security_guidance(task_description)

        cross_project: List[Dict[str, Any]] = []
        if (
            include_cross_project
            and self._memory
            and self.config.features.cross_project_recommendations
        ):
            try:
                miner = PatternMiner(self._memory)
                cross_project = miner.recommend_patterns(
                    project_name=self.config.project_name,
                    limit=5,
                )
            except Exception as e:
                logger.debug(f"Cross-project recommendations failed: {e}")

        return ImplementationContext(
            task=task_description,
            implementation_plan=plan,
            relevant_code=relevant_code,
            patterns=patterns,
            dependencies=dependencies,
            preferences=preferences,
            security=security,
            suggested_files=plan.estimated_files,
            cross_project_recommendations=cross_project,
        )

    def to_markdown(self, context: ImplementationContext) -> str:
        """Render context as markdown."""
        lines: List[str] = []
        lines.append(f"# Task: {context.task}")
        lines.append("")

        lines.append("## Implementation Plan")
        for i, step in enumerate(context.implementation_plan.steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        if context.implementation_plan.estimated_files:
            lines.append("## Suggested Files")
            for f in context.implementation_plan.estimated_files:
                lines.append(f"- {f}")
            lines.append("")

        if context.relevant_code:
            lines.append("## Relevant Code")
            for ref in context.relevant_code:
                lines.append(f"### {ref.file}:{ref.line}")
                lines.append(f"- Type: {ref.element_type}")
                if ref.name:
                    lines.append(f"- Name: {ref.name}")
                lines.append(f"- Similarity: {ref.similarity:.2f}")
                lines.append("```")
                lines.append(ref.snippet)
                lines.append("```")
            lines.append("")

        if context.patterns:
            lines.append("## Matching Patterns")
            for p in context.patterns[:5]:
                name = p.get("name", "unknown")
                desc = p.get("description", "")
                conf = p.get("confidence", p.get("confidence_score", 0))
                lines.append(f"- {name} (confidence: {conf:.2f})")
                if desc:
                    lines.append(f"Description: {desc}")
            lines.append("")

        if context.dependencies:
            lines.append("## Dependencies")
            for dep in context.dependencies[:10]:
                name = dep.get("name", "unknown")
                rel = dep.get("rel_type", "depends_on")
                lines.append(f"- {name} ({rel})")
            lines.append("")

        if context.preferences:
            lines.append("## Preferences")
            for key, value in context.preferences.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        if context.security:
            lines.append("## Security Guidance")
            for req in context.security.get("requirements", []):
                lines.append(f"- {req}")
            lines.append("")

        if context.cross_project_recommendations:
            lines.append("## Cross-Project Recommendations")
            for rec in context.cross_project_recommendations:
                name = rec.get("pattern_name", rec.get("pattern_id", "pattern"))
                reason = rec.get("reason", "")
                lines.append(f"- {name}: {reason}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _generate_plan(
        self,
        task_description: str,
        patterns: List[Dict[str, Any]],
        suggested_files: List[str],
    ) -> ImplementationPlan:
        steps = [
            "Review existing relevant code paths and patterns.",
        ]

        if patterns:
            names = [p.get("name", "pattern") for p in patterns[:3]]
            steps.append(f"Apply learned patterns where appropriate: {', '.join(names)}.")

        steps.extend(
            [
                "Implement the change in the target files.",
                "Update or add tests covering the new behavior.",
                "Verify behavior with local checks (lint/test/run).",
            ]
        )

        return ImplementationPlan(
            steps=steps,
            estimated_files=list(dict.fromkeys(suggested_files))[:10],
        )

    def _derive_security_guidance(self, task_description: str) -> Dict[str, Any]:
        text = task_description.lower()
        requirements: List[str] = ["Validate inputs and handle errors clearly."]

        if any(k in text for k in ["auth", "token", "jwt", "login", "password"]):
            requirements.extend(
                [
                    "Validate authentication tokens and expiration.",
                    "Avoid logging sensitive credentials.",
                    "Use secure storage for secrets.",
                ]
            )

        if any(k in text for k in ["db", "sql", "query", "database"]):
            requirements.append("Use parameterized queries to avoid injection.")

        return {"requirements": requirements}
