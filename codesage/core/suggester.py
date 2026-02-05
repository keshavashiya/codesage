"""Code suggestion engine with RAG."""

from typing import Any, Dict, List, Optional

from codesage.utils.config import Config
from codesage.utils.logging import get_logger
from codesage.models.suggestion import Suggestion
from codesage.llm.provider import LLMProvider
from codesage.llm.embeddings import EmbeddingService
from codesage.llm.prompts import CODE_SUGGESTION_SYSTEM, CODE_SUGGESTION_PROMPT
from codesage.storage.manager import StorageManager
from codesage.memory.hooks import MemoryHooks
from codesage.memory.memory_manager import MemoryManager

logger = get_logger("suggester")


class Suggester:
    """Provides intelligent code suggestions using RAG.

    Uses semantic search to find similar code and LLM
    to generate explanations.
    """

    def __init__(self, config: Config):
        """Initialize the suggester.

        Args:
            config: CodeSage configuration
        """
        self.config = config

        # Initialize embedding service
        self.embedder = EmbeddingService(
            config.llm,
            config.cache_dir,
            config.performance,
        )

        # Initialize unified storage manager (uses LanceDB for vectors)
        self.storage = StorageManager(
            config=config,
            embedding_fn=self.embedder,
        )

        # Legacy compatibility
        self.db = self.storage.db
        self.vector_store = self.storage.vector_store

        # Initialize LLM for explanations
        self.llm = LLMProvider(config.llm)

        # Initialize memory system for pattern-aware suggestions
        self._memory_manager: Optional[MemoryManager] = None
        self._memory_hooks: Optional[MemoryHooks] = None
        if config.memory.enabled:
            self._memory_manager = MemoryManager(
                global_dir=config.memory.global_dir,
                embedding_fn=self.embedder.embed_batch,
            )
            self._memory_hooks = MemoryHooks(
                memory_manager=self._memory_manager,
                embedding_fn=self.embedder.embed_batch,
                enabled=True,
            )
            logger.debug("Memory system enabled for suggester")

    def find_similar(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.2,
        include_explanations: bool = True,
        include_graph_context: bool = True,
    ) -> List[Suggestion]:
        """Find similar code based on query.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            include_explanations: Generate LLM explanations for top results
            include_graph_context: Include caller/callee info from graph

        Returns:
            List of suggestions sorted by similarity
        """
        # Query with context (includes graph relationships)
        results = self.storage.query_with_context(
            query_text=query,
            n_results=limit * 2,  # Get extra for filtering
            include_graph=include_graph_context,
        )

        suggestions = []

        for result in results:
            similarity = result.get("similarity", 0)

            # Skip low similarity results
            if similarity < min_similarity:
                continue

            metadata = result.get("metadata", {})
            element_id = metadata.get("id", "")

            # Get full element from database
            element = self.db.get_element(element_id) if element_id else None

            # Build suggestion with graph context
            suggestion = Suggestion(
                file=element.file if element else metadata.get("file", "unknown"),
                line=element.line_start if element else metadata.get("line_start", 0),
                code=element.code if element else result.get("document", ""),
                similarity=similarity,
                language=element.language if element else metadata.get("language", "python"),
                element_type=element.type if element else metadata.get("type", "unknown"),
                name=element.name if element else metadata.get("name"),
                docstring=element.docstring if element else None,
            )

            # Add graph context if available
            if include_graph_context:
                suggestion.callers = result.get("callers", [])
                suggestion.callees = result.get("callees", [])
                if metadata.get("type") == "class":
                    suggestion.superclasses = result.get("superclasses", [])
                    suggestion.subclasses = result.get("subclasses", [])
                suggestion.dependencies = result.get("dependencies", [])
                suggestion.dependents = result.get("dependents", [])
                suggestion.impact_score = result.get("impact_score")

            # Generate explanation for top 3 results when requested
            if include_explanations and len(suggestions) < 3:
                suggestion.explanation = self._explain_match(
                    query,
                    suggestion.code,
                    suggestion.language,
                    similarity,
                )

            suggestions.append(suggestion)

            if len(suggestions) >= limit:
                break

        # Record interaction for learning
        if self._memory_hooks and suggestions:
            self._memory_hooks.on_query(
                query=query,
                project_name=self.config.project_name,
                results=suggestions,
            )

        return suggestions

    def find_similar_with_patterns(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.2,
        include_explanations: bool = True,
        include_cross_project: bool = False,
    ) -> Dict[str, Any]:
        """Find similar code and enrich with learned patterns.

        Combines semantic code search with pattern matching from
        the developer memory system.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            include_explanations: Generate LLM explanations
            include_cross_project: Include cross-project recommendations

        Returns:
            Dictionary with:
                - suggestions: List of code suggestions
                - matching_patterns: Patterns that match the query
                - recommendations: Pattern-based recommendations
                - cross_project_recommendations: Patterns from other projects
        """
        result = {
            "suggestions": [],
            "matching_patterns": [],
            "recommendations": [],
            "cross_project_recommendations": [],
        }

        # Get code suggestions
        suggestions = self.find_similar(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            include_explanations=include_explanations,
            include_graph_context=True,
        )
        result["suggestions"] = suggestions

        # Find matching patterns from memory
        if self._memory_manager:
            try:
                patterns = self._memory_manager.find_similar_patterns(
                    query=query,
                    limit=5,
                )
                result["matching_patterns"] = patterns

                # Get pattern recommendations based on query context
                if not suggestions and patterns:
                    # No code matches but patterns exist - suggest adopting patterns
                    result["recommendations"] = [
                        {
                            "type": "adopt_pattern",
                            "pattern": p,
                            "reason": f"Consider using the '{p.get('name', 'unknown')}' pattern",
                        }
                        for p in patterns[:3]
                    ]

                # Record the pattern-aware query
                if self._memory_hooks:
                    self._memory_hooks.on_suggestion(
                        query=query,
                        project_name=self.config.project_name,
                        suggestion=(
                            f"Found {len(suggestions)} matches, {len(patterns)} patterns"
                        ),
                    )

                # Cross-project recommendations (opt-in)
                if (
                    include_cross_project
                    and self.config.features.cross_project_recommendations
                ):
                    from codesage.memory.pattern_miner import PatternMiner

                    miner = PatternMiner(self._memory_manager)
                    result["cross_project_recommendations"] = miner.recommend_patterns(
                        project_name=self.config.project_name,
                        limit=5,
                    )

            except Exception as e:
                logger.warning(f"Failed to get patterns: {e}")

        return result

    def get_pattern_context(self, element_name: str) -> Dict[str, Any]:
        """Get pattern context for a code element.

        Args:
            element_name: Name of the code element

        Returns:
            Dictionary with relevant patterns and style info
        """
        context = {
            "patterns": [],
            "style_recommendations": [],
            "similar_code": [],
        }

        if not self._memory_manager:
            return context

        try:
            # Search for patterns related to this element
            patterns = self._memory_manager.find_similar_patterns(
                query=element_name,
                limit=5,
            )
            context["patterns"] = patterns

            # Get preferred structures
            structures = self._memory_manager.get_preferred_structures(
                min_confidence=0.5
            )
            context["style_recommendations"] = structures

        except Exception as e:
            logger.warning(f"Failed to get pattern context: {e}")

        return context

    def _explain_match(
        self,
        query: str,
        code: str,
        language: str,
        similarity: float,
    ) -> Optional[str]:
        """Generate LLM explanation for why code matches query.

        Args:
            query: User's search query
            code: Matched code
            language: Programming language
            similarity: Similarity score

        Returns:
            Explanation string or None on failure
        """
        try:
            # Truncate code if too long
            code_snippet = code[:500] if len(code) > 500 else code

            prompt = CODE_SUGGESTION_PROMPT.format(
                query=query,
                code=code_snippet,
                language=language,
                similarity=similarity,
            )

            response = self.llm.generate(
                prompt=prompt,
                system_prompt=CODE_SUGGESTION_SYSTEM,
            )

            return response.strip()
        except Exception:
            return None

    def search_by_name(
        self,
        name: str,
        element_type: Optional[str] = None,
    ) -> List[Suggestion]:
        """Search for code elements by name.

        Args:
            name: Name to search for
            element_type: Optional type filter (function, class, method)

        Returns:
            List of matching suggestions
        """
        # Use database search for exact matches
        elements = self.db.get_all_elements()

        suggestions = []

        for element in elements:
            if element.name and name.lower() in element.name.lower():
                if element_type and element.type != element_type:
                    continue

                suggestions.append(Suggestion(
                    file=element.file,
                    line=element.line_start,
                    code=element.code,
                    similarity=1.0,  # Exact name match
                    language=element.language,
                    element_type=element.type,
                    name=element.name,
                    docstring=element.docstring,
                ))

        return suggestions
