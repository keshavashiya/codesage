"""Code context builder for chat interface.

Retrieves relevant code from the codebase based on user queries
and formats it for inclusion in LLM prompts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from codesage.core.suggester import Suggester
from codesage.utils.config import Config
from codesage.utils.logging import get_logger

from .prompts import CODE_BLOCK_TEMPLATE, CONTEXT_TEMPLATE, SHORT_CONTEXT_TEMPLATE

logger = get_logger("chat.context")


class CodeContextBuilder:
    """Builds code context for chat conversations.

    Retrieves relevant code snippets from the indexed codebase
    and formats them for inclusion in LLM prompts.
    """

    # Maximum characters for context to avoid token limits
    MAX_CONTEXT_CHARS = 4000
    MAX_CODE_CHARS = 500

    def __init__(self, config: Config):
        """Initialize the context builder.

        Args:
            config: CodeSage configuration
        """
        self.config = config
        self._suggester: Optional[Suggester] = None

    @property
    def suggester(self) -> Suggester:
        """Lazy-load the suggester."""
        if self._suggester is None:
            self._suggester = Suggester(self.config)
        return self._suggester

    def build_context(
        self,
        query: str,
        limit: int = 3,
        min_similarity: float = 0.3,
        max_chars: Optional[int] = None,
    ) -> str:
        """Build context string from relevant code.

        Args:
            query: User query to find relevant code for
            limit: Maximum number of code snippets
            min_similarity: Minimum similarity threshold
            max_chars: Maximum characters for context

        Returns:
            Formatted context string with code blocks
        """
        max_chars = max_chars or self.MAX_CONTEXT_CHARS

        try:
            suggestions = self.suggester.find_similar(
                query=query,
                limit=limit,
                min_similarity=min_similarity,
                include_explanations=False,
            )
        except Exception as e:
            logger.warning(f"Failed to get code context: {e}")
            return ""

        if not suggestions:
            return ""

        # Build code blocks
        code_blocks = []
        total_chars = 0

        for suggestion in suggestions:
            # Truncate code if needed
            code = suggestion.code
            if len(code) > self.MAX_CODE_CHARS:
                code = code[: self.MAX_CODE_CHARS] + "\n... [truncated]"

            # Format graph context
            graph_info = []

            if hasattr(suggestion, "callers") and suggestion.callers:
                caller_names = [c.get("name", "?") for c in suggestion.callers[:3]]
                graph_info.append(f"Called by: {', '.join(caller_names)}")

            if hasattr(suggestion, "callees") and suggestion.callees:
                callee_names = [c.get("name", "?") for c in suggestion.callees[:3]]
                graph_info.append(f"Calls: {', '.join(callee_names)}")

            if hasattr(suggestion, "superclasses") and suggestion.superclasses:
                parents = [p.get("name", "?") for p in suggestion.superclasses]
                graph_info.append(f"Inherits: {', '.join(parents)}")

            graph_context = ""
            if graph_info:
                graph_context = "\nGraph: " + " | ".join(graph_info) + "\n"

            block = CODE_BLOCK_TEMPLATE.format(
                file_path=suggestion.file,
                line_start=suggestion.line,
                element_type=suggestion.element_type,
                name=suggestion.name or "anonymous",
                similarity=suggestion.similarity,
                language=suggestion.language,
                graph_context=graph_context,
                code=code,
            )

            # Check if adding this block would exceed limit
            if total_chars + len(block) > max_chars:
                break

            code_blocks.append(block)
            total_chars += len(block)

        if not code_blocks:
            return ""

        return CONTEXT_TEMPLATE.format(code_blocks="\n".join(code_blocks))

    def build_short_context(
        self,
        query: str,
        limit: int = 2,
        min_similarity: float = 0.4,
    ) -> str:
        """Build a shorter context for limited space.

        Args:
            query: User query
            limit: Maximum snippets
            min_similarity: Minimum similarity

        Returns:
            Short formatted context string
        """
        try:
            suggestions = self.suggester.find_similar(
                query=query,
                limit=limit,
                min_similarity=min_similarity,
                include_explanations=False,
            )
        except Exception as e:
            logger.warning(f"Failed to get short context: {e}")
            return ""

        if not suggestions:
            return ""

        blocks = []
        for s in suggestions:
            # Very short code preview
            code_lines = s.code.split("\n")[:5]
            code_preview = "\n".join(code_lines)
            if len(s.code.split("\n")) > 5:
                code_preview += "\n..."

            blocks.append(
                f"**{s.file}:{s.line}** ({s.similarity:.0%})\n```{s.language}\n{code_preview}\n```"
            )

        return SHORT_CONTEXT_TEMPLATE.format(code_blocks="\n\n".join(blocks))

    def get_code_refs(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get code references for a query.

        Args:
            query: Search query
            limit: Maximum references

        Returns:
            List of code reference dictionaries
        """
        try:
            suggestions = self.suggester.find_similar(
                query=query,
                limit=limit,
                include_explanations=False,
            )
        except Exception as e:
            logger.warning(f"Failed to get code refs: {e}")
            return []

        return [
            {
                "file": str(s.file),
                "line": s.line,
                "name": s.name,
                "type": s.element_type,
                "similarity": s.similarity,
            }
            for s in suggestions
        ]

    def search_code(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Search for code and return detailed results.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity

        Returns:
            List of search result dictionaries
        """
        try:
            suggestions = self.suggester.find_similar(
                query=query,
                limit=limit,
                min_similarity=min_similarity,
                include_explanations=True,
            )
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return []

        return [
            {
                "file": str(s.file),
                "line": s.line,
                "name": s.name,
                "type": s.element_type,
                "language": s.language,
                "similarity": s.similarity,
                "code": s.code,
                "explanation": s.explanation,
            }
            for s in suggestions
        ]
