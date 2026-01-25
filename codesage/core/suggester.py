"""Code suggestion engine with RAG."""

from typing import List, Optional

from codesage.utils.config import Config
from codesage.models.suggestion import Suggestion
from codesage.llm.provider import LLMProvider
from codesage.llm.embeddings import EmbeddingService
from codesage.llm.prompts import CODE_SUGGESTION_SYSTEM, CODE_SUGGESTION_PROMPT
from codesage.storage.database import Database
from codesage.storage.vector_store import VectorStore


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
        self.db = Database(config.storage.db_path)

        # Initialize embedding service
        self.embedder = EmbeddingService(config.llm, config.cache_dir)

        # Initialize vector store with embedder
        self.vector_store = VectorStore(
            config.storage.chroma_path,
            self.embedder.embedder,
        )

        # Initialize LLM for explanations
        self.llm = LLMProvider(config.llm)

    def find_similar(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.2,
        include_explanations: bool = True,
    ) -> List[Suggestion]:
        """Find similar code based on query.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            include_explanations: Generate LLM explanations for top results

        Returns:
            List of suggestions sorted by similarity
        """
        # Query vector store
        results = self.vector_store.query(
            query_text=query,
            n_results=limit * 2,  # Get extra for filtering
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

            # Build suggestion
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

        return suggestions

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
