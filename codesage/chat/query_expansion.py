"""Query expansion system for improving context retrieval accuracy.

This module provides intelligent query understanding by:
1. Classifying user intent
2. Expanding domain-specific terms and synonyms
3. Adding conversation context
4. Detecting and handling ambiguity
5. Calculating confidence scores
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from codesage.utils.logging import get_logger

logger = get_logger("chat.query_expansion")


class UserIntent(Enum):
    """Classification of user query intent."""

    EXPLAIN = auto()  # Understanding code ("how does X work?")
    IMPLEMENT = auto()  # Building new features ("add X functionality")
    DEBUG = auto()  # Fixing issues ("fix", "error", "bug")
    REFACTOR = auto()  # Code improvements ("refactor", "clean up")
    REVIEW = auto()  # Code review ("review", "check")
    SEARCH = auto()  # Finding code ("find", "where is")
    EXPLORE = auto()  # Open-ended ("tell me about", "what is")
    UNKNOWN = auto()  # Unclear intent


@dataclass
class ExpandedQuery:
    """Result of query expansion with all context."""

    original_query: str
    intent: UserIntent
    intent_confidence: float
    expanded_terms: List[str]
    enhanced_query: str
    context_additions: Dict[str, Any]
    is_ambiguous: bool
    ambiguity_hints: List[str]
    confidence_score: float
    suggested_clarifications: List[str]


@dataclass
class ConversationContext:
    """Tracks conversation state for context enhancement."""

    discussed_files: Set[str] = field(default_factory=set)
    recent_topics: List[str] = field(default_factory=list)
    recent_queries: List[str] = field(default_factory=list)
    current_mode: str = "brainstorm"
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    def add_discussion(self, files: List[str], topics: List[str]):
        """Record files and topics from recent interaction."""
        self.discussed_files.update(files)
        self.recent_topics.extend(topics)
        # Keep only last 10 topics
        self.recent_topics = self.recent_topics[-10:]

    def add_query(self, query: str):
        """Record user query for context."""
        self.recent_queries.append(query)
        self.recent_queries = self.recent_queries[-5:]


class QueryExpander:
    """Intelligent query expansion for better context retrieval."""

    # Domain-specific synonyms for software development
    DOMAIN_SYNONYMS: Dict[str, List[str]] = {
        # Authentication & Authorization
        "auth": [
            "authentication",
            "authorization",
            "login",
            "logout",
            "session",
            "token",
            "credential",
            "password",
            "oauth",
            "jwt",
            "identity",
        ],
        "login": ["authentication", "signin", "auth", "session", "credential"],
        "logout": ["signout", "session", "invalidate", "clear"],
        # Data & Storage
        "dto": [
            "data transfer object",
            "model",
            "schema",
            "payload",
            "request",
            "response",
        ],
        "orm": ["object relational mapping", "database", "sql", "model", "entity"],
        "db": ["database", "storage", "persistence", "sql", "query"],
        "cache": ["memoization", "redis", "memcached", "storage", "buffer", "ttl"],
        # API & Web
        "api": ["endpoint", "route", "controller", "handler", "rest", "graphql"],
        "endpoint": ["api", "route", "url", "path", "handler"],
        "middleware": ["interceptor", "filter", "pipeline", "handler"],
        # Testing
        "test": ["unittest", "pytest", "mock", "fixture", "coverage", "assert"],
        "mock": ["stub", "fake", "test double", "unittest.mock", "monkeypatch"],
        # Performance
        "perf": [
            "performance",
            "optimization",
            "speed",
            "latency",
            "throughput",
            "efficiency",
            "fast",
            "slow",
            "bottleneck",
        ],
        "optimize": ["performance", "improve", "efficient", "fast", "speed up"],
        "memory": ["ram", "heap", "leak", "usage", "consumption"],
        # Error Handling
        "error": ["exception", "raise", "catch", "handle", "error handling", "fail"],
        "exception": ["error", "raise", "catch", "try", "except", "finally"],
        "debug": ["debugger", "breakpoint", "trace", "logging", "print", "inspect"],
        # Architecture
        "service": ["business logic", "use case", "interactor", "domain"],
        "repo": ["repository", "data access", "dao", "persistence", "store"],
        "controller": ["handler", "endpoint", "view", "api", "route"],
        "model": ["entity", "domain", "data class", "schema", "orm"],
        # Patterns
        "singleton": ["single instance", "global", "shared", "one instance"],
        "factory": ["creator", "builder", "constructor", "instantiate"],
        "observer": ["pub sub", "event", "listener", "callback", "subscribe"],
        "decorator": ["wrapper", "aspect", "annotation", "attribute"],
    }

    # Intent patterns for classification
    INTENT_PATTERNS: Dict[UserIntent, List[str]] = {
        UserIntent.EXPLAIN: [
            r"how (does|do|is|are|can|should)",
            r"what (is|are|does|do)",
            r"explain",
            r"tell me about",
            r"show me",
            r"understand",
            r"meaning of",
        ],
        UserIntent.IMPLEMENT: [
            r"add\b",
            r"create\b",
            r"implement\b",
            r"build\b",
            r"write\b",
            r"generate\b",
            r"new (feature|function|method|class)",
            r"support\b",
        ],
        UserIntent.DEBUG: [
            r"fix\b",
            r"bug\b",
            r"error\b",
            r"issue\b",
            r"broken\b",
            r"not working",
            r"fails?\b",
            r"crash",
            r"exception",
            r"debug\b",
        ],
        UserIntent.REFACTOR: [
            r"refactor\b",
            r"clean\b",
            r"restructure",
            r"simplify",
            r"improve\b",
            r"better\b",
            r"optimize\b",
            r"rename\b",
            r"extract\b",
            r"move\b",
        ],
        UserIntent.REVIEW: [
            r"review\b",
            r"check\b",
            r"analyze\b",
            r"audit\b",
            r"quality",
            r"security",
            r"vulnerability",
            r"smell",
        ],
        UserIntent.SEARCH: [
            r"find\b",
            r"search\b",
            r"locate\b",
            r"where (is|are)",
            r"look for",
            r"get\b",
            r"show\b",
        ],
        UserIntent.EXPLORE: [
            r"explore\b",
            r"discover",
            r"learn about",
            r"overview",
            r"architecture",
            r"structure",
            r"patterns?\b",
        ],
    }

    # Ambiguous terms that need clarification
    AMBIGUOUS_TERMS: Dict[str, List[str]] = {
        "auth": [
            "Authentication (login/signup)",
            "Authorization (permissions/RBAC)",
            "API authentication (keys/tokens)",
        ],
        "handler": ["HTTP request handler", "Exception handler", "Event handler"],
        "model": ["Data model/ORM", "MVC Model", "AI/ML model"],
        "service": ["Business logic service", "Microservice", "External API service"],
        "controller": ["MVC Controller", "Game controller", "Hardware controller"],
        "client": ["API client", "Frontend/client-side", "Customer/client entity"],
        "worker": ["Background job worker", "Web worker", "Service worker"],
        "queue": ["Message queue", "Task queue", "Job queue"],
        "config": ["Configuration files", "App settings", "Environment config"],
        "manager": ["Resource manager", "State manager", "Connection manager"],
    }

    def __init__(self, llm_provider: Optional[Any] = None):
        """Initialize the query expander.

        Args:
            llm_provider: Optional LLM provider for advanced expansion
        """
        self.llm = llm_provider
        self._build_synonym_index()

    def _build_synonym_index(self):
        """Build reverse index for synonym lookup."""
        self._term_to_synonyms: Dict[str, List[str]] = {}
        for term, synonyms in self.DOMAIN_SYNONYMS.items():
            self._term_to_synonyms[term] = synonyms
            # Also map each synonym back to the root term
            for syn in synonyms:
                if syn not in self._term_to_synonyms:
                    self._term_to_synonyms[syn] = [term] + [
                        s for s in synonyms if s != syn
                    ]

    def expand(
        self, query: str, conversation: Optional[ConversationContext] = None
    ) -> ExpandedQuery:
        """Expand a user query with full context.

        Args:
            query: Original user query
            conversation: Optional conversation context

        Returns:
            ExpandedQuery with all expansion results
        """
        original_query = query.strip()

        # Step 1: Classify intent
        intent, intent_confidence = self._classify_intent(original_query)

        # Step 2: Extract and expand terms
        expanded_terms = self._expand_terms(original_query)

        # Step 3: Check for ambiguity
        is_ambiguous, ambiguity_hints, suggested_clarifications = (
            self._detect_ambiguity(original_query, expanded_terms)
        )

        # Step 4: Enhance with conversation context
        context_additions = {}
        if conversation:
            enhanced_query = self._add_conversation_context(
                original_query, conversation, intent
            )
            context_additions = self._extract_context_additions(conversation)
        else:
            enhanced_query = original_query

        # Step 5: Calculate overall confidence
        confidence_score = self._calculate_confidence(
            original_query, intent_confidence, expanded_terms, is_ambiguous
        )

        return ExpandedQuery(
            original_query=original_query,
            intent=intent,
            intent_confidence=intent_confidence,
            expanded_terms=expanded_terms,
            enhanced_query=enhanced_query,
            context_additions=context_additions,
            is_ambiguous=is_ambiguous,
            ambiguity_hints=ambiguity_hints,
            confidence_score=confidence_score,
            suggested_clarifications=suggested_clarifications,
        )

    def _classify_intent(self, query: str) -> Tuple[UserIntent, float]:
        """Classify the user's intent from their query.

        Args:
            query: User query string

        Returns:
            Tuple of (intent, confidence)
        """
        query_lower = query.lower()
        scores: Dict[UserIntent, int] = {intent: 0 for intent in UserIntent}

        # Pattern matching
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[intent] += 1

        # Command detection
        if query.startswith("/"):
            cmd = query.split()[0].lower()
            if cmd in ["/plan", "/implement"]:
                scores[UserIntent.IMPLEMENT] += 2
            elif cmd in ["/review", "/check"]:
                scores[UserIntent.REVIEW] += 2
            elif cmd in ["/debug", "/fix"]:
                scores[UserIntent.DEBUG] += 2
            elif cmd in ["/search", "/find"]:
                scores[UserIntent.SEARCH] += 2
            elif cmd in ["/explain", "/how"]:
                scores[UserIntent.EXPLAIN] += 2

        # Find highest scoring intent
        max_score = max(scores.values())
        if max_score == 0:
            return UserIntent.UNKNOWN, 0.3

        # Get all intents with max score
        best_intents = [i for i, s in scores.items() if s == max_score]

        if len(best_intents) == 1:
            confidence = min(0.9, 0.5 + (max_score * 0.1))
            return best_intents[0], confidence
        else:
            # Multiple intents with same score - ambiguous
            return UserIntent.UNKNOWN, 0.4

    def _expand_terms(self, query: str) -> List[str]:
        """Expand query with domain-specific synonyms.

        Args:
            query: User query

        Returns:
            List of expanded search terms
        """
        query_lower = query.lower()
        terms = set()

        # Split into words and phrases
        words = re.findall(r"\b\w+\b", query_lower)

        for word in words:
            terms.add(word)

            # Check for exact synonym matches
            if word in self._term_to_synonyms:
                terms.update(self._term_to_synonyms[word])

            # Check for partial matches
            for term, synonyms in self.DOMAIN_SYNONYMS.items():
                if term in word or word in term:
                    terms.add(term)
                    terms.update(synonyms)

        # Remove stopwords and very short terms
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "and",
            "but",
            "or",
            "yet",
            "so",
            "if",
            "because",
            "although",
            "though",
            "while",
            "where",
            "when",
            "that",
            "which",
            "who",
            "whom",
            "whose",
            "what",
            "this",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        }

        terms = {t for t in terms if len(t) > 2 and t not in stopwords}

        return sorted(list(terms))

    def _detect_ambiguity(
        self, query: str, expanded_terms: List[str]
    ) -> Tuple[bool, List[str], List[str]]:
        """Detect if the query is ambiguous and needs clarification.

        Args:
            query: Original query
            expanded_terms: Expanded search terms

        Returns:
            Tuple of (is_ambiguous, ambiguity_hints, suggested_clarifications)
        """
        query_lower = query.lower()
        ambiguous_terms_found = []
        suggestions = []

        for term, options in self.AMBIGUOUS_TERMS.items():
            # Check if term appears in query or expanded terms
            if term in query_lower or term in expanded_terms:
                ambiguous_terms_found.append(term)
                suggestions.extend(options)

        # Also check for vague quantifiers
        vague_patterns = [
            (r"\b(it|this|that|these|those)\b", "unclear reference"),
            (r"\b(all|every|each)\b", "overly broad scope"),
            (r"\b(some|few|many)\b", "imprecise quantity"),
        ]

        for pattern, hint in vague_patterns:
            if re.search(pattern, query_lower):
                ambiguous_terms_found.append(f"vague language ({hint})")

        is_ambiguous = len(ambiguous_terms_found) > 0

        return is_ambiguous, ambiguous_terms_found, suggestions[:6]  # Limit suggestions

    def _add_conversation_context(
        self, query: str, conversation: ConversationContext, intent: UserIntent
    ) -> str:
        """Enhance query with conversation context.

        Args:
            query: Original query
            conversation: Conversation context
            intent: Classified intent

        Returns:
            Enhanced query string
        """
        context_parts = []

        # Add recent topics for continuity
        if conversation.recent_topics:
            recent = conversation.recent_topics[-3:]
            context_parts.append(f"Recent context: {', '.join(recent)}")

        # Add discussed files
        if conversation.discussed_files:
            files = list(conversation.discussed_files)[-3:]
            context_parts.append(f"Relevant files: {', '.join(files)}")

        # Add current mode context
        if conversation.current_mode:
            mode_context = {
                "brainstorm": "exploratory",
                "implement": "implementation-focused",
                "review": "quality-focused",
            }.get(conversation.current_mode, "")
            if mode_context:
                context_parts.append(f"Mode: {mode_context}")

        # Add intent-specific context
        if intent == UserIntent.DEBUG and conversation.recent_queries:
            # For debugging, recent queries might provide context
            last_query = conversation.recent_queries[-1]
            if "error" in last_query.lower() or "bug" in last_query.lower():
                context_parts.append("Related to previous issue")

        if context_parts:
            return f"{query}\n\n[Context: {' | '.join(context_parts)}]"

        return query

    def _extract_context_additions(
        self, conversation: ConversationContext
    ) -> Dict[str, Any]:
        """Extract context additions for search enhancement.

        Args:
            conversation: Conversation context

        Returns:
            Dictionary of context additions
        """
        return {
            "recent_files": list(conversation.discussed_files)[-5:],
            "recent_topics": conversation.recent_topics[-5:],
            "current_mode": conversation.current_mode,
        }

    def _calculate_confidence(
        self,
        query: str,
        intent_confidence: float,
        expanded_terms: List[str],
        is_ambiguous: bool,
    ) -> float:
        """Calculate overall confidence score for the query.

        Args:
            query: Original query
            intent_confidence: Intent classification confidence
            expanded_terms: Number of expanded terms
            is_ambiguous: Whether query is ambiguous

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from intent
        confidence = intent_confidence

        # Boost for query length (longer queries tend to be clearer)
        query_length = len(query.split())
        if query_length >= 5:
            confidence += 0.1
        elif query_length <= 2:
            confidence -= 0.15

        # Boost for expanded terms (more context = better)
        if len(expanded_terms) >= 5:
            confidence += 0.1
        elif len(expanded_terms) <= 2:
            confidence -= 0.1

        # Penalty for ambiguity
        if is_ambiguous:
            confidence -= 0.25

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def get_search_queries(
        self, expanded: ExpandedQuery, max_queries: int = 3
    ) -> List[str]:
        """Generate multiple search queries from expanded query.

        Args:
            expanded: Expanded query result
            max_queries: Maximum number of queries to generate

        Returns:
            List of search query strings
        """
        queries = []

        # Query 1: Original query with context
        if expanded.context_additions:
            context_terms = []
            if expanded.context_additions.get("recent_topics"):
                context_terms.extend(expanded.context_additions["recent_topics"][:2])
            if context_terms:
                queries.append(f"{expanded.original_query} {' '.join(context_terms)}")
            else:
                queries.append(expanded.original_query)
        else:
            queries.append(expanded.original_query)

        # Query 2: Expanded terms
        if len(expanded.expanded_terms) > 2:
            # Combine top expanded terms with original
            top_terms = expanded.expanded_terms[:5]
            queries.append(f"{expanded.original_query} {' '.join(top_terms)}")

        # Query 3: Intent-focused
        if expanded.intent != UserIntent.UNKNOWN:
            intent_keywords = {
                UserIntent.EXPLAIN: "how it works implementation",
                UserIntent.IMPLEMENT: "add create implementation example",
                UserIntent.DEBUG: "fix error exception handling",
                UserIntent.REFACTOR: "improve clean refactor pattern",
                UserIntent.REVIEW: "check analyze review quality",
                UserIntent.SEARCH: "find locate definition",
                UserIntent.EXPLORE: "architecture structure overview",
            }.get(expanded.intent, "")

            if intent_keywords:
                queries.append(f"{expanded.original_query} {intent_keywords}")

        return queries[:max_queries]


# Convenience function for direct usage
def expand_query(
    query: str, conversation: Optional[ConversationContext] = None
) -> ExpandedQuery:
    """Expand a query with default settings.

    Args:
        query: User query
        conversation: Optional conversation context

    Returns:
        ExpandedQuery result
    """
    expander = QueryExpander()
    return expander.expand(query, conversation)
