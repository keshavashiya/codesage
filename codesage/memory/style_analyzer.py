"""Style analyzer for detecting coding patterns from code elements.

Analyzes code elements to extract naming conventions, docstring styles,
typing patterns, and other coding preferences. Language-aware: patterns
carry a ``"languages"`` field that restricts which languages they apply to.
An empty list (or absent field) means the pattern is universal.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from codesage.utils.logging import get_logger

from .models import LearnedPattern, PatternCategory

logger = get_logger("memory.style_analyzer")


@dataclass
class StyleMatch:
    """Represents a detected style pattern match."""

    pattern_name: str
    category: PatternCategory
    description: str
    pattern_text: str
    examples: List[str]
    confidence: float


class StyleAnalyzer:
    """Analyzes code elements to detect coding style patterns.

    Detects patterns in:
        - Naming conventions (snake_case, camelCase, etc.)
        - Docstring / doc-comment styles
        - Type annotation usage
        - Import organization
        - Error handling patterns

    Each pattern dict entry may include a ``"languages"`` key listing the
    languages it applies to. An empty list or absent key means universal.
    """

    # Naming convention patterns
    NAMING_PATTERNS: Dict[str, Any] = {
        "snake_case_functions": {
            "pattern": r"^[a-z][a-z0-9_]*$",
            "description": "Function names use snake_case",
            "applies_to": ["function", "method"],
            "languages": ["python", "rust", "go"],
        },
        "snake_case_variables": {
            "pattern": r"^[a-z][a-z0-9_]*$",
            "description": "Variable names use snake_case",
            "applies_to": ["variable"],
            "languages": ["python", "rust", "go"],
        },
        "camel_case_functions": {
            "pattern": r"^[a-z][a-zA-Z0-9]*$",
            "description": "Function / method names use camelCase",
            "applies_to": ["function", "method"],
            "languages": ["javascript", "typescript"],
        },
        "camel_case_variables": {
            "pattern": r"^[a-z][a-zA-Z0-9]*$",
            "description": "Variable names use camelCase",
            "applies_to": ["variable"],
            "languages": ["javascript", "typescript"],
        },
        "pascal_case_classes": {
            "pattern": r"^[A-Z][a-zA-Z0-9]*$",
            "description": "Class names use PascalCase",
            "applies_to": ["class"],
            "languages": [],
        },
        "screaming_snake_case_constants": {
            "pattern": r"^[A-Z][A-Z0-9_]*$",
            "description": "Constants use SCREAMING_SNAKE_CASE",
            "applies_to": ["constant"],
            "languages": [],
        },
        "private_prefix_underscore": {
            "pattern": r"^_[a-z][a-z0-9_]*$",
            "description": "Private members prefixed with underscore",
            "applies_to": ["function", "method", "variable"],
            "languages": ["python"],
        },
        "private_pascal_case_classes": {
            "pattern": r"^_[A-Z][a-zA-Z0-9]*$",
            "description": "Private class names use _PascalCase",
            "applies_to": ["class"],
            "languages": ["python"],
        },
        "ast_visitor_methods": {
            "pattern": r"^visit_[A-Z][a-zA-Z0-9]*$|^generic_visit$",
            "description": "AST NodeVisitor protocol methods (visit_NodeType)",
            "applies_to": ["function", "method"],
            "languages": ["python"],
        },
        "dunder_methods": {
            "pattern": r"^__[a-z][a-z0-9_]*__$",
            "description": "Uses dunder methods for special behavior",
            "applies_to": ["method"],
            "languages": ["python"],
        },
    }

    # Docstring / doc-comment style patterns
    DOCSTRING_PATTERNS: Dict[str, Any] = {
        "google_docstring": {
            "pattern": r"(Args:|Returns:|Raises:|Attributes:|Example:)",
            "description": "Uses Google-style docstrings",
            "languages": ["python"],
        },
        "numpy_docstring": {
            "pattern": r"(Parameters\n-+|Returns\n-+|Raises\n-+)",
            "description": "Uses NumPy-style docstrings",
            "languages": ["python"],
        },
        "sphinx_docstring": {
            "pattern": r"(:param\s|:returns:|:raises:|:type\s)",
            "description": "Uses Sphinx/reStructuredText-style docstrings",
            "languages": ["python"],
        },
        "one_liner_docstring": {
            "pattern": r'^"""[^"]+"""$',
            "description": "Uses single-line docstrings for simple functions",
            "languages": ["python"],
        },
        "jsdoc": {
            "pattern": r"@param\s|@returns?\s|@throws?\s",
            "description": "Uses JSDoc comments",
            "languages": ["javascript", "typescript"],
        },
        "rustdoc": {
            "pattern": r"///|//!",
            "description": "Uses Rustdoc line doc-comments",
            "languages": ["rust"],
        },
        "godoc": {
            "pattern": r"^// [A-Z]",
            "description": "Uses Go documentation comments",
            "languages": ["go"],
        },
    }

    # Type annotation patterns
    TYPING_PATTERNS: Dict[str, Any] = {
        "type_hints_parameters": {
            "pattern": r"def\s+\w+\s*\([^)]*:\s*\w+",
            "description": "Uses type hints for function parameters",
            "languages": ["python"],
        },
        "type_hints_return": {
            "pattern": r"def\s+\w+\s*\([^)]*\)\s*->\s*\w+",
            "description": "Uses return type annotations",
            "languages": ["python"],
        },
        "optional_types": {
            "pattern": r"Optional\[|Union\[.*None\]|\s*\|\s*None",
            "description": "Uses Optional or Union with None for nullable types",
            "languages": ["python"],
        },
        "list_type_hints": {
            "pattern": r"List\[|list\[",
            "description": "Uses List type hints for lists",
            "languages": ["python"],
        },
        "dict_type_hints": {
            "pattern": r"Dict\[|dict\[",
            "description": "Uses Dict type hints for dictionaries",
            "languages": ["python"],
        },
        "typescript_interfaces": {
            "pattern": r"interface\s+\w+|type\s+\w+\s*=",
            "description": "Uses TypeScript interfaces/type aliases",
            "languages": ["typescript"],
        },
        "rust_result_option": {
            "pattern": r"\bResult<|\bOption<",
            "description": "Uses Rust Result/Option types",
            "languages": ["rust"],
        },
    }

    # Import patterns (universal — all languages have some import/use concept)
    IMPORT_PATTERNS: Dict[str, Any] = {
        "absolute_imports": {
            "pattern": r"^from\s+\w+\.\w+",
            "description": "Uses absolute imports",
            "languages": [],
        },
        "relative_imports": {
            "pattern": r"^from\s+\.",
            "description": "Uses relative imports",
            "languages": [],
        },
        "import_grouping": {
            "pattern": r"(^import\s+\w+\n)+\n(^from\s+\w+)",
            "description": "Groups stdlib imports before third-party",
            "languages": [],
        },
        "from_imports": {
            "pattern": r"^from\s+\S+\s+import\s+",
            "description": "Prefers 'from X import Y' style",
            "languages": [],
        },
    }

    # Error handling patterns
    ERROR_HANDLING_PATTERNS: Dict[str, Any] = {
        "specific_exceptions": {
            "pattern": r"except\s+(?!Exception)[A-Z]\w+Error",
            "description": "Catches specific exception types",
            "languages": [],
        },
        "exception_chaining": {
            "pattern": r"raise\s+\w+\s+from\s+\w+",
            "description": "Uses exception chaining (raise ... from ...)",
            "languages": ["python"],
        },
        "context_managers": {
            "pattern": r"with\s+\w+\([^)]*\)\s*(as\s+\w+)?:",
            "description": "Uses context managers for resource handling",
            "languages": ["python"],
        },
        "try_except_else": {
            "pattern": r"try:.*except.*else:",
            "description": "Uses try/except/else pattern",
            "languages": [],
        },
        "try_except_finally": {
            "pattern": r"try:.*except.*finally:",
            "description": "Uses try/except/finally pattern",
            "languages": [],
        },
    }

    # ---------------------------------------------------------------------------
    # Single source of truth: names of patterns that apply to Python only.
    # Replaces all scattered local ``_PYTHON_SPECIFIC_PATTERNS`` sets in the
    # codebase — import from here instead of redefining locally.
    # ---------------------------------------------------------------------------
    PYTHON_ONLY_PATTERN_NAMES: FrozenSet[str] = frozenset(
        name
        for dct in (
            NAMING_PATTERNS,
            DOCSTRING_PATTERNS,
            TYPING_PATTERNS,
            IMPORT_PATTERNS,
            ERROR_HANDLING_PATTERNS,
        )
        for name, cfg in dct.items()
        if cfg.get("languages") == ["python"]
    )

    def __init__(self) -> None:
        """Initialize the style analyzer."""
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        all_patterns = {
            **self.NAMING_PATTERNS,
            **self.DOCSTRING_PATTERNS,
            **self.TYPING_PATTERNS,
            **self.IMPORT_PATTERNS,
            **self.ERROR_HANDLING_PATTERNS,
        }

        for name, config in all_patterns.items():
            try:
                self._compiled_patterns[name] = re.compile(
                    config["pattern"], re.MULTILINE
                )
            except re.error as e:
                logger.warning(f"Failed to compile pattern {name}: {e}")

    # ---------------------------------------------------------------------------
    # Language filtering
    # ---------------------------------------------------------------------------

    @classmethod
    def get_patterns_for_language(cls, pattern_dict: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Return only patterns applicable to the given language.

        A pattern applies when its ``"languages"`` list is empty/absent (universal)
        or explicitly contains *language*.
        """
        result: Dict[str, Any] = {}
        for name, cfg in pattern_dict.items():
            langs = cfg.get("languages", [])
            if not langs or language in langs:
                result[name] = cfg
        return result

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------

    def analyze_element(
        self,
        element_type: str,
        name: str,
        code: str,
        docstring: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[StyleMatch]:
        """Analyze a code element for style patterns.

        Args:
            element_type: Type of element (function, class, method, etc.).
            name: Name of the element.
            code: Source code of the element.
            docstring: Optional docstring.
            language: Primary language of the element (default: auto-detected).

        Returns:
            List of detected style patterns applicable to *language*.
        """
        matches: List[StyleMatch] = []

        language = language or "python"

        # Analyze naming conventions
        matches.extend(self._analyze_naming(element_type, name, language))

        # Analyze docstring / doc-comment style
        if docstring:
            matches.extend(self._analyze_docstring(docstring, language))

        # Analyze code patterns (typing, imports, error handling)
        matches.extend(self._analyze_code_patterns(code, language))

        return matches

    def _analyze_naming(self, element_type: str, name: str, language: str = "python") -> List[StyleMatch]:
        """Analyze naming conventions filtered by language."""
        matches: List[StyleMatch] = []
        applicable = self.get_patterns_for_language(self.NAMING_PATTERNS, language)

        for pattern_name, config in applicable.items():
            applies_to = config.get("applies_to", [])
            if element_type not in applies_to:
                continue

            pattern = self._compiled_patterns.get(pattern_name)
            if pattern and pattern.match(name):
                matches.append(
                    StyleMatch(
                        pattern_name=pattern_name,
                        category=PatternCategory.NAMING,
                        description=config["description"],
                        pattern_text=config["pattern"],
                        examples=[name],
                        confidence=0.8,
                    )
                )

        return matches

    def _analyze_docstring(self, docstring: str, language: str = "python") -> List[StyleMatch]:
        """Analyze docstring / doc-comment style filtered by language."""
        matches: List[StyleMatch] = []
        applicable = self.get_patterns_for_language(self.DOCSTRING_PATTERNS, language)

        for pattern_name, config in applicable.items():
            pattern = self._compiled_patterns.get(pattern_name)
            if pattern and pattern.search(docstring):
                matches.append(
                    StyleMatch(
                        pattern_name=pattern_name,
                        category=PatternCategory.DOCSTRING,
                        description=config["description"],
                        pattern_text=config["pattern"],
                        examples=[docstring[:100] + "..." if len(docstring) > 100 else docstring],
                        confidence=0.8,
                    )
                )

        return matches

    def _analyze_code_patterns(self, code: str, language: str = "python") -> List[StyleMatch]:
        """Analyze code for typing, import, and error handling patterns filtered by language."""
        matches: List[StyleMatch] = []

        # Typing patterns
        for pattern_name, config in self.get_patterns_for_language(self.TYPING_PATTERNS, language).items():
            pattern = self._compiled_patterns.get(pattern_name)
            if pattern and pattern.search(code):
                match = pattern.search(code)
                example = match.group(0) if match else ""
                matches.append(
                    StyleMatch(
                        pattern_name=pattern_name,
                        category=PatternCategory.TYPING,
                        description=config["description"],
                        pattern_text=config["pattern"],
                        examples=[example] if example else [],
                        confidence=0.7,
                    )
                )

        # Import patterns
        for pattern_name, config in self.get_patterns_for_language(self.IMPORT_PATTERNS, language).items():
            pattern = self._compiled_patterns.get(pattern_name)
            if pattern and pattern.search(code):
                match = pattern.search(code)
                example = match.group(0) if match else ""
                matches.append(
                    StyleMatch(
                        pattern_name=pattern_name,
                        category=PatternCategory.IMPORTS,
                        description=config["description"],
                        pattern_text=config["pattern"],
                        examples=[example] if example else [],
                        confidence=0.7,
                    )
                )

        # Error handling patterns
        for pattern_name, config in self.get_patterns_for_language(self.ERROR_HANDLING_PATTERNS, language).items():
            pattern = self._compiled_patterns.get(pattern_name)
            if pattern and pattern.search(code):
                match = pattern.search(code)
                example = match.group(0) if match else ""
                matches.append(
                    StyleMatch(
                        pattern_name=pattern_name,
                        category=PatternCategory.ERROR_HANDLING,
                        description=config["description"],
                        pattern_text=config["pattern"],
                        examples=[example] if example else [],
                        confidence=0.7,
                    )
                )

        return matches

    def analyze_elements(
        self,
        elements: List[Dict[str, Any]],
        language: Optional[str] = None,
    ) -> Dict[str, List[StyleMatch]]:
        """Analyze multiple code elements.

        Args:
            elements: List of element dictionaries with keys:
                - type: Element type
                - name: Element name
                - code: Source code
                - docstring: Optional docstring
            language: Primary language for all elements (default: auto-detected).

        Returns:
            Dictionary mapping element IDs to their style matches.
        """
        results: Dict[str, List[StyleMatch]] = {}

        # Determine language dynamically from first element if present
        language = language or (elements[0].get("language", "python") if elements else "python")

        for element in elements:
            element_id = element.get("id", element.get("name", "unknown"))
            matches = self.analyze_element(
                element_type=element.get("type", "unknown"),
                name=element.get("name", ""),
                code=element.get("code", ""),
                docstring=element.get("docstring"),
                language=language,
            )
            if matches:
                results[element_id] = matches

        return results

    def aggregate_patterns(
        self,
        all_matches: Dict[str, List[StyleMatch]],
    ) -> List[Tuple[str, int, float]]:
        """Aggregate pattern matches to find common patterns.

        Args:
            all_matches: Dictionary of element IDs to style matches.

        Returns:
            List of (pattern_name, count, avg_confidence) tuples.
        """
        pattern_stats: Dict[str, Dict[str, Any]] = {}

        for element_matches in all_matches.values():
            for match in element_matches:
                if match.pattern_name not in pattern_stats:
                    pattern_stats[match.pattern_name] = {
                        "count": 0,
                        "total_confidence": 0.0,
                        "match": match,
                    }

                pattern_stats[match.pattern_name]["count"] += 1
                pattern_stats[match.pattern_name]["total_confidence"] += match.confidence

        results = []
        for name, stats in pattern_stats.items():
            avg_confidence = stats["total_confidence"] / stats["count"]
            results.append((name, stats["count"], avg_confidence))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def to_learned_patterns(
        self,
        aggregated: List[Tuple[str, int, float]],
        all_matches: Dict[str, List[StyleMatch]],
        min_occurrences: int = 2,
        min_confidence: float = 0.5,
    ) -> List[LearnedPattern]:
        """Convert aggregated matches to LearnedPattern objects.

        Args:
            aggregated: Aggregated pattern statistics.
            all_matches: Original match data for examples.
            min_occurrences: Minimum occurrences to include.
            min_confidence: Minimum confidence to include.

        Returns:
            List of LearnedPattern objects.
        """
        patterns = []

        pattern_examples: Dict[str, List[str]] = {}
        pattern_info: Dict[str, StyleMatch] = {}

        for element_matches in all_matches.values():
            for match in element_matches:
                if match.pattern_name not in pattern_examples:
                    pattern_examples[match.pattern_name] = []
                    pattern_info[match.pattern_name] = match

                pattern_examples[match.pattern_name].extend(match.examples)

        for name, count, avg_confidence in aggregated:
            if count < min_occurrences or avg_confidence < min_confidence:
                continue

            info = pattern_info.get(name)
            if not info:
                continue

            examples = list(set(pattern_examples.get(name, [])))[:5]

            pattern = LearnedPattern.create(
                name=name,
                category=info.category,
                description=info.description,
                pattern_text=info.pattern_text,
                examples=examples,
                occurrence_count=count,
                confidence_score=min(avg_confidence, 1.0),
            )
            patterns.append(pattern)

        return patterns
