"""Style analyzer for detecting coding patterns from code elements.

Analyzes code elements to extract naming conventions, docstring styles,
typing patterns, and other coding preferences.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
        - Docstring styles (Google, NumPy, reStructuredText)
        - Type annotation usage
        - Import organization
        - Error handling patterns
    """

    # Naming convention patterns
    NAMING_PATTERNS = {
        "snake_case_functions": {
            "pattern": r"^[a-z][a-z0-9_]*$",
            "description": "Function names use snake_case",
            "applies_to": ["function", "method"],
        },
        "snake_case_variables": {
            "pattern": r"^[a-z][a-z0-9_]*$",
            "description": "Variable names use snake_case",
            "applies_to": ["variable"],
        },
        "pascal_case_classes": {
            "pattern": r"^[A-Z][a-zA-Z0-9]*$",
            "description": "Class names use PascalCase",
            "applies_to": ["class"],
        },
        "screaming_snake_case_constants": {
            "pattern": r"^[A-Z][A-Z0-9_]*$",
            "description": "Constants use SCREAMING_SNAKE_CASE",
            "applies_to": ["constant"],
        },
        "private_prefix_underscore": {
            "pattern": r"^_[a-z][a-z0-9_]*$",
            "description": "Private members prefixed with underscore",
            "applies_to": ["function", "method", "variable"],
        },
        "dunder_methods": {
            "pattern": r"^__[a-z][a-z0-9_]*__$",
            "description": "Uses dunder methods for special behavior",
            "applies_to": ["method"],
        },
    }

    # Docstring style patterns
    DOCSTRING_PATTERNS = {
        "google_docstring": {
            "pattern": r"(Args:|Returns:|Raises:|Attributes:|Example:)",
            "description": "Uses Google-style docstrings",
        },
        "numpy_docstring": {
            "pattern": r"(Parameters\n-+|Returns\n-+|Raises\n-+)",
            "description": "Uses NumPy-style docstrings",
        },
        "sphinx_docstring": {
            "pattern": r"(:param\s|:returns:|:raises:|:type\s)",
            "description": "Uses Sphinx/reStructuredText-style docstrings",
        },
        "one_liner_docstring": {
            "pattern": r'^"""[^"]+"""$',
            "description": "Uses single-line docstrings for simple functions",
        },
    }

    # Type annotation patterns
    TYPING_PATTERNS = {
        "type_hints_parameters": {
            "pattern": r"def\s+\w+\s*\([^)]*:\s*\w+",
            "description": "Uses type hints for function parameters",
        },
        "type_hints_return": {
            "pattern": r"def\s+\w+\s*\([^)]*\)\s*->\s*\w+",
            "description": "Uses return type annotations",
        },
        "optional_types": {
            "pattern": r"Optional\[|Union\[.*None\]|\s*\|\s*None",
            "description": "Uses Optional or Union with None for nullable types",
        },
        "list_type_hints": {
            "pattern": r"List\[|list\[",
            "description": "Uses List type hints for lists",
        },
        "dict_type_hints": {
            "pattern": r"Dict\[|dict\[",
            "description": "Uses Dict type hints for dictionaries",
        },
    }

    # Import patterns
    IMPORT_PATTERNS = {
        "absolute_imports": {
            "pattern": r"^from\s+\w+\.\w+",
            "description": "Uses absolute imports",
        },
        "relative_imports": {
            "pattern": r"^from\s+\.",
            "description": "Uses relative imports",
        },
        "import_grouping": {
            "pattern": r"(^import\s+\w+\n)+\n(^from\s+\w+)",
            "description": "Groups stdlib imports before third-party",
        },
        "from_imports": {
            "pattern": r"^from\s+\S+\s+import\s+",
            "description": "Prefers 'from X import Y' style",
        },
    }

    # Error handling patterns
    ERROR_HANDLING_PATTERNS = {
        "specific_exceptions": {
            "pattern": r"except\s+(?!Exception)[A-Z]\w+Error",
            "description": "Catches specific exception types",
        },
        "exception_chaining": {
            "pattern": r"raise\s+\w+\s+from\s+\w+",
            "description": "Uses exception chaining (raise ... from ...)",
        },
        "context_managers": {
            "pattern": r"with\s+\w+\([^)]*\)\s*(as\s+\w+)?:",
            "description": "Uses context managers for resource handling",
        },
        "try_except_else": {
            "pattern": r"try:.*except.*else:",
            "description": "Uses try/except/else pattern",
        },
        "try_except_finally": {
            "pattern": r"try:.*except.*finally:",
            "description": "Uses try/except/finally pattern",
        },
    }

    def __init__(self) -> None:
        """Initialize the style analyzer."""
        # Compile patterns
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

    def analyze_element(
        self,
        element_type: str,
        name: str,
        code: str,
        docstring: Optional[str] = None,
    ) -> List[StyleMatch]:
        """Analyze a code element for style patterns.

        Args:
            element_type: Type of element (function, class, method, etc.).
            name: Name of the element.
            code: Source code of the element.
            docstring: Optional docstring.

        Returns:
            List of detected style patterns.
        """
        matches = []

        # Analyze naming conventions
        matches.extend(self._analyze_naming(element_type, name))

        # Analyze docstring style
        if docstring:
            matches.extend(self._analyze_docstring(docstring))

        # Analyze code patterns
        matches.extend(self._analyze_code_patterns(code))

        return matches

    def _analyze_naming(self, element_type: str, name: str) -> List[StyleMatch]:
        """Analyze naming conventions.

        Args:
            element_type: Type of element.
            name: Element name.

        Returns:
            List of naming pattern matches.
        """
        matches = []

        for pattern_name, config in self.NAMING_PATTERNS.items():
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

    def _analyze_docstring(self, docstring: str) -> List[StyleMatch]:
        """Analyze docstring style.

        Args:
            docstring: Docstring text.

        Returns:
            List of docstring pattern matches.
        """
        matches = []

        for pattern_name, config in self.DOCSTRING_PATTERNS.items():
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

    def _analyze_code_patterns(self, code: str) -> List[StyleMatch]:
        """Analyze code for typing, import, and error handling patterns.

        Args:
            code: Source code.

        Returns:
            List of pattern matches.
        """
        matches = []

        # Typing patterns
        for pattern_name, config in self.TYPING_PATTERNS.items():
            pattern = self._compiled_patterns.get(pattern_name)
            if pattern and pattern.search(code):
                # Extract example
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
        for pattern_name, config in self.IMPORT_PATTERNS.items():
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
        for pattern_name, config in self.ERROR_HANDLING_PATTERNS.items():
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
    ) -> Dict[str, List[StyleMatch]]:
        """Analyze multiple code elements.

        Args:
            elements: List of element dictionaries with keys:
                - type: Element type
                - name: Element name
                - code: Source code
                - docstring: Optional docstring

        Returns:
            Dictionary mapping element IDs to their style matches.
        """
        results = {}

        for element in elements:
            element_id = element.get("id", element.get("name", "unknown"))
            matches = self.analyze_element(
                element_type=element.get("type", "unknown"),
                name=element.get("name", ""),
                code=element.get("code", ""),
                docstring=element.get("docstring"),
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

        # Calculate averages and sort by count
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

        # Collect examples for each pattern
        pattern_examples: Dict[str, List[str]] = {}
        pattern_info: Dict[str, StyleMatch] = {}

        for element_matches in all_matches.values():
            for match in element_matches:
                if match.pattern_name not in pattern_examples:
                    pattern_examples[match.pattern_name] = []
                    pattern_info[match.pattern_name] = match

                pattern_examples[match.pattern_name].extend(match.examples)

        # Create LearnedPattern objects
        for name, count, avg_confidence in aggregated:
            if count < min_occurrences or avg_confidence < min_confidence:
                continue

            info = pattern_info.get(name)
            if not info:
                continue

            # Deduplicate and limit examples
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
