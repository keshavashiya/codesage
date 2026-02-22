"""Tree-sitter based static analysis checker for non-Python source files.

Uses real AST parsing via tree-sitter for accurate complexity, structure,
and naming analysis of Rust, Go, JavaScript, and TypeScript files.

Falls back to GenericFileChecker if tree-sitter packages are not installed.

Rules:
- TS-COMPLEXITY       : Cyclomatic complexity exceeds threshold (>10 branches)
- TS-LONG-FN         : Function body exceeds MAX_FUNCTION_LINES lines (>60)
- TS-DEEP-NEST       : Nesting depth exceeds MAX_NESTING_DEPTH (>4)
- TS-TOO-MANY-PARAMS : Function has more than MAX_PARAMS parameters (>5)
- TS-RUST-NAMING     : Violates Rust naming conventions (snake_case/PascalCase/SCREAMING_SNAKE)
- TS-GO-NAMING       : Violates Go naming conventions (camelCase/PascalCase)
- TS-JS-NAMING       : Violates JavaScript naming conventions
- TS-TS-NAMING       : Violates TypeScript naming conventions
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from codesage.review.models import ReviewFinding
from codesage.utils.language_detector import EXTENSION_TO_LANGUAGE
from codesage.utils.logging import get_logger
from codesage.utils.treesitter_utils import TREESITTER_AVAILABLE, get_parser

logger = get_logger("review.treesitter_checks")

# ---------------------------------------------------------------------------
# AST node type constants per language
# ---------------------------------------------------------------------------

# Nodes that increment cyclomatic complexity (branches)
COMPLEXITY_NODES: Dict[str, Set[str]] = {
    "rust": {
        "if_expression",
        "while_expression",
        "for_expression",
        "match_arm",
        "loop_expression",
    },
    "go": {
        "if_statement",
        "for_statement",
        "switch_case",
        "select_statement",
        "type_switch_statement",
    },
    "javascript": {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_case",
        "ternary_expression",
    },
    "typescript": {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_case",
        "ternary_expression",
    },
}

# Nodes that define function scope for structural checks
FUNCTION_NODES: Dict[str, Set[str]] = {
    "rust": {"function_item"},
    "go": {"function_declaration", "method_declaration"},
    "javascript": {
        "function_declaration",
        "arrow_function",
        "function_expression",
        "method_definition",
    },
    "typescript": {
        "function_declaration",
        "arrow_function",
        "function_expression",
        "method_definition",
    },
}

# Nodes that open a new nesting scope (for depth calculation)
_NESTING_NODES: Dict[str, Set[str]] = {
    "rust": {
        "if_expression",
        "while_expression",
        "for_expression",
        "loop_expression",
        "match_expression",
    },
    "go": {"if_statement", "for_statement", "switch_statement"},
    "javascript": {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
    },
    "typescript": {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
    },
}

# Per-language naming rules: kind → (regex_pattern, convention_name, rule_id)
NAMING_RULES: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    "rust": {
        "function": (r"^[a-z][a-z0-9_]*$", "snake_case", "TS-RUST-NAMING"),
        "struct": (r"^[A-Z][a-zA-Z0-9]*$", "PascalCase", "TS-RUST-NAMING"),
        "enum": (r"^[A-Z][a-zA-Z0-9]*$", "PascalCase", "TS-RUST-NAMING"),
        "const": (r"^[A-Z][A-Z0-9_]*$", "SCREAMING_SNAKE_CASE", "TS-RUST-NAMING"),
    },
    "go": {
        "function": (r"^[a-zA-Z][a-zA-Z0-9]*$", "camelCase/PascalCase", "TS-GO-NAMING"),
    },
    "javascript": {
        "function": (
            r"^[a-z_$][a-zA-Z0-9_$]*$|^[A-Z][a-zA-Z0-9]*$",
            "camelCase/PascalCase",
            "TS-JS-NAMING",
        ),
        "class": (r"^[A-Z][a-zA-Z0-9]*$", "PascalCase", "TS-JS-NAMING"),
    },
    "typescript": {
        "function": (
            r"^[a-z_$][a-zA-Z0-9_$]*$|^[A-Z][a-zA-Z0-9]*$",
            "camelCase/PascalCase",
            "TS-TS-NAMING",
        ),
        "class": (r"^[A-Z][a-zA-Z0-9]*$", "PascalCase", "TS-TS-NAMING"),
        "interface": (r"^[A-Z][a-zA-Z0-9]*$", "PascalCase", "TS-TS-NAMING"),
    },
}

# Tree-sitter node type → naming rule kind, per language
_NAMING_NODE_KINDS: Dict[str, Dict[str, str]] = {
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "const_item": "const",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "function",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
    },
}

# Parameter list node type per language
_PARAM_LIST_NODES: Dict[str, str] = {
    "rust": "parameters",
    "go": "parameter_list",
    "javascript": "formal_parameters",
    "typescript": "formal_parameters",
}

# Parameter item node types per language
_PARAM_ITEM_NODES: Dict[str, Set[str]] = {
    "rust": {"parameter", "self_parameter", "variadic_parameter"},
    "go": {"parameter_declaration", "variadic_parameter_declaration"},
    "javascript": {
        "identifier",
        "assignment_pattern",
        "rest_pattern",
        "object_pattern",
        "array_pattern",
    },
    "typescript": {
        "required_parameter",
        "optional_parameter",
        "rest_parameter",
    },
}

# Extension → language string (non-Python subset of EXTENSION_TO_LANGUAGE)
_EXT_TO_LANG: Dict[str, str] = {
    ext: lang
    for ext, lang in EXTENSION_TO_LANGUAGE.items()
    if lang != "python"
}

# Thresholds (consistent with generic_checks.py)
MAX_COMPLEXITY = 10
MAX_FUNCTION_LINES = 60
MAX_NESTING_DEPTH = 4
MAX_PARAMS = 5


# ---------------------------------------------------------------------------
# TreeSitterReviewChecker
# ---------------------------------------------------------------------------


class TreeSitterReviewChecker:
    """AST-based static analysis for non-Python source files using tree-sitter.

    Provides significantly more accurate results than text/regex analysis by
    using the actual parse tree.  Falls back to ``GenericFileChecker`` if
    tree-sitter is unavailable or a specific grammar is not installed.

    Parser instances are shared via ``codesage.utils.treesitter_utils.get_parser``
    so each language is initialised at most once per process.

    Usage::

        checker = TreeSitterReviewChecker()
        findings = checker.check(Path("src/main.rs"), content)
    """

    def check(self, file_path: Path, content: str) -> List[ReviewFinding]:
        """Run all tree-sitter checks on the given file.

        Args:
            file_path: Used for language detection and in finding locations.
            content:   Full source text of the file.

        Returns:
            List of ReviewFinding (may be empty).
        """
        if not TREESITTER_AVAILABLE:
            return self._fallback(file_path, content)

        language = _EXT_TO_LANG.get(file_path.suffix.lower())
        if language is None:
            return []

        parser = self._get_parser(language)
        if parser is None:
            return self._fallback(file_path, content)

        try:
            tree = parser.parse(content.encode("utf-8"))
        except Exception as exc:
            logger.debug(f"tree-sitter parse failed for {file_path}: {exc}")
            return self._fallback(file_path, content)

        findings: List[ReviewFinding] = []

        # --- Function-level checks ---
        fn_types = FUNCTION_NODES.get(language, set())
        for fn_node in self._walk(tree.root_node, fn_types):
            findings.extend(self._check_complexity(fn_node, language, file_path))
            findings.extend(self._check_structure(fn_node, language, file_path))

        # --- Declaration naming checks ---
        naming_kinds = _NAMING_NODE_KINDS.get(language, {})
        if naming_kinds:
            for decl_node in self._walk(tree.root_node, set(naming_kinds.keys())):
                kind = naming_kinds.get(decl_node.type)
                if kind:
                    finding = self._check_naming(decl_node, kind, language, file_path)
                    if finding:
                        findings.append(finding)

        return findings

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_complexity(
        self, fn_node: Any, language: str, file_path: Path
    ) -> List[ReviewFinding]:
        """Count branch nodes inside a function; flag if > MAX_COMPLEXITY."""
        branch_types = COMPLEXITY_NODES.get(language, set())
        if not branch_types:
            return []

        count = sum(1 for _ in self._walk(fn_node, branch_types))
        if count <= MAX_COMPLEXITY:
            return []

        fn_name = self._get_name(fn_node) or "?"
        line = fn_node.start_point[0] + 1
        return [
            ReviewFinding(
                rule_id="TS-COMPLEXITY",
                file=file_path,
                line=line,
                message=(
                    f"Function '{fn_name}' has cyclomatic complexity {count} "
                    f"(limit: {MAX_COMPLEXITY}); consider extracting sub-functions"
                ),
                severity="warning",
                category="complexity",
                source="static",
            )
        ]

    def _check_structure(
        self, fn_node: Any, language: str, file_path: Path
    ) -> List[ReviewFinding]:
        """Check function length, nesting depth, and parameter count."""
        findings: List[ReviewFinding] = []
        fn_name = self._get_name(fn_node) or "?"
        line_start = fn_node.start_point[0] + 1
        line_end = fn_node.end_point[0] + 1
        fn_lines = line_end - line_start + 1

        if fn_lines > MAX_FUNCTION_LINES:
            findings.append(
                ReviewFinding(
                    rule_id="TS-LONG-FN",
                    file=file_path,
                    line=line_start,
                    message=(
                        f"Function '{fn_name}' is {fn_lines} lines long "
                        f"(limit: {MAX_FUNCTION_LINES}); consider splitting"
                    ),
                    severity="warning",
                    category="complexity",
                    source="static",
                )
            )

        max_depth = self._max_nesting_depth(fn_node, language)
        if max_depth > MAX_NESTING_DEPTH:
            findings.append(
                ReviewFinding(
                    rule_id="TS-DEEP-NEST",
                    file=file_path,
                    line=line_start,
                    message=(
                        f"Function '{fn_name}' has nesting depth {max_depth} "
                        f"(limit: {MAX_NESTING_DEPTH}); extract sub-functions"
                    ),
                    severity="warning",
                    category="complexity",
                    source="static",
                )
            )

        param_count = self._param_count(fn_node, language)
        if param_count > MAX_PARAMS:
            findings.append(
                ReviewFinding(
                    rule_id="TS-TOO-MANY-PARAMS",
                    file=file_path,
                    line=line_start,
                    message=(
                        f"Function '{fn_name}' has {param_count} parameters "
                        f"(limit: {MAX_PARAMS}); consider a config struct"
                    ),
                    severity="suggestion",
                    category="structure",
                    source="static",
                )
            )

        return findings

    def _check_naming(
        self, node: Any, kind: str, language: str, file_path: Path
    ) -> Optional[ReviewFinding]:
        """Check that a declaration follows the language's naming convention."""
        lang_rules = NAMING_RULES.get(language, {})
        rule = lang_rules.get(kind)
        if rule is None:
            return None

        pattern, convention, rule_id = rule
        name = self._get_name(node)
        if not name or len(name) <= 1:
            return None

        # Skip single-letter generics (common in Go/Rust like T, E, K, V)
        if len(name) == 1:
            return None

        if not re.match(pattern, name):
            line = node.start_point[0] + 1
            return ReviewFinding(
                rule_id=rule_id,
                file=file_path,
                line=line,
                message=(
                    f"{kind.capitalize()} '{name}' does not follow "
                    f"{convention} naming convention"
                ),
                severity="suggestion",
                category="naming",
                source="static",
            )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _walk(self, node: Any, types: Set[str]) -> Iterator[Any]:
        """DFS generator yielding all nodes whose type is in *types*."""
        if node.type in types:
            yield node
        for child in node.children:
            yield from self._walk(child, types)

    def _max_nesting_depth(self, fn_node: Any, language: str) -> int:
        """Compute the maximum nesting depth within a function body."""
        nesting_types = _NESTING_NODES.get(language, set())
        if not nesting_types:
            return 0

        def _depth(node: Any, current: int) -> int:
            d = current + (1 if node.type in nesting_types else 0)
            max_d = d
            for child in node.children:
                max_d = max(max_d, _depth(child, d))
            return max_d

        return _depth(fn_node, 0)

    def _param_count(self, fn_node: Any, language: str) -> int:
        """Count parameters in a function node."""
        param_list_type = _PARAM_LIST_NODES.get(language)
        if not param_list_type:
            return 0

        item_types = _PARAM_ITEM_NODES.get(language, set())

        for child in fn_node.children:
            if child.type == param_list_type:
                if not item_types:
                    # Fallback: count non-punctuation children
                    return sum(
                        1 for c in child.children
                        if c.type not in (",", "(", ")", " ", "\n")
                    )
                return sum(1 for c in child.children if c.type in item_types)

            # Arrow functions with single parameter (no formal_parameters wrapper)
            if language in ("javascript", "typescript") and child.type == "identifier":
                # This is a single-param arrow function: x => ...
                return 1

        return 0

    def _get_name(self, node: Any) -> Optional[str]:
        """Extract the primary name identifier from a declaration node."""
        for child in node.children:
            if child.type in (
                "identifier",
                "type_identifier",
                "field_identifier",
                "property_identifier",
            ):
                return child.text.decode("utf-8")
        return None

    def _get_parser(self, language: str) -> Optional[Any]:
        """Return a cached tree-sitter Parser for the given language.

        Delegates to ``codesage.utils.treesitter_utils.get_parser`` which
        maintains a single process-wide cache.
        """
        return get_parser(language)

    def _fallback(self, file_path: Path, content: str) -> List[ReviewFinding]:
        """Fall back to GenericFileChecker when tree-sitter is unavailable."""
        from codesage.review.checks.generic_checks import GenericFileChecker
        return GenericFileChecker().check(file_path, content)
