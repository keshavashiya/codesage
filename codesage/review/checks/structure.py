"""Structural code analysis: function length, nesting depth, parameter count, god classes."""

import ast
import logging
from pathlib import Path
from typing import List

from codesage.review.models import ReviewFinding

logger = logging.getLogger(__name__)

# Default thresholds
MAX_FUNCTION_LINES = 50
MAX_NESTING_DEPTH = 4
MAX_PARAMETERS = 5
GOD_CLASS_METHODS = 20
GOD_CLASS_LINES = 500


class _NestingVisitor(ast.NodeVisitor):
    """Measure maximum nesting depth in a function body."""

    def __init__(self) -> None:
        self.max_depth = 0
        self._current_depth = 0

    def _enter_block(self, node: ast.AST) -> None:
        self._current_depth += 1
        if self._current_depth > self.max_depth:
            self.max_depth = self._current_depth
        self.generic_visit(node)
        self._current_depth -= 1

    visit_If = _enter_block
    visit_For = _enter_block
    visit_AsyncFor = _enter_block
    visit_While = _enter_block
    visit_With = _enter_block
    visit_AsyncWith = _enter_block
    visit_Try = _enter_block
    visit_ExceptHandler = _enter_block


def _function_line_count(node: ast.AST) -> int:
    """Count the number of lines a function spans."""
    if not hasattr(node, "end_lineno") or node.end_lineno is None:
        # Fallback: count body statements
        return len(getattr(node, "body", []))
    return node.end_lineno - node.lineno + 1


def _count_params(node: ast.FunctionDef) -> int:
    """Count total parameters (positional + keyword, excluding self/cls)."""
    args = node.args
    params = (
        args.posonlyargs + args.args + args.kwonlyargs
    )
    # Exclude 'self' and 'cls'
    return len([p for p in params if p.arg not in ("self", "cls")])


class StructureChecker:
    """Detect structural issues: long functions, deep nesting, too many params, god classes."""

    def __init__(
        self,
        max_function_lines: int = MAX_FUNCTION_LINES,
        max_nesting_depth: int = MAX_NESTING_DEPTH,
        max_parameters: int = MAX_PARAMETERS,
        god_class_methods: int = GOD_CLASS_METHODS,
        god_class_lines: int = GOD_CLASS_LINES,
    ) -> None:
        self.max_function_lines = max_function_lines
        self.max_nesting_depth = max_nesting_depth
        self.max_parameters = max_parameters
        self.god_class_methods = god_class_methods
        self.god_class_lines = god_class_lines

    def check(self, file_path: Path, content: str) -> List[ReviewFinding]:
        """Run all structure checks on a Python file.

        Args:
            file_path: Path to the file (for reporting).
            content: Source code content.

        Returns:
            List of structural findings.
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return []

        findings: List[ReviewFinding] = []

        findings.extend(self._check_functions(tree, file_path))
        findings.extend(self._check_classes(tree, file_path))

        return findings

    def _check_functions(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Check functions for length, nesting, and parameter count."""
        findings: List[ReviewFinding] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            name = node.name

            # Function length
            line_count = _function_line_count(node)
            if line_count > self.max_function_lines:
                findings.append(
                    ReviewFinding(
                        severity="warning",
                        category="structure",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-LONG-FUNCTION",
                        message=f"Function '{name}()' is {line_count} lines (max: {self.max_function_lines})",
                        suggestion="Extract helper functions to reduce complexity",
                        source="static",
                    )
                )

            # Nesting depth
            visitor = _NestingVisitor()
            visitor.visit(node)
            if visitor.max_depth > self.max_nesting_depth:
                findings.append(
                    ReviewFinding(
                        severity="warning",
                        category="structure",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-DEEP-NESTING",
                        message=f"Function '{name}()' has nesting depth {visitor.max_depth} (max: {self.max_nesting_depth})",
                        suggestion="Use early returns, guard clauses, or extract inner logic",
                        source="static",
                    )
                )

            # Too many parameters
            param_count = _count_params(node)
            if param_count > self.max_parameters:
                findings.append(
                    ReviewFinding(
                        severity="suggestion",
                        category="structure",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-TOO-MANY-PARAMS",
                        message=f"Function '{name}()' has {param_count} parameters (max: {self.max_parameters})",
                        suggestion="Group related parameters into a dataclass or config object",
                        source="static",
                    )
                )

        return findings

    def _check_classes(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Check for god classes (too many methods or too many lines)."""
        findings: List[ReviewFinding] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # Count methods (including static/class methods)
            method_count = sum(
                1 for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )

            # Class line count
            class_lines = _function_line_count(node)

            if method_count > self.god_class_methods:
                findings.append(
                    ReviewFinding(
                        severity="warning",
                        category="structure",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-GOD-CLASS",
                        message=f"Class '{node.name}' has {method_count} methods (max: {self.god_class_methods}) — possible god class",
                        suggestion="Split into smaller, focused classes with single responsibility",
                        source="static",
                    )
                )
            elif class_lines > self.god_class_lines:
                findings.append(
                    ReviewFinding(
                        severity="suggestion",
                        category="structure",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-LARGE-CLASS",
                        message=f"Class '{node.name}' is {class_lines} lines (max: {self.god_class_lines})",
                        suggestion="Consider extracting some functionality into separate classes",
                        source="static",
                    )
                )

        return findings
