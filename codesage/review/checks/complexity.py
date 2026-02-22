"""Cyclomatic complexity analysis via AST.

Counts decision points per function and flags those exceeding a threshold.
"""

import ast
import logging
from pathlib import Path
from typing import List

from codesage.review.models import ReviewFinding

logger = logging.getLogger(__name__)

# Each of these AST node types adds 1 to cyclomatic complexity
_DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
)


class _ComplexityVisitor(ast.NodeVisitor):
    """Walks a function body counting decision points."""

    def __init__(self) -> None:
        self.complexity = 1  # Base complexity

    def _increment(self, node: ast.AST) -> None:
        self.complexity += 1
        self.generic_visit(node)

    visit_If = _increment
    visit_For = _increment
    visit_AsyncFor = _increment
    visit_While = _increment
    visit_ExceptHandler = _increment
    visit_With = _increment
    visit_AsyncWith = _increment
    visit_Assert = _increment

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each 'and' / 'or' adds one path
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # Ternary expression: x if cond else y
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # List/dict/set comprehension with ifs
        self.complexity += len(node.ifs)
        self.generic_visit(node)


def _compute_complexity(func_node: ast.AST) -> int:
    """Compute cyclomatic complexity for a function/method node."""
    visitor = _ComplexityVisitor()
    visitor.visit(func_node)
    return visitor.complexity


class ComplexityChecker:
    """Detect functions with excessive cyclomatic complexity."""

    def __init__(self, threshold: int = 10) -> None:
        self.threshold = threshold

    def check(self, file_path: Path, content: str) -> List[ReviewFinding]:
        """Check all functions in a Python file for complexity.

        Args:
            file_path: Path to the file (for reporting).
            content: Source code content.

        Returns:
            List of findings for functions exceeding the threshold.
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return []

        findings: List[ReviewFinding] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            complexity = _compute_complexity(node)
            if complexity > self.threshold:
                # Determine parent class name if method
                qualifier = node.name
                severity = "warning" if complexity <= 20 else "high"

                findings.append(
                    ReviewFinding(
                        severity=severity,
                        category="complexity",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-HIGH-COMPLEXITY",
                        message=f"Function '{qualifier}()' has cyclomatic complexity {complexity} (threshold: {self.threshold})",
                        suggestion="Break into smaller functions or simplify conditional logic",
                        source="static",
                    )
                )

        return findings
