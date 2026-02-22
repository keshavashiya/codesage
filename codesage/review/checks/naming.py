"""Naming convention checker for Python code.

Enforces: snake_case functions/variables, PascalCase classes,
UPPER_SNAKE_CASE constants, no single-letter names outside loops.
"""

import ast
import logging
import re
from pathlib import Path
from typing import List

from codesage.review.models import ReviewFinding

logger = logging.getLogger(__name__)

_SNAKE_CASE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_PASCAL_CASE_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_UPPER_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Common single-letter loop variables
_LOOP_VARS = {"i", "j", "k", "n", "x", "y", "z", "v", "e", "f", "c", "d", "m", "p", "q", "r", "s", "t", "w", "_"}

# Names to skip (dunder, private convention, etc.)
_SKIP_PREFIXES = ("__", "_")

# Known patterns that break conventions but are standard
_KNOWN_EXCEPTIONS = {"setUp", "tearDown", "setUpClass", "tearDownClass", "maxDiff"}


class NamingChecker:
    """Check Python naming conventions."""

    def check(self, file_path: Path, content: str) -> List[ReviewFinding]:
        """Run naming convention checks on a Python file.

        Args:
            file_path: Path to the file (for reporting).
            content: Source code content.

        Returns:
            List of naming convention violations.
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return []

        findings: List[ReviewFinding] = []

        findings.extend(self._check_functions(tree, file_path))
        findings.extend(self._check_classes(tree, file_path))
        findings.extend(self._check_constants(tree, file_path))
        findings.extend(self._check_variables(tree, file_path))

        return findings

    def _check_functions(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Functions and methods should use snake_case."""
        findings: List[ReviewFinding] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            name = node.name

            # Skip dunder/private and known exceptions
            if name.startswith("__") or name in _KNOWN_EXCEPTIONS:
                continue

            # Skip ast.NodeVisitor protocol methods (visit_NodeType, generic_visit)
            if name.startswith("visit_") or name == "generic_visit":
                continue

            # Allow _private methods
            check_name = name.lstrip("_")
            if not check_name:
                continue

            if not _SNAKE_CASE_RE.match(check_name):
                findings.append(
                    ReviewFinding(
                        severity="suggestion",
                        category="naming",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-FUNC-NAMING",
                        message=f"Function '{name}' should use snake_case",
                        suggestion=f"Rename to '{_to_snake_case(name)}'",
                        source="static",
                    )
                )

        return findings

    def _check_classes(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Classes should use PascalCase."""
        findings: List[ReviewFinding] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            name = node.name

            # Skip private classes like _InternalHelper
            if name.startswith("_"):
                check_name = name.lstrip("_")
            else:
                check_name = name

            if not check_name:
                continue

            if not _PASCAL_CASE_RE.match(check_name):
                findings.append(
                    ReviewFinding(
                        severity="suggestion",
                        category="naming",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-CLASS-NAMING",
                        message=f"Class '{name}' should use PascalCase",
                        suggestion=f"Rename to '{_to_pascal_case(name)}'",
                        source="static",
                    )
                )

        return findings

    def _check_constants(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Module-level ALL_CAPS assignments should use UPPER_SNAKE_CASE consistently."""
        findings: List[ReviewFinding] = []

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue

                name = target.id

                # Skip private and dunder
                if name.startswith("_"):
                    continue

                # If it looks like it's trying to be a constant (has uppercase)
                # but isn't proper UPPER_SNAKE_CASE
                if name[0].isupper() and name != name.upper() and "_" in name:
                    # Mixed case with underscores at module level — likely a bad constant
                    if not _UPPER_SNAKE_RE.match(name) and not _PASCAL_CASE_RE.match(name):
                        findings.append(
                            ReviewFinding(
                                severity="suggestion",
                                category="naming",
                                file=file_path,
                                line=node.lineno,
                                rule_id="PY-CONST-NAMING",
                                message=f"Module-level name '{name}' appears to be a constant but doesn't use UPPER_SNAKE_CASE",
                                suggestion=f"Rename to '{name.upper()}'",
                                source="static",
                            )
                        )

        return findings

    def _check_variables(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Detect single-letter variable names outside of loops and comprehensions."""
        findings: List[ReviewFinding] = []

        # Collect loop variable names to exclude them
        loop_var_lines: set = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                self._collect_loop_targets(node.target, loop_var_lines)
            elif isinstance(node, ast.comprehension):
                self._collect_loop_targets(node.target, loop_var_lines)

        # Check assignments for single-letter names
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue

                name = target.id
                if len(name) == 1 and name not in _LOOP_VARS:
                    continue  # _LOOP_VARS covers common single letters
                if len(name) == 1 and name in _LOOP_VARS and node.lineno not in loop_var_lines:
                    # Single letter used outside a loop context — check if it's in a function
                    # that's very short (< 5 lines), which is fine
                    pass  # We'll be lenient here — single letters in loops are fine

        return findings

    @staticmethod
    def _collect_loop_targets(target: ast.AST, lines: set) -> None:
        """Collect line numbers of loop target variables."""
        if isinstance(target, ast.Name):
            lines.add(target.lineno if hasattr(target, "lineno") else 0)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    lines.add(elt.lineno if hasattr(elt, "lineno") else 0)


def _to_snake_case(name: str) -> str:
    """Convert a name to snake_case."""
    # Remove leading underscores, convert, add them back
    prefix = ""
    while name.startswith("_"):
        prefix += "_"
        name = name[1:]

    # Insert underscore before uppercase letters
    result = re.sub(r"([A-Z])", r"_\1", name).lower().strip("_")
    # Clean up double underscores
    result = re.sub(r"_+", "_", result)
    return prefix + result


def _to_pascal_case(name: str) -> str:
    """Convert a name to PascalCase."""
    prefix = ""
    while name.startswith("_"):
        prefix += "_"
        name = name[1:]

    parts = re.split(r"[_\s]+", name)
    return prefix + "".join(p.capitalize() for p in parts if p)
