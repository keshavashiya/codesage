"""Python-specific bad practice detection via AST analysis.

Detects: magic numbers, bare except, mutable default args,
unused imports, and missing type hints on public functions.
"""

import ast
import logging
from pathlib import Path
from typing import List, Set

from codesage.review.models import ReviewFinding

logger = logging.getLogger(__name__)

# Numbers that are commonly used and not "magic"
_ACCEPTABLE_NUMBERS = {0, 1, -1, 2, 0.0, 1.0, -1.0, 100, 1000}

# Built-in names that might appear as "unused" but are fine
_BUILTIN_IMPORTS = {"__future__"}


class PythonBadPracticeChecker:
    """Detect common Python bad practices using AST analysis."""

    def check(self, file_path: Path, content: str) -> List[ReviewFinding]:
        """Run all bad practice checks on a Python file.

        Args:
            file_path: Path to the file (for reporting).
            content: Source code content.

        Returns:
            List of findings from all checks.
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return []

        findings: List[ReviewFinding] = []
        lines = content.splitlines()

        findings.extend(self._check_magic_numbers(tree, file_path, lines))
        findings.extend(self._check_bare_except(tree, file_path))
        findings.extend(self._check_mutable_defaults(tree, file_path))
        findings.extend(self._check_unused_imports(tree, file_path, content))
        findings.extend(self._check_missing_type_hints(tree, file_path))

        return findings

    def _check_magic_numbers(
        self, tree: ast.AST, file_path: Path, lines: List[str]
    ) -> List[ReviewFinding]:
        """Detect magic numbers outside of constant assignments."""
        findings: List[ReviewFinding] = []

        # Pre-pass: collect node IDs that are safe to ignore
        safe_ids: Set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Function parameter defaults (including keyword-only defaults)
                for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
                    for c in ast.walk(d):
                        if isinstance(c, ast.Constant):
                            safe_ids.add(id(c))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                # Annotated assignments: dataclass/Pydantic field defaults
                # e.g. temperature: float = 0.3
                for c in ast.walk(node.value):
                    if isinstance(c, ast.Constant):
                        safe_ids.add(id(c))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, (int, float)):
                continue
            if node.value in _ACCEPTABLE_NUMBERS:
                continue

            # Skip nodes identified in the pre-pass
            if id(node) in safe_ids:
                continue

            lineno = getattr(node, "lineno", None)
            if lineno is None:
                continue

            line = lines[lineno - 1].strip() if lineno <= len(lines) else ""

            # Skip constant assignments (UPPER_CASE = value)
            if "=" in line:
                lhs = line.split("=")[0].strip()
                if lhs.isupper() or (lhs.replace("_", "").isupper() and "_" in lhs):
                    continue

            # Skip comment/def/class lines
            if line.startswith("#") or line.startswith("def ") or line.startswith("class "):
                continue

            findings.append(
                ReviewFinding(
                    severity="suggestion",
                    category="practice",
                    file=file_path,
                    line=lineno,
                    rule_id="PY-MAGIC-NUM",
                    message=f"Magic number {node.value} — consider using a named constant",
                    suggestion="Extract to a named constant for clarity",
                    source="static",
                )
            )

        return findings

    def _check_bare_except(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Detect bare except clauses (except: without exception type)."""
        findings: List[ReviewFinding] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(
                    ReviewFinding(
                        severity="warning",
                        category="practice",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-BARE-EXCEPT",
                        message="Bare except clause catches all exceptions including SystemExit and KeyboardInterrupt",
                        suggestion="Use 'except Exception:' to catch only standard exceptions",
                        source="static",
                    )
                )

        return findings

    def _check_mutable_defaults(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Detect mutable default arguments (def f(x=[]))."""
        findings: List[ReviewFinding] = []
        mutable_types = (ast.List, ast.Dict, ast.Set)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for default in node.args.defaults + node.args.kw_defaults:
                if default is None:
                    continue
                if isinstance(default, mutable_types):
                    findings.append(
                        ReviewFinding(
                            severity="warning",
                            category="practice",
                            file=file_path,
                            line=node.lineno,
                            rule_id="PY-MUTABLE-DEFAULT",
                            message=f"Mutable default argument in '{node.name}()' — shared across calls",
                            suggestion="Use None as default and create inside the function body",
                            source="static",
                        )
                    )
                elif isinstance(default, ast.Call):
                    # Detect dict(), list(), set() calls as defaults (these are fine)
                    # but also detect custom class instantiation which might be shared
                    pass

        return findings

    def _check_unused_imports(
        self, tree: ast.AST, file_path: Path, content: str
    ) -> List[ReviewFinding]:
        """Detect imports that are never used in the file."""
        findings: List[ReviewFinding] = []

        # Collect line numbers of imports inside `if TYPE_CHECKING:` blocks
        # (these are legitimate for avoiding circular imports / type-only usage)
        type_checking_linenos: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                is_tc = (
                    (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                    or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
                )
                if is_tc:
                    for child in ast.walk(node):
                        if isinstance(child, (ast.Import, ast.ImportFrom)):
                            type_checking_linenos.add(child.lineno)

        # Collect all imported names
        imports: List[tuple] = []  # (name, alias, lineno)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if node.lineno in type_checking_linenos:
                    continue
                for alias in node.names:
                    used_name = alias.asname or alias.name.split(".")[0]
                    imports.append((used_name, alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if node.lineno in type_checking_linenos:
                    continue
                if node.module and node.module.split(".")[0] in _BUILTIN_IMPORTS:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    used_name = alias.asname or alias.name
                    imports.append((used_name, alias.name, node.lineno))

        # Collect all used names (excluding import lines themselves)
        used_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # For a.b.c, collect 'a'
                n = node
                while isinstance(n, ast.Attribute):
                    n = n.value
                if isinstance(n, ast.Name):
                    used_names.add(n.id)

        # Check __all__ exports
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    used_names.add(elt.value)

        for used_name, orig_name, lineno in imports:
            if used_name not in used_names:
                findings.append(
                    ReviewFinding(
                        severity="suggestion",
                        category="practice",
                        file=file_path,
                        line=lineno,
                        rule_id="PY-UNUSED-IMPORT",
                        message=f"Unused import: '{orig_name}'",
                        suggestion="Remove the unused import",
                        source="static",
                    )
                )

        return findings

    def _check_missing_type_hints(
        self, tree: ast.AST, file_path: Path
    ) -> List[ReviewFinding]:
        """Detect public functions missing return type hints."""
        findings: List[ReviewFinding] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Skip private/dunder methods
            if node.name.startswith("_"):
                continue

            # Skip test methods — pytest doesn't use return values
            if node.name.startswith("test"):
                continue

            if node.returns is None:
                findings.append(
                    ReviewFinding(
                        severity="suggestion",
                        category="practice",
                        file=file_path,
                        line=node.lineno,
                        rule_id="PY-MISSING-RETURN-TYPE",
                        message=f"Public function '{node.name}()' missing return type hint",
                        suggestion="Add a return type annotation, e.g., '-> None' or '-> str'",
                        source="static",
                    )
                )

        return findings
