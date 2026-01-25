"""Python code parser using AST."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

from codesage.parsers.base import BaseParser
from codesage.models.code_element import CodeElement
from codesage.utils.logging import get_logger

logger = get_logger("parser.python")


class PythonParser(BaseParser):
    """Parser for Python source files using the built-in AST module."""

    @property
    def language(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> List[str]:
        return [".py"]

    def parse_file(self, file_path: Path) -> List[CodeElement]:
        """Parse a Python file and extract code elements."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return self.parse_code(content, file_path)
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                content = file_path.read_text(encoding="latin-1")
                return self.parse_code(content, file_path)
            except Exception:
                return []
        except Exception as e:
            # Log but don't crash on parse errors
            logger.warning(f"Error parsing {file_path}: {e}")
            return []

    def parse_code(self, code: str, file_path: Path) -> List[CodeElement]:
        """Parse Python code string and extract elements."""
        elements = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Invalid Python syntax
            return elements

        # Extract functions (including async)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                element = self._extract_function(node, code, file_path)
                if element:
                    elements.append(element)
            elif isinstance(node, ast.ClassDef):
                element = self._extract_class(node, code, file_path)
                if element:
                    elements.append(element)

        return elements

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        code: str,
        file_path: Path,
    ) -> Optional[CodeElement]:
        """Extract function information from AST node."""
        try:
            # Get function source code
            func_code = ast.get_source_segment(code, node)
            if not func_code:
                return None

            # Get docstring
            docstring = ast.get_docstring(node)

            # Build signature
            signature = self._build_signature(node)

            # Get parameter names
            params = [arg.arg for arg in node.args.args]

            # Determine type (method vs function)
            is_method = params and params[0] in ("self", "cls")
            elem_type = "method" if is_method else "function"

            return CodeElement.create(
                file=file_path,
                type=elem_type,
                code=func_code,
                language="python",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                name=node.name,
                docstring=docstring,
                signature=signature,
                parameters=params,
            )
        except Exception:
            return None

    def _extract_class(
        self,
        node: ast.ClassDef,
        code: str,
        file_path: Path,
    ) -> Optional[CodeElement]:
        """Extract class information from AST node."""
        try:
            class_code = ast.get_source_segment(code, node)
            if not class_code:
                return None

            docstring = ast.get_docstring(node)

            # Build class signature with bases
            bases = [self._get_name(base) for base in node.bases]
            if bases:
                signature = f"class {node.name}({', '.join(bases)})"
            else:
                signature = f"class {node.name}"

            return CodeElement.create(
                file=file_path,
                type="class",
                code=class_code,
                language="python",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                name=node.name,
                docstring=docstring,
                signature=signature,
            )
        except Exception:
            return None

    def _build_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Build function signature string."""
        parts = []

        # Add async prefix if applicable
        if isinstance(node, ast.AsyncFunctionDef):
            parts.append("async ")

        parts.append(f"def {node.name}(")

        # Build arguments
        args = []

        # Positional-only args (Python 3.8+)
        for arg in node.args.posonlyargs:
            args.append(self._format_arg(arg))

        if node.args.posonlyargs:
            args.append("/")

        # Regular args
        defaults_offset = len(node.args.args) - len(node.args.defaults)
        for i, arg in enumerate(node.args.args):
            arg_str = self._format_arg(arg)
            if i >= defaults_offset:
                arg_str += "=..."
            args.append(arg_str)

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        elif node.args.kwonlyargs:
            args.append("*")

        # Keyword-only args
        kw_defaults_map = {
            i: d for i, d in enumerate(node.args.kw_defaults) if d is not None
        }
        for i, arg in enumerate(node.args.kwonlyargs):
            arg_str = self._format_arg(arg)
            if i in kw_defaults_map:
                arg_str += "=..."
            args.append(arg_str)

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        parts.append(", ".join(args))
        parts.append(")")

        # Return type annotation
        if node.returns:
            parts.append(f" -> {self._get_annotation(node.returns)}")

        return "".join(parts)

    def _format_arg(self, arg: ast.arg) -> str:
        """Format a function argument."""
        if arg.annotation:
            return f"{arg.arg}: {self._get_annotation(arg.annotation)}"
        return arg.arg

    def _get_annotation(self, node: ast.expr) -> str:
        """Get string representation of type annotation."""
        try:
            return ast.unparse(node)
        except Exception:
            return "..."

    def _get_name(self, node: ast.expr) -> str:
        """Get name from AST node (for class bases, etc.)."""
        try:
            return ast.unparse(node)
        except Exception:
            if isinstance(node, ast.Name):
                return node.id
            return "..."
