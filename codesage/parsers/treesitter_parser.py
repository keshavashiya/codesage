"""Tree-sitter based parser for multiple languages.

Supports JavaScript, TypeScript, Go, and Rust.
Requires the tree-sitter package and corresponding language grammars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from codesage.parsers.base import BaseParser
from codesage.models.code_element import CodeElement
from codesage.utils.logging import get_logger

logger = get_logger("parser.treesitter")


# Language configuration: node types and extraction rules
LANGUAGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs", ".cjs"],
        "grammar_module": "tree_sitter_javascript",
        "node_types": {
            "function": [
                "function_declaration",
                "arrow_function",
                "function_expression",
                "generator_function_declaration",
            ],
            "class": ["class_declaration"],
            "method": ["method_definition"],
        },
        "name_field": "name",
        "body_field": "body",
    },
    "typescript": {
        "extensions": [".ts", ".tsx", ".mts", ".cts"],
        "grammar_module": "tree_sitter_typescript",
        "use_tsx": True,  # Use TSX grammar for better JSX support
        "node_types": {
            "function": [
                "function_declaration",
                "arrow_function",
                "function_expression",
                "generator_function_declaration",
            ],
            "class": ["class_declaration"],
            "method": ["method_definition"],
            "interface": ["interface_declaration"],
            "type_alias": ["type_alias_declaration"],
        },
        "name_field": "name",
        "body_field": "body",
    },
    "go": {
        "extensions": [".go"],
        "grammar_module": "tree_sitter_go",
        "node_types": {
            "function": ["function_declaration"],
            "method": ["method_declaration"],
            "struct": ["type_declaration"],
            "interface": ["type_declaration"],
        },
        "name_field": "name",
        "body_field": "body",
    },
    "rust": {
        "extensions": [".rs"],
        "grammar_module": "tree_sitter_rust",
        "node_types": {
            "function": ["function_item"],
            "method": ["function_item"],  # Methods in impl blocks
            "struct": ["struct_item"],
            "enum": ["enum_item"],
            "impl": ["impl_item"],
            "trait": ["trait_item"],
            "mod": ["mod_item"],
        },
        "name_field": "name",
        "body_field": "body",
    },
}


class TreeSitterParser(BaseParser):
    """Multi-language parser using Tree-sitter.

    This parser uses Tree-sitter grammars to parse JavaScript, TypeScript,
    Go, and Rust source files. Each language requires its grammar package
    to be installed (e.g., tree-sitter-javascript).
    """

    def __init__(self, language: str):
        """Initialize parser for a specific language.

        Args:
            language: Language identifier (javascript, typescript, go, rust)

        Raises:
            ValueError: If language is not supported
            ImportError: If required tree-sitter packages are not installed
        """
        if language not in LANGUAGE_CONFIG:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {list(LANGUAGE_CONFIG.keys())}"
            )

        self._language = language
        self._config = LANGUAGE_CONFIG[language]
        self._parser = None
        self._ts_language = None

        # Try to initialize tree-sitter
        self._init_parser()

    def _init_parser(self) -> None:
        """Initialize the tree-sitter parser with language grammar."""
        try:
            import tree_sitter
        except ImportError:
            raise ImportError(
                "tree-sitter package not installed. "
                "Install with: pipx inject pycodesage 'pycodesage[multi-language]' (or pip install 'pycodesage[multi-language]')"
            )

        try:
            if self._language == "typescript" and self._config.get("use_tsx"):
                # TypeScript has separate TS and TSX grammars
                import tree_sitter_typescript
                self._ts_language = tree_sitter.Language(
                    tree_sitter_typescript.language_tsx()
                )
            elif self._language == "typescript":
                import tree_sitter_typescript
                self._ts_language = tree_sitter.Language(
                    tree_sitter_typescript.language_typescript()
                )
            elif self._language == "javascript":
                import tree_sitter_javascript
                self._ts_language = tree_sitter.Language(
                    tree_sitter_javascript.language()
                )
            elif self._language == "go":
                import tree_sitter_go
                self._ts_language = tree_sitter.Language(
                    tree_sitter_go.language()
                )
            elif self._language == "rust":
                import tree_sitter_rust
                self._ts_language = tree_sitter.Language(
                    tree_sitter_rust.language()
                )
            else:
                raise ImportError(f"No grammar loader for {self._language}")

        except ImportError as e:
            raise ImportError(
                f"tree-sitter grammar for {self._language} not installed. "
                f"Install with: pipx inject pycodesage tree-sitter-{self._language} (or pip install tree-sitter-{self._language})"
            ) from e

        # Create parser
        self._parser = tree_sitter.Parser(self._ts_language)

    @property
    def language(self) -> str:
        """Get the language identifier."""
        return self._language

    @property
    def file_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return self._config["extensions"]

    def parse_file(self, file_path: Path) -> List[CodeElement]:
        """Parse a file and extract code elements."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return self.parse_code(content, file_path)
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
                return self.parse_code(content, file_path)
            except Exception:
                return []
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return []

    def parse_code(self, code: str, file_path: Path) -> List[CodeElement]:
        """Parse code string and extract code elements."""
        if not self._parser:
            return []

        elements = []

        try:
            # Parse the code
            tree = self._parser.parse(code.encode("utf-8"))
            root = tree.root_node

            # Walk the tree and extract elements
            self._extract_elements(root, code, file_path, elements)

        except Exception as e:
            logger.warning(f"Error parsing code: {e}")

        return elements

    def _extract_elements(
        self,
        node: Any,
        code: str,
        file_path: Path,
        elements: List[CodeElement],
        parent_name: str = "",
    ) -> None:
        """Recursively extract code elements from AST nodes."""
        node_types = self._config["node_types"]

        # Check if this node is one we want to extract
        for elem_type, type_names in node_types.items():
            if node.type in type_names:
                element = self._create_element(
                    node, code, file_path, elem_type, parent_name
                )
                if element:
                    elements.append(element)
                    # For classes/structs/impls, use their name as parent for methods
                    if elem_type in ("class", "struct", "impl", "interface", "trait"):
                        parent_name = element.name
                break

        # Recurse into children
        for child in node.children:
            self._extract_elements(child, code, file_path, elements, parent_name)

    def _create_element(
        self,
        node: Any,
        code: str,
        file_path: Path,
        elem_type: str,
        parent_name: str = "",
    ) -> Optional[CodeElement]:
        """Create a CodeElement from a tree-sitter node."""
        try:
            # Get the source code for this node
            start_byte = node.start_byte
            end_byte = node.end_byte
            node_code = code[start_byte:end_byte]

            # Get the name
            name = self._get_node_name(node)
            if not name:
                return None

            # Qualify method names with parent
            if parent_name and elem_type in ("method", "function"):
                qualified_name = f"{parent_name}.{name}"
            else:
                qualified_name = name

            # Get docstring/comment
            docstring = self._get_docstring(node, code)

            # Build signature
            signature = self._build_signature(node, code, elem_type)

            # Get line numbers (1-indexed)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1

            return CodeElement.create(
                file=file_path,
                type=elem_type,
                code=node_code,
                language=self._language,
                line_start=line_start,
                line_end=line_end,
                name=qualified_name,
                docstring=docstring,
                signature=signature,
            )

        except Exception as e:
            logger.debug(f"Error creating element: {e}")
            return None

    def _get_node_name(self, node: Any) -> Optional[str]:
        """Extract the name from a node."""
        # Try to find a 'name' child node
        for child in node.children:
            if child.type in ("identifier", "property_identifier", "type_identifier"):
                return child.text.decode("utf-8")
            if child.type == "name":
                return child.text.decode("utf-8")

        # For Go type declarations, look deeper
        if node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    for subchild in child.children:
                        if subchild.type in ("identifier", "type_identifier"):
                            return subchild.text.decode("utf-8")

        # For Rust impl blocks, get the type name
        if node.type == "impl_item":
            for child in node.children:
                if child.type == "type_identifier":
                    return child.text.decode("utf-8")
                if child.type == "generic_type":
                    for subchild in child.children:
                        if subchild.type == "type_identifier":
                            return subchild.text.decode("utf-8")

        return None

    def _get_docstring(self, node: Any, code: str) -> Optional[str]:
        """Extract documentation comment before a node."""
        # Look for comments immediately before this node
        start_line = node.start_point[0]

        if start_line == 0:
            return None

        # Get lines before the node
        lines = code.split("\n")
        doc_lines = []

        # Check the line(s) immediately before
        for i in range(start_line - 1, max(-1, start_line - 10), -1):
            line = lines[i].strip()

            # JavaScript/TypeScript JSDoc
            if line.startswith("*/"):
                # Find start of block comment
                for j in range(i, max(-1, i - 50), -1):
                    doc_lines.insert(0, lines[j].strip())
                    if lines[j].strip().startswith("/**"):
                        break
                break
            # Single line comments
            elif line.startswith("//") or line.startswith("#"):
                doc_lines.insert(0, line.lstrip("/#").strip())
            # Rust doc comments
            elif line.startswith("///"):
                doc_lines.insert(0, line.lstrip("/").strip())
            elif line == "":
                continue
            else:
                break

        if doc_lines:
            return "\n".join(doc_lines).strip()

        return None

    def _build_signature(self, node: Any, code: str, elem_type: str) -> str:
        """Build a signature string for the element."""
        # Get the first line of the node as a basic signature
        start_byte = node.start_byte
        end_byte = node.end_byte
        node_code = code[start_byte:end_byte]

        # For functions/methods, try to get just the declaration line
        lines = node_code.split("\n")
        first_line = lines[0].strip()

        # Clean up the signature
        if elem_type in ("function", "method"):
            # Find the end of the function signature (before body)
            if self._language in ("javascript", "typescript"):
                # Look for { or =>
                for i, line in enumerate(lines):
                    if "{" in line or "=>" in line:
                        sig_lines = lines[:i+1]
                        sig = " ".join(l.strip() for l in sig_lines)
                        # Trim at { or =>
                        if "{" in sig:
                            sig = sig[:sig.index("{")].strip()
                        return sig
            elif self._language == "go":
                # Go functions end signature at {
                if "{" in first_line:
                    return first_line[:first_line.index("{")].strip()
            elif self._language == "rust":
                # Rust functions: fn name(...) -> Type
                if "{" in first_line:
                    return first_line[:first_line.index("{")].strip()

        return first_line
