"""Extracts relationships from source files for graph storage.

Analyzes code to extract:
- CALLS: Function/method call relationships
- IMPORTS: Import/use/require dependencies
- INHERITS: Class/trait inheritance and implementations
- CONTAINS: Containment (file->class, class->method)

Supports Python (via AST), Rust, Go, JavaScript, and TypeScript (tree-sitter
preferred; regex fallback when tree-sitter is unavailable).
"""

import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from codesage.storage.kuzu_store import CodeNode, CodeRelationship
from codesage.models.code_element import CodeElement
from codesage.utils.language_detector import EXTENSION_TO_LANGUAGE
from codesage.utils.logging import get_logger
from codesage.utils.treesitter_utils import TREESITTER_AVAILABLE as _TS_AVAILABLE, get_parser as _get_ts_parser

logger = get_logger("core.relationship_extractor")

# ---------------------------------------------------------------------------
# Tree-sitter node types for relationship extraction per language
# ---------------------------------------------------------------------------

# Call expression node type per language
_TS_CALL_NODE: Dict[str, str] = {
    "rust": "call_expression",
    "go": "call_expression",
    "javascript": "call_expression",
    "typescript": "call_expression",
}

# Method call node type (separate from call_expression in Rust)
_TS_METHOD_CALL_NODE: Dict[str, Optional[str]] = {
    "rust": "method_call_expression",
    "go": None,
    "javascript": None,
    "typescript": None,
}

# Import/use node types per language
_TS_IMPORT_NODE: Dict[str, Set[str]] = {
    "rust": {"use_declaration"},
    "go": {"import_spec"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
}

# Inheritance/impl node types per language
_TS_INHERITS_NODE: Dict[str, Set[str]] = {
    "rust": {"impl_item"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "go": set(),  # Go uses structural typing, no explicit inheritance
}

# ---------------------------------------------------------------------------
# Language detection (canonical source: language_detector.EXTENSION_TO_LANGUAGE)
# ---------------------------------------------------------------------------

_EXT_TO_LANG: Dict[str, str] = EXTENSION_TO_LANGUAGE  # includes python + all others

# ---------------------------------------------------------------------------
# Per-language keyword sets (call-like patterns that aren't real calls)
# ---------------------------------------------------------------------------

_RUST_KEYWORDS: frozenset = frozenset({
    "if", "while", "for", "match", "return", "let", "mut", "pub", "fn",
    "use", "mod", "impl", "struct", "enum", "trait", "where", "type",
    "const", "static", "extern", "unsafe", "async", "await", "move",
    "loop", "break", "continue", "true", "false", "Some", "None",
    "Ok", "Err", "self", "Self", "super", "crate", "ref", "box",
})
_GO_KEYWORDS: frozenset = frozenset({
    "if", "else", "for", "switch", "case", "return", "var", "func",
    "type", "struct", "interface", "go", "defer", "select", "chan",
    "map", "range", "make", "new", "len", "cap", "append", "copy",
    "delete", "panic", "recover", "print", "println", "true", "false", "nil",
})
_JS_KEYWORDS: frozenset = frozenset({
    "if", "else", "for", "while", "switch", "case", "return", "function",
    "class", "const", "let", "var", "typeof", "instanceof", "new", "delete",
    "void", "throw", "catch", "try", "finally", "import", "export", "default",
    "from", "async", "await", "yield", "get", "set", "static", "true", "false",
    "null", "undefined", "this", "super", "of", "in",
})

_LANG_KEYWORDS: Dict[str, frozenset] = {
    "rust": _RUST_KEYWORDS,
    "go": _GO_KEYWORDS,
    "javascript": _JS_KEYWORDS,
    "typescript": _JS_KEYWORDS,
}

# ---------------------------------------------------------------------------
# Compiled regex patterns per language
# ---------------------------------------------------------------------------

_RUST_IMPORT_RE = re.compile(
    r"^\s*use\s+([\w:{}*,\s]+)\s*;",
    re.MULTILINE,
)
_RUST_IMPL_FOR_RE = re.compile(
    r"impl(?:<[^>]+>)?\s+([\w:]+)\s+for\s+(\w+)"
)
_RUST_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[!(]")
_RUST_METHOD_RE = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

_GO_IMPORT_RE = re.compile(r'"([\w./\-]+)"')
_GO_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_GO_METHOD_RE = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

_JS_IMPORT_RE = re.compile(r'from\s+[\'"]([^\'"]+)[\'"]')
_JS_REQUIRE_RE = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)')
_JS_EXTENDS_RE = re.compile(r"class\s+(\w+)\s+extends\s+(\w+)")
_JS_CALL_RE = re.compile(r"\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(")
_JS_METHOD_RE = re.compile(r"\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(")


def _detect_language(file_path: Path) -> str:
    """Detect language from file extension."""
    return _EXT_TO_LANG.get(file_path.suffix.lower(), "unknown")


def _generate_id(identifier: str) -> str:
    """Generate a stable short ID from a string."""
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Python extractor (AST-based — unchanged from original)
# ---------------------------------------------------------------------------


class RelationshipExtractor:
    """Extracts code relationships from Python source using AST."""

    def __init__(self) -> None:
        self._current_file: Optional[Path] = None
        self._current_scope: List[str] = []
        self._element_map: Dict[str, CodeElement] = {}
        self._name_to_id: Dict[str, str] = {}

    def extract_relationships(
        self,
        file_path: Path,
        code: str,
        elements: List[CodeElement],
    ) -> Tuple[List[CodeNode], List[CodeRelationship]]:
        """Extract all relationships from a Python file."""
        self._current_file = file_path
        self._element_map = {el.id: el for el in elements}
        self._name_to_id = {el.name: el.id for el in elements if el.name}

        nodes: List[CodeNode] = []
        relationships: List[CodeRelationship] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return nodes, relationships

        file_node = self._create_file_node(file_path)
        nodes.append(file_node)

        import_rels, module_nodes = self._extract_imports(tree, file_node.id)
        relationships.extend(import_rels)
        nodes.extend(module_nodes)

        containment_rels = self._extract_containment(tree, file_node.id, elements)
        relationships.extend(containment_rels)

        inheritance_rels = self._extract_inheritance(tree, elements)
        relationships.extend(inheritance_rels)

        call_rels = self._extract_calls(tree, elements)
        relationships.extend(call_rels)

        logger.debug(
            f"Extracted {len(relationships)} relationships from {file_path.name}"
        )
        return nodes, relationships

    def _create_file_node(self, file_path: Path) -> CodeNode:
        file_id = _generate_id(f"file:{file_path}")
        return CodeNode(
            id=file_id,
            name=file_path.name,
            node_type="file",
            file=str(file_path),
            language="python",
        )

    def _create_module_node(self, module_name: str, language: str = "python") -> CodeNode:
        module_id = _generate_id(f"module:{module_name}")
        return CodeNode(
            id=module_id,
            name=module_name,
            node_type="module",
            file="",
            language=language,
        )

    def _extract_imports(
        self,
        tree: ast.AST,
        file_id: str,
    ) -> Tuple[List[CodeRelationship], List[CodeNode]]:
        relationships = []
        module_nodes = []
        seen_modules: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    if module_name not in seen_modules:
                        seen_modules.add(module_name)
                        module_node = self._create_module_node(module_name)
                        module_nodes.append(module_node)
                        relationships.append(CodeRelationship(
                            source_id=file_id,
                            target_id=module_node.id,
                            rel_type="IMPORTS",
                            metadata={"import_type": "module"},
                        ))

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module
                    if module_name not in seen_modules:
                        seen_modules.add(module_name)
                        module_node = self._create_module_node(module_name)
                        module_nodes.append(module_node)
                        import_names = [alias.name for alias in node.names]
                        relationships.append(CodeRelationship(
                            source_id=file_id,
                            target_id=module_node.id,
                            rel_type="IMPORTS",
                            metadata={
                                "import_type": "from",
                                "names": import_names[:5],
                            },
                        ))

        return relationships, module_nodes

    def _extract_containment(
        self,
        tree: ast.AST,
        file_id: str,
        elements: List[CodeElement],
    ) -> List[CodeRelationship]:
        relationships = []
        element_by_name = {el.name: el for el in elements if el.name}

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in element_by_name:
                    element = element_by_name[node.name]
                    relationships.append(CodeRelationship(
                        source_id=file_id,
                        target_id=element.id,
                        rel_type="CONTAINS",
                    ))

            elif isinstance(node, ast.ClassDef):
                if node.name in element_by_name:
                    class_element = element_by_name[node.name]
                    relationships.append(CodeRelationship(
                        source_id=file_id,
                        target_id=class_element.id,
                        rel_type="CONTAINS",
                    ))
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_name = item.name
                            for el in elements:
                                if (
                                    el.name == method_name
                                    and el.type == "method"
                                    and el.line_start == item.lineno
                                ):
                                    relationships.append(CodeRelationship(
                                        source_id=class_element.id,
                                        target_id=el.id,
                                        rel_type="CONTAINS",
                                    ))
                                    break

        return relationships

    def _extract_inheritance(
        self,
        tree: ast.AST,
        elements: List[CodeElement],
    ) -> List[CodeRelationship]:
        relationships = []
        class_elements = {el.name: el for el in elements if el.type == "class" and el.name}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name not in class_elements:
                    continue
                child_element = class_elements[node.name]

                for base in node.bases:
                    base_name = self._get_base_name(base)
                    if base_name and base_name in class_elements:
                        parent_element = class_elements[base_name]
                        relationships.append(CodeRelationship(
                            source_id=child_element.id,
                            target_id=parent_element.id,
                            rel_type="INHERITS",
                        ))
                    elif base_name:
                        parent_id = _generate_id(f"class:{base_name}")
                        relationships.append(CodeRelationship(
                            source_id=child_element.id,
                            target_id=parent_id,
                            rel_type="INHERITS",
                            metadata={"external": True, "base_name": base_name},
                        ))

        return relationships

    def _extract_calls(
        self,
        tree: ast.AST,
        elements: List[CodeElement],
    ) -> List[CodeRelationship]:
        relationships = []
        element_by_name = {el.name: el for el in elements if el.name}
        seen_calls: Set[Tuple[str, str]] = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller_name = node.name
                if caller_name not in element_by_name:
                    continue
                caller_element = element_by_name[caller_name]

                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee_name = self._get_call_name(child)
                        if callee_name and callee_name in element_by_name:
                            callee_element = element_by_name[callee_name]
                            call_key = (caller_element.id, callee_element.id)
                            if call_key not in seen_calls:
                                seen_calls.add(call_key)
                                relationships.append(CodeRelationship(
                                    source_id=caller_element.id,
                                    target_id=callee_element.id,
                                    rel_type="CALLS",
                                    metadata={"call_line": child.lineno},
                                ))

        return relationships

    def _get_base_name(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            return self._get_base_name(node.value)
        return None

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _generate_id(self, identifier: str) -> str:
        return _generate_id(identifier)


# ---------------------------------------------------------------------------
# Multi-language regex-based extractor (Rust / Go / JS / TS)
# ---------------------------------------------------------------------------


class MultiLanguageRelationshipExtractor:
    """Extracts relationships from non-Python source files.

    Covers Rust, Go, JavaScript, and TypeScript.  Uses tree-sitter for
    accurate AST-based extraction when available; falls back to regex-based
    extraction otherwise.

    Tree-sitter advantages over regex:
    - Skips comments and string literals (no false positives)
    - Handles macros (Rust), arrow functions (JS/TS), etc. correctly
    - Knows exact call/method-call distinction
    """

    def extract_relationships(
        self,
        file_path: Path,
        code: str,
        elements: List[CodeElement],
        language: str,
    ) -> Tuple[List[CodeNode], List[CodeRelationship]]:
        """Extract relationships from a non-Python source file."""
        nodes: List[CodeNode] = []
        relationships: List[CodeRelationship] = []

        # File node
        file_id = _generate_id(f"file:{file_path}")
        file_node = CodeNode(
            id=file_id,
            name=file_path.name,
            node_type="file",
            file=str(file_path),
            language=language,
        )
        nodes.append(file_node)

        # CONTAINS: file → all top-level elements
        for el in elements:
            relationships.append(CodeRelationship(
                source_id=file_id,
                target_id=el.id,
                rel_type="CONTAINS",
            ))

        # Try tree-sitter extraction first; fall back to regex on failure
        ts_parser = _get_ts_parser(language) if _TS_AVAILABLE else None
        use_ts = ts_parser is not None

        if use_ts:
            try:
                tree = ts_parser.parse(code.encode("utf-8"))
                import_rels, module_nodes = self._extract_imports_ts(
                    tree, file_id, language
                )
                inherit_rels = self._extract_inheritance_ts(tree, elements, language)
                call_rels = self._extract_calls_ts(tree, elements, language)
                relationships.extend(import_rels)
                nodes.extend(module_nodes)
                relationships.extend(inherit_rels)
                relationships.extend(call_rels)
                logger.debug(
                    f"[{language}] tree-sitter: {len(call_rels)} calls, "
                    f"{len(inherit_rels)} inherits, {len(import_rels)} imports "
                    f"from {file_path.name}"
                )
            except Exception as exc:
                logger.debug(
                    f"[{language}] tree-sitter extraction failed for {file_path.name}: "
                    f"{exc} — falling back to regex"
                )
                use_ts = False

        if not use_ts:
            # Regex-based fallback
            import_rels, module_nodes = self._extract_imports(
                code, file_id, language
            )
            inherit_rels = self._extract_inheritance(code, elements, language)
            call_rels = self._extract_calls(code, elements, language)
            relationships.extend(import_rels)
            nodes.extend(module_nodes)
            relationships.extend(inherit_rels)
            relationships.extend(call_rels)

        logger.debug(
            f"[{language}] Extracted {len(relationships)} relationships "
            f"from {file_path.name}"
        )
        return nodes, relationships

    # ------------------------------------------------------------------
    # Tree-sitter extraction methods
    # ------------------------------------------------------------------

    @staticmethod
    def _ts_walk(node: Any, node_types: Set[str]):
        """DFS generator yielding all nodes of the given types."""
        if node.type in node_types:
            yield node
        for child in node.children:
            yield from MultiLanguageRelationshipExtractor._ts_walk(child, node_types)

    def _extract_calls_ts(
        self,
        tree: Any,
        elements: List[CodeElement],
        language: str,
    ) -> List[CodeRelationship]:
        """Extract CALLS relationships using tree-sitter AST.

        Walks all call_expression (and method_call_expression for Rust) nodes,
        resolves callee names against known elements.  Comment lines and
        string literals are automatically excluded (not in the AST).
        """
        relationships: List[CodeRelationship] = []
        seen: Set[Tuple[str, str]] = set()

        # Build element lookup dicts
        element_by_name: Dict[str, CodeElement] = {
            el.name: el for el in elements if el.name
        }
        # Simple-name fallback: "Struct.method" → first match for "method"
        element_by_simple: Dict[str, CodeElement] = {}
        for el in elements:
            if el.name:
                simple = el.name.split(".")[-1]
                if simple not in element_by_simple:
                    element_by_simple[simple] = el

        # Build caller lookup by line range
        callers_by_range = [
            (el.line_start, el.line_end, el)
            for el in elements
            if el.type in ("function", "method", "constructor") and el.name
        ]

        def find_caller(line: int) -> Optional[CodeElement]:
            for ls, le, el in callers_by_range:
                if ls <= line <= le:
                    return el
            return None

        def _add_call(call_node: Any, callee_name: Optional[str]) -> None:
            if not callee_name:
                return
            line = call_node.start_point[0] + 1
            caller = find_caller(line)
            if caller is None:
                return
            callee = element_by_name.get(callee_name) or element_by_simple.get(callee_name)
            if callee is None or callee.id == caller.id:
                return
            key = (caller.id, callee.id)
            if key in seen:
                return
            seen.add(key)
            relationships.append(CodeRelationship(
                source_id=caller.id,
                target_id=callee.id,
                rel_type="CALLS",
                metadata={"call_line": line, "language": language},
            ))

        # Walk call_expression nodes
        call_node_type = _TS_CALL_NODE.get(language, "call_expression")
        for call_node in self._ts_walk(tree.root_node, {call_node_type}):
            callee_name = self._get_callee_name_ts(call_node, language)
            _add_call(call_node, callee_name)

        # Rust: also walk method_call_expression nodes
        method_call_type = _TS_METHOD_CALL_NODE.get(language)
        if method_call_type:
            for call_node in self._ts_walk(tree.root_node, {method_call_type}):
                # method_call_expression: look for field_identifier or identifier
                callee_name = None
                for child in call_node.children:
                    if child.type in ("identifier", "field_identifier"):
                        callee_name = child.text.decode("utf-8")
                        break
                _add_call(call_node, callee_name)

        return relationships

    def _get_callee_name_ts(self, call_node: Any, language: str) -> Optional[str]:
        """Extract the callee function name from a call_expression node."""
        # Try named field "function" first (most grammars use this)
        fn_field = call_node.child_by_field_name("function")
        if fn_field is not None:
            if fn_field.type == "identifier":
                return fn_field.text.decode("utf-8")
            # Qualified: module.func, obj.method, Struct::method
            for child in reversed(fn_field.children):
                if child.type in (
                    "identifier",
                    "field_identifier",
                    "property_identifier",
                ):
                    return child.text.decode("utf-8")
            return None

        # Fallback: take the first identifier child of the call node
        for child in call_node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _extract_imports_ts(
        self,
        tree: Any,
        file_id: str,
        language: str,
    ) -> Tuple[List[CodeRelationship], List[CodeNode]]:
        """Extract IMPORTS relationships using tree-sitter AST."""
        relationships: List[CodeRelationship] = []
        module_nodes: List[CodeNode] = []
        seen: Set[str] = set()

        def _add_module(module_name: str) -> None:
            if not module_name or module_name in seen:
                return
            seen.add(module_name)
            mid = _generate_id(f"module:{module_name}")
            module_nodes.append(CodeNode(
                id=mid,
                name=module_name,
                node_type="module",
                file="",
                language=language,
            ))
            relationships.append(CodeRelationship(
                source_id=file_id,
                target_id=mid,
                rel_type="IMPORTS",
                metadata={"language": language},
            ))

        import_node_types = _TS_IMPORT_NODE.get(language, set())
        for imp_node in self._ts_walk(tree.root_node, import_node_types):
            if language == "rust":
                # use_declaration: get the path text, take the root crate name
                raw = imp_node.text.decode("utf-8", errors="replace")
                # Strip "use " prefix and trailing ";"
                raw = re.sub(r"^use\s+", "", raw).rstrip(";").strip()
                root = raw.split("::")[0].strip("{} \t")
                if root:
                    _add_module(root)

            elif language == "go":
                # import_spec: path is a string literal
                for child in imp_node.children:
                    if child.type == "interpreted_string_literal":
                        pkg = child.text.decode("utf-8").strip('"\'')
                        name = pkg.split("/")[-1] if pkg else ""
                        if name and not pkg.startswith("."):
                            _add_module(name)

            elif language in ("javascript", "typescript"):
                # import_statement: look for string child (the module path)
                for child in imp_node.children:
                    if child.type in ("string", "string_fragment"):
                        mod = child.text.decode("utf-8").strip("\"'`")
                        name = mod.lstrip("./").split("/")[0] if mod else ""
                        if name:
                            _add_module(name)
                    elif child.type == "string_literal":
                        mod = child.text.decode("utf-8").strip("\"'`")
                        name = mod.lstrip("./").split("/")[0] if mod else ""
                        if name:
                            _add_module(name)

        return relationships, module_nodes

    def _extract_inheritance_ts(
        self,
        tree: Any,
        elements: List[CodeElement],
        language: str,
    ) -> List[CodeRelationship]:
        """Extract INHERITS relationships using tree-sitter AST."""
        relationships: List[CodeRelationship] = []
        element_by_name: Dict[str, CodeElement] = {
            el.name: el for el in elements if el.name
        }

        inherits_node_types = _TS_INHERITS_NODE.get(language, set())
        if not inherits_node_types:
            return relationships

        for node in self._ts_walk(tree.root_node, inherits_node_types):
            if language == "rust" and node.type == "impl_item":
                # impl Trait for Struct → Struct INHERITS Trait
                # Check if this has a "trait" keyword (impl Trait for Type)
                raw = node.text.decode("utf-8", errors="replace")
                m = re.search(r"impl(?:<[^>]+>)?\s+([\w:]+)\s+for\s+(\w+)", raw)
                if m:
                    trait_name = m.group(1).split("::")[-1]
                    struct_name = m.group(2)
                    if struct_name in element_by_name:
                        child_el = element_by_name[struct_name]
                        if trait_name in element_by_name:
                            parent_el = element_by_name[trait_name]
                            relationships.append(CodeRelationship(
                                source_id=child_el.id,
                                target_id=parent_el.id,
                                rel_type="INHERITS",
                            ))
                        else:
                            parent_id = _generate_id(f"trait:{trait_name}")
                            relationships.append(CodeRelationship(
                                source_id=child_el.id,
                                target_id=parent_id,
                                rel_type="INHERITS",
                                metadata={"external": True, "trait": trait_name},
                            ))

            elif language in ("javascript", "typescript") and node.type == "class_declaration":
                # class Child extends Parent
                class_name = None
                parent_name = None
                for child in node.children:
                    if child.type in ("identifier", "type_identifier"):
                        if class_name is None:
                            class_name = child.text.decode("utf-8")
                    elif child.type == "class_heritage":
                        # class_heritage contains the extends clause
                        for sub in child.children:
                            if sub.type in ("identifier", "type_identifier"):
                                parent_name = sub.text.decode("utf-8")
                                break
                    elif child.type == "extends_clause":
                        for sub in child.children:
                            if sub.type in ("identifier", "type_identifier"):
                                parent_name = sub.text.decode("utf-8")
                                break

                if class_name and parent_name and class_name in element_by_name:
                    child_el = element_by_name[class_name]
                    if parent_name in element_by_name:
                        parent_el = element_by_name[parent_name]
                        relationships.append(CodeRelationship(
                            source_id=child_el.id,
                            target_id=parent_el.id,
                            rel_type="INHERITS",
                        ))
                    else:
                        parent_id = _generate_id(f"class:{parent_name}")
                        relationships.append(CodeRelationship(
                            source_id=child_el.id,
                            target_id=parent_id,
                            rel_type="INHERITS",
                            metadata={"external": True, "base": parent_name},
                        ))

        return relationships

    # ------------------------------------------------------------------
    # Original regex-based extraction (kept as fallback)
    # ------------------------------------------------------------------

    # IMPORTS

    def _extract_imports(
        self,
        code: str,
        file_id: str,
        language: str,
    ) -> Tuple[List[CodeRelationship], List[CodeNode]]:
        relationships: List[CodeRelationship] = []
        module_nodes: List[CodeNode] = []
        seen: Set[str] = set()

        def _add_module(module_name: str) -> None:
            if not module_name or module_name in seen:
                return
            seen.add(module_name)
            mid = _generate_id(f"module:{module_name}")
            module_nodes.append(CodeNode(
                id=mid,
                name=module_name,
                node_type="module",
                file="",
                language=language,
            ))
            relationships.append(CodeRelationship(
                source_id=file_id,
                target_id=mid,
                rel_type="IMPORTS",
                metadata={"language": language},
            ))

        if language == "rust":
            for m in _RUST_IMPORT_RE.finditer(code):
                raw = m.group(1).strip()
                # Normalise: take the crate/module root
                root = raw.split("::")[0].strip("{} \t")
                if root:
                    _add_module(root)

        elif language == "go":
            # Find import blocks: import ( "pkg" ... ) or import "pkg"
            in_import = False
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith("import"):
                    in_import = True
                if in_import:
                    for m in _GO_IMPORT_RE.finditer(line):
                        pkg = m.group(1)
                        # Skip non-standard (relative) paths
                        if pkg and not pkg.startswith("."):
                            _add_module(pkg.split("/")[-1])
                    if ")" in stripped:
                        in_import = False

        elif language in ("javascript", "typescript"):
            for m in _JS_IMPORT_RE.finditer(code):
                mod = m.group(1)
                # Skip relative paths — focus on package imports
                name = mod.lstrip("./").split("/")[0] if mod else ""
                if name:
                    _add_module(name)
            for m in _JS_REQUIRE_RE.finditer(code):
                mod = m.group(1)
                name = mod.lstrip("./").split("/")[0] if mod else ""
                if name:
                    _add_module(name)

        return relationships, module_nodes

    # ------------------------------------------------------------------
    # INHERITS
    # ------------------------------------------------------------------

    def _extract_inheritance(
        self,
        code: str,
        elements: List[CodeElement],
        language: str,
    ) -> List[CodeRelationship]:
        relationships: List[CodeRelationship] = []
        element_by_name = {el.name: el for el in elements if el.name}

        if language == "rust":
            # impl Trait for Struct  →  Struct INHERITS Trait
            for m in _RUST_IMPL_FOR_RE.finditer(code):
                trait_name = m.group(1).split("::")[-1]  # strip path prefix
                struct_name = m.group(2)
                if struct_name in element_by_name:
                    child = element_by_name[struct_name]
                    if trait_name in element_by_name:
                        parent = element_by_name[trait_name]
                        relationships.append(CodeRelationship(
                            source_id=child.id,
                            target_id=parent.id,
                            rel_type="INHERITS",
                        ))
                    else:
                        # External trait — create placeholder
                        parent_id = _generate_id(f"trait:{trait_name}")
                        relationships.append(CodeRelationship(
                            source_id=child.id,
                            target_id=parent_id,
                            rel_type="INHERITS",
                            metadata={"external": True, "trait": trait_name},
                        ))

        elif language in ("javascript", "typescript"):
            for m in _JS_EXTENDS_RE.finditer(code):
                child_name = m.group(1)
                parent_name = m.group(2)
                if child_name in element_by_name:
                    child = element_by_name[child_name]
                    if parent_name in element_by_name:
                        parent = element_by_name[parent_name]
                        relationships.append(CodeRelationship(
                            source_id=child.id,
                            target_id=parent.id,
                            rel_type="INHERITS",
                        ))
                    else:
                        parent_id = _generate_id(f"class:{parent_name}")
                        relationships.append(CodeRelationship(
                            source_id=child.id,
                            target_id=parent_id,
                            rel_type="INHERITS",
                            metadata={"external": True, "base": parent_name},
                        ))

        # Go uses interface satisfaction (structural typing) — no explicit
        # inheritance to extract statically.
        return relationships

    # ------------------------------------------------------------------
    # CALLS
    # ------------------------------------------------------------------

    def _extract_calls(
        self,
        code: str,
        elements: List[CodeElement],
        language: str,
    ) -> List[CodeRelationship]:
        """Find function calls within each element's line range."""
        relationships: List[CodeRelationship] = []
        # Primary lookup: full name (e.g. "MoodVector.clamp")
        element_by_name = {el.name: el for el in elements if el.name}
        # Secondary lookup: simple unqualified name (e.g. "clamp" → first match)
        element_by_simple: Dict[str, CodeElement] = {}
        for el in elements:
            if el.name:
                simple = el.name.split(".")[-1]
                if simple not in element_by_simple:
                    element_by_simple[simple] = el

        keywords = _LANG_KEYWORDS.get(language, frozenset())
        seen: Set[Tuple[str, str]] = set()

        if language == "rust":
            call_re = _RUST_CALL_RE
            method_re = _RUST_METHOD_RE
        elif language == "go":
            call_re = _GO_CALL_RE
            method_re = _GO_METHOD_RE
        else:  # javascript / typescript
            call_re = _JS_CALL_RE
            method_re = _JS_METHOD_RE

        lines = code.splitlines()

        # Only look inside function/method elements
        callers = [
            el for el in elements
            if el.type in ("function", "method", "constructor")
            and el.name
        ]

        for caller in callers:
            # Lines that belong to this function body (1-indexed → 0-indexed)
            start = max(0, caller.line_start - 1)
            end = min(len(lines), caller.line_end)
            body_lines = lines[start:end]

            for lineno_offset, line in enumerate(body_lines):
                # Skip comment lines
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("#"):
                    continue

                # Collect all potential callee names on this line
                candidates: Set[str] = set()
                for m in call_re.finditer(line):
                    name = m.group(1)
                    if name and name not in keywords and len(name) > 1:
                        candidates.add(name)
                for m in method_re.finditer(line):
                    name = m.group(1)
                    if name and name not in keywords and len(name) > 1:
                        candidates.add(name)

                for callee_name in candidates:
                    # Try full name first, then simple name
                    callee = element_by_name.get(callee_name) or element_by_simple.get(callee_name)
                    if callee is None:
                        continue
                    if callee.id == caller.id:
                        continue  # Skip self-calls

                    edge_key = (caller.id, callee.id)
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)

                    actual_line = caller.line_start + lineno_offset
                    relationships.append(CodeRelationship(
                        source_id=caller.id,
                        target_id=callee.id,
                        rel_type="CALLS",
                        metadata={"call_line": actual_line, "language": language},
                    ))

        return relationships


# ---------------------------------------------------------------------------
# Cross-file call extraction for non-Python languages
# ---------------------------------------------------------------------------


def extract_cross_file_calls_non_python(
    file_path: Path,
    file_elements: List,
    global_elements: Dict[str, object],
    language: str,
) -> List[CodeRelationship]:
    """Extract cross-file CALLS relationships for non-Python source files.

    Uses the same line-range + regex approach as the same-file pass,
    but matches callees defined in OTHER files.
    """
    try:
        code = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            code = file_path.read_text(encoding="latin-1")
        except Exception:
            return []
    except Exception:
        return []

    keywords = _LANG_KEYWORDS.get(language, frozenset())
    if language == "rust":
        call_re, method_re = _RUST_CALL_RE, _RUST_METHOD_RE
    elif language == "go":
        call_re, method_re = _GO_CALL_RE, _GO_METHOD_RE
    else:
        call_re, method_re = _JS_CALL_RE, _JS_METHOD_RE

    lines = code.splitlines()
    this_file = str(file_path)
    relationships: List[CodeRelationship] = []
    seen: Set[Tuple[str, str]] = set()

    callers = [
        el for el in file_elements
        if el.type in ("function", "method", "constructor") and el.name
    ]

    # Build simple-name lookup for global elements (handles "Type.method" → "method")
    global_by_simple: Dict[str, object] = {}
    for name, el in global_elements.items():
        simple = name.split(".")[-1]
        if simple not in global_by_simple:
            global_by_simple[simple] = el

    for caller in callers:
        start = max(0, caller.line_start - 1)
        end = min(len(lines), caller.line_end)
        body_lines = lines[start:end]

        for lineno_offset, line in enumerate(body_lines):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("#"):
                continue

            candidates: Set[str] = set()
            for m in call_re.finditer(line):
                name = m.group(1)
                if name and name not in keywords and len(name) > 1:
                    candidates.add(name)
            for m in method_re.finditer(line):
                name = m.group(1)
                if name and name not in keywords and len(name) > 1:
                    candidates.add(name)

            for callee_name in candidates:
                callee = global_elements.get(callee_name) or global_by_simple.get(callee_name)
                if callee is None:
                    continue
                if str(getattr(callee, "file", "")) == this_file:
                    continue  # Same-file — already indexed

                edge_key = (caller.id, callee.id)
                if edge_key in seen:
                    continue
                seen.add(edge_key)

                actual_line = caller.line_start + lineno_offset
                relationships.append(CodeRelationship(
                    source_id=caller.id,
                    target_id=callee.id,
                    rel_type="CALLS",
                    metadata={
                        "call_line": actual_line,
                        "cross_file": True,
                        "language": language,
                    },
                ))

    return relationships


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_relationships_from_file(
    file_path: Path,
    elements: List[CodeElement],
) -> Tuple[List[CodeNode], List[CodeRelationship]]:
    """Extract relationships from any supported source file.

    Dispatches to the Python AST extractor for .py files and to the
    regex-based multi-language extractor for Rust/Go/JS/TS files.
    """
    try:
        code = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            code = file_path.read_text(encoding="latin-1")
        except Exception:
            return [], []
    except Exception:
        return [], []

    language = _detect_language(file_path)

    if language == "python":
        extractor = RelationshipExtractor()
        return extractor.extract_relationships(file_path, code, elements)

    elif language in ("rust", "go", "javascript", "typescript"):
        extractor = MultiLanguageRelationshipExtractor()
        return extractor.extract_relationships(file_path, code, elements, language)

    else:
        # Unsupported language — still create a file node for containment
        file_id = _generate_id(f"file:{file_path}")
        file_node = CodeNode(
            id=file_id,
            name=file_path.name,
            node_type="file",
            file=str(file_path),
            language=language,
        )
        nodes = [file_node]
        rels = [
            CodeRelationship(source_id=file_id, target_id=el.id, rel_type="CONTAINS")
            for el in elements
        ]
        return nodes, rels


def extract_cross_file_calls(
    file_path: Path,
    file_elements: List,
    global_elements: Dict[str, object],
) -> List[CodeRelationship]:
    """Extract CALLS relationships that cross file boundaries.

    Dispatches to Python AST or regex-based extraction depending on the
    file's language.  Only returns edges where the callee is defined in a
    DIFFERENT file than the caller.
    """
    language = _detect_language(file_path)

    if language == "python":
        return _extract_cross_file_calls_python(
            file_path, file_elements, global_elements
        )
    elif language in ("rust", "go", "javascript", "typescript"):
        return extract_cross_file_calls_non_python(
            file_path, file_elements, global_elements, language
        )
    return []


def _extract_cross_file_calls_python(
    file_path: Path,
    file_elements: List,
    global_elements: Dict[str, object],
) -> List[CodeRelationship]:
    """Python-specific cross-file call extraction using AST (original logic)."""
    try:
        code = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            code = file_path.read_text(encoding="latin-1")
        except Exception:
            return []
    except Exception:
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    caller_by_name = {el.name: el for el in file_elements if el.name}
    this_file_str = str(file_path)

    relationships: List[CodeRelationship] = []
    seen: Set[Tuple[str, str]] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        caller_name = node.name
        if caller_name not in caller_by_name:
            continue
        caller_el = caller_by_name[caller_name]

        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            callee_name: Optional[str] = None
            func = child.func
            if isinstance(func, ast.Name):
                callee_name = func.id
            elif isinstance(func, ast.Attribute):
                callee_name = func.attr

            if not callee_name or callee_name not in global_elements:
                continue

            callee_el = global_elements[callee_name]
            if str(getattr(callee_el, "file", "")) == this_file_str:
                continue

            edge_key = (caller_el.id, callee_el.id)
            if edge_key in seen:
                continue
            seen.add(edge_key)
            relationships.append(CodeRelationship(
                source_id=caller_el.id,
                target_id=callee_el.id,
                rel_type="CALLS",
                metadata={"call_line": child.lineno, "cross_file": True},
            ))

    return relationships
