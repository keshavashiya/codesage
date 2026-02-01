"""Extracts relationships from Python code for graph storage.

Analyzes AST to extract:
- CALLS: Function/method call relationships
- IMPORTS: Import dependencies
- INHERITS: Class inheritance
- CONTAINS: Containment (file->class, class->method)
"""

import ast
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from codesage.storage.kuzu_store import CodeNode, CodeRelationship
from codesage.models.code_element import CodeElement
from codesage.utils.logging import get_logger

logger = get_logger("core.relationship_extractor")


class RelationshipExtractor:
    """Extracts code relationships from Python AST.

    Analyzes parsed code to identify and extract relationships
    between code elements for storage in the graph database.
    """

    def __init__(self) -> None:
        """Initialize the relationship extractor."""
        self._current_file: Optional[Path] = None
        self._current_scope: List[str] = []  # Stack of current class/function names
        self._element_map: Dict[str, CodeElement] = {}  # id -> element
        self._name_to_id: Dict[str, str] = {}  # qualified_name -> id

    def extract_relationships(
        self,
        file_path: Path,
        code: str,
        elements: List[CodeElement],
    ) -> Tuple[List[CodeNode], List[CodeRelationship]]:
        """Extract all relationships from a Python file.

        Args:
            file_path: Path to the source file.
            code: Source code content.
            elements: Parsed code elements from the file.

        Returns:
            Tuple of (file/module nodes, relationships).
        """
        self._current_file = file_path
        self._element_map = {el.id: el for el in elements}
        self._name_to_id = {el.name: el.id for el in elements if el.name}

        nodes: List[CodeNode] = []
        relationships: List[CodeRelationship] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return nodes, relationships

        # Create file node
        file_node = self._create_file_node(file_path)
        nodes.append(file_node)

        # Extract imports
        import_rels, module_nodes = self._extract_imports(tree, file_node.id)
        relationships.extend(import_rels)
        nodes.extend(module_nodes)

        # Extract containment (file contains classes/functions)
        containment_rels = self._extract_containment(tree, file_node.id, elements)
        relationships.extend(containment_rels)

        # Extract inheritance relationships
        inheritance_rels = self._extract_inheritance(tree, elements)
        relationships.extend(inheritance_rels)

        # Extract call relationships
        call_rels = self._extract_calls(tree, elements)
        relationships.extend(call_rels)

        logger.debug(
            f"Extracted {len(relationships)} relationships from {file_path.name}"
        )

        return nodes, relationships

    def _create_file_node(self, file_path: Path) -> CodeNode:
        """Create a node representing a file.

        Args:
            file_path: Path to the file.

        Returns:
            CodeNode for the file.
        """
        file_id = self._generate_id(f"file:{file_path}")
        return CodeNode(
            id=file_id,
            name=file_path.name,
            node_type="file",
            file=str(file_path),
            language="python",
        )

    def _create_module_node(self, module_name: str) -> CodeNode:
        """Create a node representing an imported module.

        Args:
            module_name: Name of the module.

        Returns:
            CodeNode for the module.
        """
        module_id = self._generate_id(f"module:{module_name}")
        return CodeNode(
            id=module_id,
            name=module_name,
            node_type="module",
            file="",
            language="python",
        )

    def _extract_imports(
        self,
        tree: ast.AST,
        file_id: str,
    ) -> Tuple[List[CodeRelationship], List[CodeNode]]:
        """Extract import relationships from AST.

        Args:
            tree: AST tree.
            file_id: ID of the source file node.

        Returns:
            Tuple of (import relationships, module nodes).
        """
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

                        # Determine import type
                        import_names = [alias.name for alias in node.names]
                        import_type = "from" if import_names else "module"

                        relationships.append(CodeRelationship(
                            source_id=file_id,
                            target_id=module_node.id,
                            rel_type="IMPORTS",
                            metadata={
                                "import_type": import_type,
                                "names": import_names[:5],  # Limit stored names
                            },
                        ))

        return relationships, module_nodes

    def _extract_containment(
        self,
        tree: ast.AST,
        file_id: str,
        elements: List[CodeElement],
    ) -> List[CodeRelationship]:
        """Extract containment relationships.

        Args:
            tree: AST tree.
            file_id: ID of the file node.
            elements: Parsed code elements.

        Returns:
            List of containment relationships.
        """
        relationships = []
        element_by_name = {el.name: el for el in elements if el.name}

        for node in ast.iter_child_nodes(tree):
            # Top-level functions and classes are contained by the file
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

                    # Methods are contained by the class
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Find the method element
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
        """Extract class inheritance relationships.

        Args:
            tree: AST tree.
            elements: Parsed code elements.

        Returns:
            List of inheritance relationships.
        """
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
                        # Internal inheritance (within same file)
                        parent_element = class_elements[base_name]
                        relationships.append(CodeRelationship(
                            source_id=child_element.id,
                            target_id=parent_element.id,
                            rel_type="INHERITS",
                        ))
                    elif base_name:
                        # External inheritance - create a placeholder node ID
                        parent_id = self._generate_id(f"class:{base_name}")
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
        """Extract function call relationships.

        Args:
            tree: AST tree.
            elements: Parsed code elements.

        Returns:
            List of call relationships.
        """
        relationships = []
        element_by_name = {el.name: el for el in elements if el.name}
        seen_calls: Set[Tuple[str, str]] = set()

        # Visit each function/method and find calls within it
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller_name = node.name
                if caller_name not in element_by_name:
                    continue

                caller_element = element_by_name[caller_name]

                # Find all calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee_name = self._get_call_name(child)
                        if callee_name and callee_name in element_by_name:
                            callee_element = element_by_name[callee_name]

                            # Avoid duplicate relationships
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
        """Get the name of a base class from AST node.

        Args:
            node: AST expression node.

        Returns:
            Base class name or None.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle cases like module.ClassName
            return node.attr
        elif isinstance(node, ast.Subscript):
            # Handle cases like Generic[T]
            return self._get_base_name(node.value)
        return None

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get the name of a called function/method.

        Args:
            node: AST Call node.

        Returns:
            Function/method name or None.
        """
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            # Method call like obj.method() - return method name
            return func.attr
        return None

    def _generate_id(self, identifier: str) -> str:
        """Generate a unique ID from an identifier.

        Args:
            identifier: String identifier.

        Returns:
            SHA256 hash (first 16 chars).
        """
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def extract_relationships_from_file(
    file_path: Path,
    elements: List[CodeElement],
) -> Tuple[List[CodeNode], List[CodeRelationship]]:
    """Convenience function to extract relationships from a file.

    Args:
        file_path: Path to the source file.
        elements: Parsed code elements from the file.

    Returns:
        Tuple of (additional nodes, relationships).
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

    extractor = RelationshipExtractor()
    return extractor.extract_relationships(file_path, code, elements)
