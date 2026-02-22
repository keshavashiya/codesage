"""Learning engine for extracting patterns from code elements.

Analyzes code elements to learn developer patterns and preferences,
then stores them in the memory system.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from codesage.utils.logging import get_logger

from .memory_manager import MemoryManager
from .models import (
    CodeStructure,
    LearnedPattern,
    PatternCategory,
    ProjectInfo,
    StructureType,
)
from .style_analyzer import StyleAnalyzer

logger = get_logger("memory.learning_engine")


class LearningEngine:
    """Extracts and learns coding patterns from code elements.

    Analyzes code elements to:
        - Extract style patterns (naming, docstrings, typing, etc.)
        - Detect code structure preferences
        - Track pattern co-occurrences
        - Learn cross-project patterns
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        """Initialize the learning engine.

        Args:
            memory_manager: Memory manager for storage.
            embedding_fn: Optional embedding function for pattern search.
        """
        self._memory = memory_manager
        self._style_analyzer = StyleAnalyzer()

        if embedding_fn:
            self._memory.set_embedding_fn(embedding_fn)

    def learn_from_elements(
        self,
        elements: List[Dict[str, Any]],
        project_name: str,
        project_path: Optional[Path] = None,
        language: Optional[str] = None,
    ) -> List[LearnedPattern]:
        """Learn patterns from a list of code elements.

        Args:
            elements: List of code element dictionaries.
            project_name: Name of the project.
            project_path: Optional path to the project.
            language: Primary programming language of the elements (default: auto-detected).

        Returns:
            List of learned patterns.
        """
        if not elements:
            return []

        logger.info(f"Learning patterns from {len(elements)} elements in {project_name}")

        # Ensure project is tracked
        self._ensure_project(project_name, project_path, len(elements))

        # Determine language dynamically
        language = language or (elements[0].get("language", "python") if elements else "python")

        # Analyze elements for style patterns (language-aware)
        all_matches = self._style_analyzer.analyze_elements(elements, language=language)

        # Aggregate patterns
        aggregated = self._style_analyzer.aggregate_patterns(all_matches)

        # Convert to LearnedPattern objects
        patterns = self._style_analyzer.to_learned_patterns(
            aggregated, all_matches, min_occurrences=2, min_confidence=0.5
        )

        # Store patterns
        for pattern in patterns:
            self._memory.add_pattern(pattern, project_name)

        # Detect and record co-occurrences
        cooccurrences = self._find_cooccurrences(all_matches)
        for p1_id, p2_id, correlation in cooccurrences:
            self._memory.record_pattern_cooccurrence(p1_id, p2_id, correlation)

        # Extract structure patterns
        structures = self._extract_structure_patterns(elements)
        for structure in structures:
            self._memory.add_structure(structure)

        # Update project stats
        self._memory.preference_store.update_project_stats(
            project_name,
            patterns_learned=len(patterns),
        )

        logger.info(f"Learned {len(patterns)} patterns from {project_name}")
        return patterns

    def _ensure_project(
        self,
        project_name: str,
        project_path: Optional[Path],
        element_count: int,
    ) -> ProjectInfo:
        """Ensure project exists in memory.

        Args:
            project_name: Project name.
            project_path: Project path.
            element_count: Number of elements.

        Returns:
            ProjectInfo object.
        """
        existing = self._memory.get_project(project_name)

        if existing:
            # Update existing project path if changed
            if project_path and existing.path != project_path:
                existing.path = project_path
                self._memory.add_project(existing)

            # Update existing project stats
            self._memory.preference_store.update_project_stats(
                project_name,
                total_elements=element_count,
            )
            return existing

        # Create new project
        project = ProjectInfo.create(
            name=project_name,
            path=project_path or Path.cwd(),
            total_elements=element_count,
        )
        self._memory.add_project(project)
        return project

    def _find_cooccurrences(
        self,
        all_matches: Dict[str, List],
    ) -> List[Tuple[str, str, float]]:
        """Find patterns that co-occur in the same elements.

        Args:
            all_matches: Dictionary of element IDs to style matches.

        Returns:
            List of (pattern1_id, pattern2_id, correlation) tuples.
        """
        # Build pattern -> elements mapping
        pattern_elements: Dict[str, Set[str]] = defaultdict(set)

        for element_id, matches in all_matches.items():
            for match in matches:
                # Generate a consistent pattern ID
                pattern_id = LearnedPattern.create(
                    name=match.pattern_name,
                    category=match.category,
                    description=match.description,
                    pattern_text=match.pattern_text,
                ).id
                pattern_elements[pattern_id].add(element_id)

        # Calculate co-occurrence correlations
        cooccurrences = []
        pattern_ids = list(pattern_elements.keys())

        for i, p1_id in enumerate(pattern_ids):
            for p2_id in pattern_ids[i + 1:]:
                # Calculate Jaccard similarity
                elements1 = pattern_elements[p1_id]
                elements2 = pattern_elements[p2_id]

                intersection = len(elements1 & elements2)
                union = len(elements1 | elements2)

                if union > 0 and intersection > 0:
                    correlation = intersection / union
                    if correlation >= 0.3:  # Minimum threshold
                        cooccurrences.append((p1_id, p2_id, correlation))

        return cooccurrences

    def _extract_structure_patterns(
        self,
        elements: List[Dict[str, Any]],
    ) -> List[CodeStructure]:
        """Extract code structure patterns from elements.

        Args:
            elements: List of code elements.

        Returns:
            List of code structure preferences.
        """
        structures = []

        # Group elements by type and analyze patterns
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for element in elements:
            by_type[element.get("type", "unknown")].append(element)

        # Analyze class hierarchies
        classes = by_type.get("class", [])
        if classes:
            structures.extend(self._analyze_class_structures(classes))

        # Analyze function patterns
        functions = by_type.get("function", []) + by_type.get("method", [])
        if functions:
            structures.extend(self._analyze_function_structures(functions))

        return structures

    def _analyze_class_structures(
        self,
        classes: List[Dict[str, Any]],
    ) -> List[CodeStructure]:
        """Analyze class structure patterns.

        Args:
            classes: List of class elements.

        Returns:
            List of class structure patterns.
        """
        structures = []

        # Check for dataclass usage
        dataclass_count = 0
        for cls in classes:
            code = cls.get("code", "")
            if "@dataclass" in code:
                dataclass_count += 1

        if dataclass_count >= 2:
            structures.append(
                CodeStructure.create(
                    structure_type=StructureType.CLASS_HIERARCHY,
                    name="dataclass_preference",
                    description="Prefers dataclasses for data structures",
                    example_code="@dataclass\nclass MyData:\n    field: str",
                    occurrence_count=dataclass_count,
                    confidence=min(dataclass_count / len(classes), 1.0),
                )
            )

        # Check for abstract base classes
        abc_count = 0
        for cls in classes:
            code = cls.get("code", "")
            if "ABC" in code or "@abstractmethod" in code:
                abc_count += 1

        if abc_count >= 2:
            structures.append(
                CodeStructure.create(
                    structure_type=StructureType.CLASS_HIERARCHY,
                    name="abstract_base_classes",
                    description="Uses abstract base classes for interfaces",
                    example_code="class Base(ABC):\n    @abstractmethod\n    def method(self): ...",
                    occurrence_count=abc_count,
                    confidence=min(abc_count / len(classes), 1.0),
                )
            )

        return structures

    def _analyze_function_structures(
        self,
        functions: List[Dict[str, Any]],
    ) -> List[CodeStructure]:
        """Analyze function structure patterns.

        Args:
            functions: List of function elements.

        Returns:
            List of function structure patterns.
        """
        structures = []

        # Check for decorator patterns
        decorated_count = 0
        decorator_types: Dict[str, int] = defaultdict(int)

        for func in functions:
            code = func.get("code", "")
            lines = code.split("\n")
            for line in lines:
                if line.strip().startswith("@"):
                    decorated_count += 1
                    decorator = line.strip().split("(")[0].lstrip("@")
                    decorator_types[decorator] += 1

        if decorated_count >= 3:
            # Find most common decorator
            most_common = max(decorator_types.items(), key=lambda x: x[1])
            structures.append(
                CodeStructure.create(
                    structure_type=StructureType.CALL_PATTERN,
                    name=f"decorator_pattern_{most_common[0]}",
                    description=f"Uses @{most_common[0]} decorator pattern",
                    example_code=f"@{most_common[0]}\ndef my_function(): ...",
                    occurrence_count=most_common[1],
                    confidence=min(most_common[1] / len(functions), 1.0),
                )
            )

        # Check for context manager usage
        context_manager_count = 0
        for func in functions:
            code = func.get("code", "")
            if "with " in code and ":" in code:
                context_manager_count += 1

        if context_manager_count >= 3:
            structures.append(
                CodeStructure.create(
                    structure_type=StructureType.CALL_PATTERN,
                    name="context_manager_usage",
                    description="Uses context managers for resource handling",
                    example_code="with open(file) as f:\n    data = f.read()",
                    occurrence_count=context_manager_count,
                    confidence=min(context_manager_count / len(functions), 1.0),
                )
            )

        return structures

    def learn_from_file(
        self,
        file_path: Path,
        project_name: str,
        parser_fn: Optional[Callable[[Path], List[Dict[str, Any]]]] = None,
    ) -> List[LearnedPattern]:
        """Learn patterns from a single file.

        Args:
            file_path: Path to the file.
            project_name: Project name.
            parser_fn: Optional function to parse the file into elements.

        Returns:
            List of learned patterns.
        """
        if parser_fn is None:
            logger.warning("No parser function provided, cannot learn from file")
            return []

        try:
            elements = parser_fn(file_path)
            return self.learn_from_elements(
                elements, project_name, file_path.parent
            )
        except Exception as e:
            logger.warning(f"Failed to learn from file {file_path}: {e}")
            return []

    def compute_project_similarity(
        self,
        project1_name: str,
        project2_name: str,
    ) -> float:
        """Compute similarity between two projects based on patterns.

        Args:
            project1_name: First project name.
            project2_name: Second project name.

        Returns:
            Similarity score (0.0-1.0).
        """
        # Get patterns for each project
        p1_patterns = set()
        p2_patterns = set()

        patterns = self._memory.preference_store.get_patterns(limit=1000)
        for pattern in patterns:
            if project1_name in pattern.source_projects:
                p1_patterns.add(pattern.id)
            if project2_name in pattern.source_projects:
                p2_patterns.add(pattern.id)

        if not p1_patterns or not p2_patterns:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(p1_patterns & p2_patterns)
        union = len(p1_patterns | p2_patterns)

        similarity = intersection / union if union > 0 else 0.0

        # Store the similarity relationship
        p1 = self._memory.get_project(project1_name)
        p2 = self._memory.get_project(project2_name)

        if p1 and p2 and similarity > 0.3:
            try:
                self._memory.memory_graph.add_project_similarity(
                    p1.id, p2.id, similarity
                )
            except Exception as e:
                logger.debug(f"Failed to store project similarity: {e}")

        return similarity

    def update_all_project_similarities(self) -> Dict[str, List[Tuple[str, float]]]:
        """Compute and update similarities between all projects.

        Returns:
            Dictionary mapping project names to similar projects.
        """
        projects = self._memory.preference_store.get_all_projects()
        project_names = [p.name for p in projects]

        similarities: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        for i, p1 in enumerate(project_names):
            for p2 in project_names[i + 1:]:
                sim = self.compute_project_similarity(p1, p2)
                if sim > 0.3:
                    similarities[p1].append((p2, sim))
                    similarities[p2].append((p1, sim))

        return dict(similarities)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns.

        Returns:
            Dictionary with learning statistics.
        """
        metrics = self._memory.get_metrics()
        pattern_store = metrics.get("preference_store", {})
        graph = metrics.get("memory_graph", {})

        # Get category breakdown
        category_counts = {}
        for category in PatternCategory:
            count = self._memory.preference_store.get_pattern_count(category)
            if count > 0:
                category_counts[category.value] = count

        return {
            "total_patterns": pattern_store.get("pattern_count", 0),
            "patterns_by_category": category_counts,
            "total_projects": len(self._memory.preference_store.get_all_projects()),
            "graph_nodes": graph.get("node_counts", {}),
            "graph_relationships": graph.get("relationship_counts", {}),
            "cross_project_patterns": len(
                self._memory.get_cross_project_patterns(min_projects=2)
            ),
        }
