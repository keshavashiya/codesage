"""Developer Memory System for CodeSage.

A global personalization layer that learns developer patterns:
- SQLite: Preferences, metrics, interaction history
- LanceDB: Pattern embeddings for semantic search
- KuzuDB: Pattern relationships, cross-project links

Usage:
    from codesage.memory import MemoryManager

    # Basic usage
    memory = MemoryManager()
    memory.add_pattern(pattern, project_name="my_project")
    similar = memory.find_similar_patterns("snake_case naming")

    # Learning from code
    from codesage.memory import LearningEngine
    engine = LearningEngine(memory)
    patterns = engine.learn_from_elements(elements, "my_project")
"""

# Models
from .models import (
    CodeStructure,
    DeveloperPreference,
    InteractionRecord,
    LearnedPattern,
    PatternCategory,
    ProjectInfo,
    RelationshipType,
    StructureType,
)

# Storage
from .preference_store import PreferenceStore
from .memory_manager import MemoryManager

# Try to import optional stores
try:
    from .pattern_store import PatternStore
except ImportError:
    PatternStore = None  # type: ignore

try:
    from .memory_graph import MemoryGraph
except ImportError:
    MemoryGraph = None  # type: ignore

# Learning
from .style_analyzer import StyleAnalyzer, StyleMatch
from .learning_engine import LearningEngine
from .pattern_miner import PatternMiner

__all__ = [
    # Models
    "PatternCategory",
    "StructureType",
    "RelationshipType",
    "LearnedPattern",
    "CodeStructure",
    "ProjectInfo",
    "DeveloperPreference",
    "InteractionRecord",
    # Storage
    "PreferenceStore",
    "PatternStore",
    "MemoryGraph",
    "MemoryManager",
    # Learning
    "StyleAnalyzer",
    "StyleMatch",
    "LearningEngine",
    "PatternMiner",
]
