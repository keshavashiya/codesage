"""Developer Memory System for CodeSage.

A global personalization layer that learns developer patterns:
- SQLite: Preferences, metrics, interaction history
- LanceDB: Pattern embeddings for semantic search
- KuzuDB: Pattern relationships, cross-project links

Usage:
    from codesage.memory import MemoryManager, DeveloperProfileManager

    # Basic usage
    memory = MemoryManager()
    memory.add_pattern(pattern, project_name="my_project")
    similar = memory.find_similar_patterns("snake_case naming")

    # Profile management
    profile = DeveloperProfileManager()
    print(profile.generate_style_guide())

    # Learning from code
    from codesage.memory import LearningEngine
    engine = LearningEngine(memory)
    patterns = engine.learn_from_elements(elements, "my_project")

    # Hooks for indexer integration
    from codesage.memory import get_memory_hooks
    hooks = get_memory_hooks()
    hooks.on_elements_indexed(elements, "my_project")
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

# Profile and Learning
from .profile import DeveloperProfileManager
from .style_analyzer import StyleAnalyzer, StyleMatch
from .learning_engine import LearningEngine
from .pattern_miner import PatternMiner

# Hooks
from .hooks import (
    MemoryHooks,
    get_memory_hooks,
    set_memory_hooks,
    enable_memory_learning,
)

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
    # Profile and Learning
    "DeveloperProfileManager",
    "StyleAnalyzer",
    "StyleMatch",
    "LearningEngine",
    "PatternMiner",
    # Hooks
    "MemoryHooks",
    "get_memory_hooks",
    "set_memory_hooks",
    "enable_memory_learning",
]
