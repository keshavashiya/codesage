"""SQL schema constants for memory system databases.

Centralizes all SQL schema definitions for:
- PreferenceStore (SQLite)
- MemoryGraph (KuzuDB)

This allows for:
- Easier schema review and modification
- Version tracking of schema changes
- Potential future migration support
"""

# ---------------------------------------------------------------------------
# PreferenceStore Schema (SQLite)
# ---------------------------------------------------------------------------

PREFERENCE_STORE_SCHEMA = """
-- Developer preferences table
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_preferences_category ON preferences(category);

-- Learned patterns table
CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    pattern_text TEXT,
    examples TEXT,
    occurrence_count INTEGER DEFAULT 1,
    confidence_score REAL DEFAULT 0.5,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_patterns_category ON patterns(category);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence_score);

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    language TEXT DEFAULT 'python',
    total_files INTEGER DEFAULT 0,
    total_elements INTEGER DEFAULT 0,
    first_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    patterns_learned INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);

-- Pattern-project links
CREATE TABLE IF NOT EXISTS pattern_projects (
    pattern_id TEXT,
    project_name TEXT,
    linked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (pattern_id, project_name)
);

-- Interaction records
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    interaction_type TEXT NOT NULL,
    project_name TEXT NOT NULL,
    query TEXT,
    response TEXT,
    accepted INTEGER,
    feedback TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_interactions_project ON interactions(project_name);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp);
"""

# ---------------------------------------------------------------------------
# MemoryGraph Schema (KuzuDB)
# ---------------------------------------------------------------------------

# Node tables
KUZU_PATTERN_NODE_SCHEMA = """
CREATE NODE TABLE IF NOT EXISTS PatternNode (
    id STRING,
    name STRING,
    category STRING,
    description STRING,
    pattern_text STRING,
    occurrence_count INT64,
    confidence DOUBLE,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    PRIMARY KEY (id)
)
"""

KUZU_PROJECT_NODE_SCHEMA = """
CREATE NODE TABLE IF NOT EXISTS ProjectNode (
    id STRING,
    name STRING,
    path STRING,
    language STRING,
    total_files INT64,
    total_elements INT64,
    first_indexed TIMESTAMP,
    last_indexed TIMESTAMP,
    PRIMARY KEY (id)
)
"""

KUZU_STRUCTURE_NODE_SCHEMA = """
CREATE NODE TABLE IF NOT EXISTS CodeStructureNode (
    id STRING,
    structure_type STRING,
    name STRING,
    description STRING,
    example_code STRING,
    occurrence_count INT64,
    confidence DOUBLE,
    PRIMARY KEY (id)
)
"""

# Relationship tables
KUZU_COOCCURS_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS CO_OCCURS (
    FROM PatternNode TO PatternNode,
    count INT64,
    correlation DOUBLE
)
"""

KUZU_LEARNED_FROM_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS LEARNED_FROM (
    FROM PatternNode TO ProjectNode,
    first_seen TIMESTAMP,
    occurrences INT64
)
"""

KUZU_SIMILAR_TO_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS SIMILAR_TO (
    FROM ProjectNode TO ProjectNode,
    similarity DOUBLE
)
"""

KUZU_PREFERS_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS PREFERS (
    FROM PatternNode TO CodeStructureNode,
    confidence DOUBLE,
    usage_count INT64
)
"""

KUZU_EVOLVES_TO_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS EVOLVES_TO (
    FROM PatternNode TO PatternNode,
    date TIMESTAMP,
    reason STRING
)
"""

# All KuzuDB schemas in order
KUZU_ALL_SCHEMAS = [
    KUZU_PATTERN_NODE_SCHEMA,
    KUZU_PROJECT_NODE_SCHEMA,
    KUZU_STRUCTURE_NODE_SCHEMA,
    KUZU_COOCCURS_REL_SCHEMA,
    KUZU_LEARNED_FROM_REL_SCHEMA,
    KUZU_SIMILAR_TO_REL_SCHEMA,
    KUZU_PREFERS_REL_SCHEMA,
    KUZU_EVOLVES_TO_REL_SCHEMA,
]
