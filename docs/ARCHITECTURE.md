# CodeSage Architecture

This document describes how CodeSage is structured internally — the major components, how data flows through them, and the reasoning behind the key decisions.

## The big picture

CodeSage has three main jobs:

1. **Index** — parse your source files, embed them, store them in three backends (SQLite, LanceDB, KuzuDB)
2. **Chat** — answer questions about the code using RAG (retrieval-augmented generation) against the index
3. **Review** — analyze diffs for issues using static analysis, security rules, and pattern deviation

These map roughly onto the top-level packages: `core/` handles indexing and search, `chat/` handles the interactive session, and `review/` handles code review.

## Storage

Three databases live in `.codesage/`:

```
.codesage/
├── codesage.db     SQLite: file list, code elements, indexing timestamps
├── lancedb/        LanceDB: vector embeddings (semantic search)
└── kuzudb/         KuzuDB: call graph (who calls whom, imports, inheritance)
```

`StorageManager` in `storage/manager.py` is the single access point for all three. Everything else talks to `StorageManager`, not to the individual stores.

**SQLite** (`storage/database.py`) is the source of truth for what's indexed. Whenever the indexer processes a file, it writes `indexed_files` and `code_elements` rows first. The other two databases store derived data — they're rebuilt from what's in SQLite during a full reindex.

**LanceDB** (`storage/lance_store.py`) stores one row per code element with a 1024 or 4096-dimensional embedding vector. Similarity search is the main query path. If you change embedding models (different dimension), the store detects the mismatch on startup and recreates the table automatically.

**KuzuDB** (`storage/kuzu_store.py`) is a graph database for relationships between elements: `CALLS`, `IMPORTS`, `INHERITS`, `CONTAINS`. It powers `trace_flow` (find who calls a function) and the graph-enriched search mode. The graph isn't required for basic search to work — it degrades gracefully if the graph is empty.

## Indexing

`core/indexer.py` runs in two phases:

**Phase 1 — Parse and embed.** For each changed file, the appropriate parser (`parsers/python_parser.py` or `parsers/treesitter_parser.py`) extracts code elements: functions, classes, methods. Each element gets its code, docstring, and line range stored in SQLite, then batched for embedding. Embeddings are written to LanceDB in sub-batches of 32 for progress granularity. The `memory/hooks.py` listener fires after each batch and feeds elements into the learning engine for pattern extraction.

**Phase 2 — Cross-file relationships.** After all files are embedded, `core/relationship_extractor.py` does a second pass to find relationships that span files. For Python files it uses the `ast` module. For Rust/Go/JS/TS it uses tree-sitter if `multi-language` is installed, falling back to regex patterns if not. Relationships are written to KuzuDB. If a source or target node isn't in the graph yet, the relationship is silently skipped (logged at DEBUG level).

The incremental mode compares file mtimes to what's in SQLite and only reprocesses files that have changed. `--full` skips this check and processes everything.

## Parsing

Two parsers:

- `parsers/python_parser.py` — uses Python's built-in `ast` module. Extracts functions, classes, methods with full metadata (decorators, docstrings, complexity estimates).
- `parsers/treesitter_parser.py` — uses tree-sitter for Rust, Go, JavaScript, and TypeScript. The language grammars are optional installs (`tree-sitter-rust` etc.) — if a grammar isn't installed, that file type is skipped.

Both parsers produce `CodeElement` objects (`models/code_element.py`). The element's ID is a SHA-256 hash of the project name + file path + element name, so it's stable across indexing runs and safe to reference from the graph.

## Search and RAG

`core/suggester.py` is what the chat engine calls when it needs to find relevant code. It implements several search strategies:

- **Vector search** — cosine similarity against LanceDB embeddings
- **Keyword search** — term frequency against element names and code
- **RRF fusion** — Reciprocal Rank Fusion merges the above rankings into one result list

The chat engine uses `chat/query_expansion.py` before calling the suggester. Query expansion tries to generate better search terms from the original question, either through LLM (if available and fast enough) or static rules. If expansion fails or times out, it falls back to the raw query.

Search results come back as `Suggestion` objects with similarity scores. The chat engine uses these to build the LLM prompt context — it doesn't pass raw code to the LLM, it selects the top-k most relevant snippets and formats them into the system prompt.

## Chat engine

`chat/engine.py` is the main class. It holds:

- The conversation history
- References to the suggester, LLM provider, and memory manager
- Handlers for each slash command

The engine has two entry points: `process_input()` for non-streaming (used by MCP) and `process_input_stream()` / `stream_message()` for streaming (used by the CLI). Streaming yields `(chunk_type, content)` tuples — the CLI renders these progressively using a Rich spinner for status messages and prints the final response as a Markdown panel.

Slash commands that need LLM output (`/plan`, `/review`, `/deep`, `/security`, etc.) work by returning a `_LLMStreamRequest` object instead of calling the LLM directly. The caller (streaming or non-streaming path) then handles the actual LLM call in the appropriate way. This lets the same handler code work for both paths without duplication.

The engine respects `Config.language` when building prompts — if the project is Rust, `/plan` appends Rust-specific notes (Result/Option, ownership, `#[cfg(test)]`) and filters out Python-specific patterns from suggestions.

## Review pipeline

`review/pipeline.py` coordinates static analysis. It runs in two modes:

- **fast** (default, <5 seconds) — static checks only
- **full** — static checks + semantic similarity search for duplication + LLM synthesis

For each changed file it dispatches to the appropriate checker based on extension:

- `.py` files → Python-specific checkers in `review/checks/` (complexity, structure, naming, python_checks)
- `.rs`, `.go`, `.js`, `.ts` etc. → `TreeSitterReviewChecker` if tree-sitter is installed, `GenericFileChecker` as fallback

The security scanner (`security/scanner.py`) runs across all file types using regex rules organized by category: injection, secrets, crypto, XSS, deserialization, config.

`review/suppression.py` handles `# codesage:ignore` comments and the `.codesageignore` file. Findings that match suppressions are counted but not shown (visible with `--verbose`).

Results come out as `ReviewFinding` objects (`review/models.py`), unified across all check types. The `UnifiedReviewResult` aggregates them with severity counts and drives the exit code (0 = clean, 1 = issues at or above the threshold).

### Review check rules

**Python checks:**

| Rule | What it flags |
|------|--------------|
| `PY-HIGH-COMPLEXITY` | Cyclomatic complexity > 10 |
| `PY-LONG-FUNCTION` | Function > 50 lines |
| `PY-DEEP-NESTING` | Nesting depth > 4 |
| `PY-TOO-MANY-PARAMS` | More than 5 parameters |
| `PY-GOD-CLASS` | Class with > 15 methods |
| `PY-LARGE-CLASS` | Class with > 300 lines |
| `PY-MISSING-RETURN-TYPE` | Public function without return type annotation |
| `PY-MAGIC-NUM` | Unexplained numeric literal |

**Rust/Go/JS/TS checks (AST-based):**

| Rule | What it flags |
|------|--------------|
| `TS-COMPLEXITY` | Cyclomatic complexity > 10 |
| `TS-LONG-FN` | Function > 60 lines |
| `TS-DEEP-NEST` | Nesting depth > 4 |
| `TS-TOO-MANY-PARAMS` | More than 5 parameters |
| `TS-RUST-NAMING` | snake_case / PascalCase / SCREAMING_SNAKE violations |
| `TS-GO-NAMING` | camelCase / PascalCase violations |
| `TS-JS-NAMING` / `TS-TS-NAMING` | camelCase / PascalCase violations |

**Generic fallback (no tree-sitter):**

| Rule | What it flags |
|------|--------------|
| `GEN-LONG-FUNCTION` | Function > 60 lines (regex-detected) |
| `GEN-HIGH-NESTING` | Indentation depth > 5 levels |
| `GEN-LONG-LINE` | Line > 120 characters |
| `GEN-TODO` | TODO / FIXME / HACK / XXX comments |

## Memory and learning

`memory/` contains the developer memory system. During indexing, the learning engine observes the code elements being added and extracts patterns — coding conventions, error handling approaches, naming styles, complexity norms for this specific codebase. These are stored in a separate LanceDB vector store (not the same one as the code elements).

`memory/style_analyzer.py` identifies patterns with named rules like `camelCase_functions`, `rust_result_option`, `jsdoc`, etc. Each pattern has a `languages` filter so Python patterns don't surface in a Rust project and vice versa.

`memory/pattern_miner.py` can cross-reference patterns across multiple indexed projects (the global MCP mode). If you've indexed three different codebases that all use a particular error handling pattern, it can recommend that pattern for a fourth project.

The memory system feeds back into review via `PAT-*` rules — if a function's complexity suddenly spikes above the project's established norm, that's a pattern deviation and gets flagged.

## MCP server

`codesage mcp serve` always runs in global mode via `global_server.py`. It serves all projects indexed on the machine from a single server process, so you configure your IDE once and never touch it again when you start working in a new project.

Internally, `global_server.py` maintains a cache of `CodeSageMCPServer` instances (one per project) and delegates tool calls to the appropriate one based on the `project_name` argument. `server.py` is not a separate user-facing mode — it's the per-project implementation that the global server wraps.

The server supports two transports: stdio (default, process-based, recommended for IDE integration) and SSE over HTTP (for multi-client or remote setups).

Tool results follow a response envelope format:

```json
{
  "confidence": "high",
  "confidence_score": 0.87,
  "narrative": "Found 3 callers of process_chat...",
  "results": [...],
  "suggested_followup": ["What does handle_bridge do?"],
  "metadata": {...}
}
```

The `narrative` is a plain-English summary the AI IDE can read directly. The `results` array has the structured data for programmatic use.

## LLM provider

`llm/provider.py` wraps LangChain's Ollama/OpenAI/Anthropic integrations behind a uniform interface. The two methods the rest of the codebase uses are `chat(messages) -> str` and `stream_chat(messages) -> Iterator[str]`.

`llm/embeddings.py` handles embedding generation. It auto-detects the embedding dimension by probing the model with a short test string, so switching from `nomic-embed-text` (768-dim) to `qwen3-embedding` (4096-dim) just works without any config change — the LanceDB store picks up the new dimension and recreates the table if needed.

## Configuration

`utils/config.py` loads `.codesage/config.yaml` and provides typed access to every setting. The config object is the single object passed around between components — nothing reads environment variables or config files directly except through `Config`.

`Config.language` returns the primary language (e.g. `"rust"`) based on the `languages` list. This is used to filter patterns, adjust prompts, and pick the right parser.

## Directory map

```
codesage/
├── cli/                          Typer CLI — commands and groups
│   ├── commands/                 init, index, chat, review
│   └── groups/                   mcp, hook
├── chat/                         Interactive chat engine
│   ├── engine.py                 Main chat handler, slash commands
│   ├── query_expansion.py        Search term expansion
│   └── context.py                Context window management
├── core/                         Indexing and search
│   ├── indexer.py                Two-phase file indexer
│   ├── suggester.py              Multi-strategy search + RRF fusion
│   ├── relationship_extractor.py Call graph extraction
│   ├── confidence.py             Confidence scoring for search results
│   └── context_provider.py       Implementation context for MCP
├── review/                       Code review pipeline
│   ├── pipeline.py               Orchestrates all checks
│   ├── checks/                   Per-language static checkers
│   │   ├── python_checks.py
│   │   ├── treesitter_checks.py  AST-based (Rust/Go/JS/TS)
│   │   ├── generic_checks.py     Regex fallback
│   │   ├── complexity.py
│   │   ├── naming.py
│   │   └── structure.py
│   ├── models.py                 ReviewFinding, UnifiedReviewResult
│   ├── output.py                 Rich terminal renderer
│   └── suppression.py            codesage:ignore handling
├── security/                     Security scanner
│   ├── scanner.py
│   └── rules/                    injection, secrets, crypto, xss, ...
├── memory/                       Developer memory and pattern learning
│   ├── learning_engine.py
│   ├── style_analyzer.py
│   ├── pattern_miner.py
│   └── memory_manager.py
├── storage/                      Database access layer
│   ├── manager.py                Single access point for all backends
│   ├── database.py               SQLite
│   ├── lance_store.py            LanceDB vector store
│   └── kuzu_store.py             KuzuDB graph store
├── mcp/                          MCP server
│   ├── server.py                 Project-mode server (12 tools)
│   └── global_server.py          Global-mode server
├── llm/                          LLM and embedding wrappers
│   ├── provider.py
│   └── embeddings.py
├── parsers/                      Source file parsers
│   ├── python_parser.py          AST-based Python parser
│   └── treesitter_parser.py      Tree-sitter parser (multi-language)
├── models/                       Shared data models
│   ├── code_element.py
│   ├── suggestion.py
│   └── context.py
├── hooks/                        Git hook installer
│   └── installer.py
└── utils/                        Utility functions
    ├── config.py                 Configuration loader
    ├── language_detector.py      Language detection
    ├── treesitter_utils.py       Shared parser cache
    └── logging.py                Logging configuration
```
