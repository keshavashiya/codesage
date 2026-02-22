# CodeSage

Local-first code intelligence CLI. Search, analyze, and chat with your codebase using natural language — all running on your machine via Ollama.

## Install

```bash
# macOS
brew install pipx && pipx ensurepath

# Linux
python3 -m pip install --user pipx && python3 -m pipx ensurepath
```

```bash
pipx install pycodesage
```

### Optional features

```bash
# Tree-sitter AST analysis for JS, TS, Go, Rust (recommended)
pipx inject pycodesage "pycodesage[multi-language]"

# MCP server for Claude Desktop / Cursor / Windsurf
pipx inject pycodesage "pycodesage[mcp]"

# Both at once
pipx inject pycodesage "pycodesage[multi-language,mcp]"
```

Without `multi-language`, review checks for non-Python files still run but use text/regex heuristics instead of real AST parsing.

## Requirements

Ollama must be running:

```bash
ollama pull qwen2.5-coder:7b   # LLM
ollama pull nomic-embed-text    # embeddings (fast, lightweight)
ollama serve
```

**Alternatively**, use any Ollama-compatible model. `qwen3-embedding` gives significantly better semantic search if you have the RAM.

## Quick start

```bash
cd your-project
codesage init       # detect languages, write .codesage/config.yaml
codesage index      # parse files, build vector + graph index
codesage chat       # ask questions about your code
codesage review     # review uncommitted changes
```

## Commands

### `codesage init`

Detects languages in the project and creates `.codesage/config.yaml`. Safe to re-run — it won't overwrite an existing config.

```bash
codesage init
codesage init --model llama3.1     # use a different LLM
codesage init --embedding-model qwen3-embedding
```

### `codesage index`

Parses source files, generates embeddings, and builds the call graph. Only processes changed files by default.

```bash
codesage index              # incremental (changed files only)
codesage index --full       # reindex everything from scratch
codesage index --clear      # wipe the index first, then reindex
codesage index --no-learn   # skip pattern learning
```

Index data lives in `.codesage/` — SQLite for metadata, LanceDB for vectors, KuzuDB for the call graph.

### `codesage chat`

Interactive session. Ask anything in plain English, or use slash commands:

```
/search <query>     semantic code search with RRF fusion
/deep <query>       multi-strategy deep analysis
/plan <task>        generate an implementation plan
/similar <name>     find similar functions/classes
/patterns [query]   show learned patterns from this codebase
/review [file]      review code changes with LLM
/security [path]    security vulnerability scan
/impact <name>      blast radius: who calls this, what breaks
/mode <mode>        switch to brainstorm / implement / review
/context            show or adjust context window settings
/stats              index statistics
/export [file]      save conversation to file
/clear              clear chat history
/help               show all commands
```

Natural language questions work too — you don't have to use slash commands.

### `codesage review`

Reviews uncommitted changes. Combines static analysis, pattern deviation detection, and (in full mode) semantic similarity search.

```bash
codesage review                    # all uncommitted changes, fast mode
codesage review --staged           # staged changes only (good for pre-commit)
codesage review --mode full        # add semantic similarity + LLM synthesis
codesage review --severity warning # block on warnings, not just high+critical
codesage review --format json      # JSON output for CI pipelines
codesage review --format sarif     # SARIF for GitHub Advanced Security
codesage review --verbose          # show timing and suppression details
codesage review path/to/subdir     # limit to a subdirectory
```

**What it checks:**

| Category | Rules |
|----------|-------|
| Python static | Long functions, high complexity, deep nesting, too many params, god classes, missing return types, magic numbers |
| Rust/Go/JS/TS static | Cyclomatic complexity, long functions, deep nesting, param count, naming conventions (requires `multi-language`) |
| Security | Hardcoded secrets, SQL injection, eval/exec, unsafe deserialization, weak crypto, XSS sinks |
| Patterns | Deviations from your codebase's own learned patterns |

Suppress a finding inline: `# codesage:ignore GEN-LONG-LINE` or `# codesage:ignore-next-line`.
Suppress a file: add it to `.codesageignore`.

### `codesage hook`

Installs a git pre-commit hook that runs `codesage review --staged` before each commit.

```bash
codesage hook install    # install the hook
codesage hook uninstall  # remove it
codesage hook status     # check if installed
```

The hook blocks commits with findings at `high` severity or above. Bypass when needed: `git commit --no-verify`.

If you use the [pre-commit](https://pre-commit.com) framework instead, this repo includes a `.pre-commit-hooks.yaml`:

```yaml
repos:
  - repo: https://github.com/keshavashiya/codesage
    rev: v0.3.1
    hooks:
      - id: codesage-review
```

### `codesage mcp`

MCP server for AI IDE integration. Always runs in global mode — all projects you've indexed with `codesage index` are available through a single server.

```bash
codesage mcp serve                    # stdio (default, for IDE use)
codesage mcp serve -t sse -p 8080     # HTTP/SSE for multi-client setups
codesage mcp setup                    # print IDE config
codesage mcp test                     # smoke-test all tools
```

## MCP Setup

Run `codesage mcp setup` to get the config, or add this to your IDE:

```json
{
  "mcpServers": {
    "codesage": {
      "command": "codesage",
      "args": ["mcp", "serve"]
    }
  }
}
```

<details>
<summary>Available MCP tools (12)</summary>

| Tool | What it does |
|------|-------------|
| `list_projects` | List all indexed projects (global mode only) |
| `get_developer_profile` | Your coding style and learned patterns |
| `search_code` | Semantic search with confidence scoring |
| `get_file_context` | File content with definitions and security notes |
| `get_stats` | Index stats: files, elements, languages |
| `review_code` | Run a code review on a file or diff |
| `analyze_security` | Security vulnerability scan |
| `explain_concept` | How is X implemented in this codebase? |
| `suggest_approach` | Implementation guidance for a task |
| `trace_flow` | Callers and callees through the call graph |
| `find_examples` | Usage examples for a function or pattern |
| `recommend_pattern` | Patterns from your codebase's memory |

</details>

## Configuration

Created by `codesage init` at `.codesage/config.yaml`. The most useful fields:

```yaml
project_name: my-project
languages:
  - python      # auto-detected

llm:
  provider: ollama
  model: qwen2.5-coder:7b
  embedding_model: nomic-embed-text
  base_url: http://localhost:11434

exclude_dirs:
  - node_modules
  - venv
  - .git
```

<details>
<summary>Full configuration reference</summary>

```yaml
llm:
  provider: ollama          # ollama | openai | anthropic
  model: qwen2.5-coder:7b
  embedding_model: nomic-embed-text
  base_url: http://localhost:11434
  temperature: 0.3
  max_tokens: 500
  request_timeout: 30.0

storage:
  vector_backend: lancedb
  use_graph: true           # enable call graph (KuzuDB)

security:
  enabled: true
  severity_threshold: medium
  block_on_critical: true

memory:
  enabled: true
  learn_on_index: true      # learn patterns during indexing
  min_pattern_confidence: 0.5

performance:
  embedding_batch_size: 200
  embedding_cache_size: 1000
  cache_enabled: true
```

</details>

## Language support

| Language | Indexing | Static review | Call graph |
|----------|----------|---------------|------------|
| Python | built-in | ✓ | ✓ |
| Rust | `multi-language` | ✓ AST-based | ✓ |
| Go | `multi-language` | ✓ AST-based | ✓ |
| TypeScript | `multi-language` | ✓ AST-based | ✓ |
| JavaScript | `multi-language` | ✓ AST-based | ✓ |

Install `pycodesage[multi-language]` for Rust/Go/JS/TS. Without it, those files are still indexed and reviewed using text/regex heuristics.

## Using OpenAI or Anthropic instead of Ollama

```bash
pipx inject pycodesage "pycodesage[openai]"
# or
pipx inject pycodesage "pycodesage[anthropic]"
```

Then set in `.codesage/config.yaml`:

```yaml
llm:
  provider: openai
  model: gpt-4o
```

## Development

```bash
git clone https://github.com/keshavashiya/codesage.git
cd codesage
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev,multi-language,mcp]"
pytest tests/ -v
```

## License

MIT
