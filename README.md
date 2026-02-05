# CodeSage

Local-first code intelligence CLI powered by Ollama. Search your codebase using natural language.

Works with Claude Desktop, Cursor, and Windsurf via MCP.

## Install

```bash
# Recommended: pipx (isolated environment)
pipx install pycodesage

# Or pip
pip install pycodesage
```

<details>
<summary>Detailed installation</summary>

```bash
# macOS
brew install pipx
pipx ensurepath

# Linux/Windows
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# Install with specific Python version
pipx install --python python3.11 pycodesage

# Add optional features
pipx inject pycodesage pycodesage[multi-language]  # JS, TS, Go, Rust
pipx inject pycodesage pycodesage[mcp]             # MCP server
```

</details>

## Requirements

**Ollama** must be running:

```bash
ollama pull qwen2.5-coder:7b
ollama pull mxbai-embed-large
ollama serve
```

## Usage

```bash
cd your-project
codesage init      # Initialize
codesage index     # Build index
codesage search "validate email"   # Search
codesage chat      # Interactive mode
```

## Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize project |
| `index` | Build code index |
| `search QUERY` | Semantic code search |
| `chat` | Interactive chat mode |
| `review` | AI code review |
| `stats` | Show index statistics |

<details>
<summary>Search options</summary>

```bash
codesage search "auth flow" --depth thorough  # Deep analysis
codesage search "api handlers" --patterns     # Include learned patterns
codesage search "database" --context          # Show surrounding code
codesage search "errors" --json               # JSON output
```

</details>

<details>
<summary>Chat commands</summary>

```
/search <query>   Semantic search
/plan <task>      Implementation plan
/deep <query>     Multi-agent analysis
/review [file]    Code review
/security [path]  Security scan
/impact <element> Blast radius analysis
/similar <code>   Find similar patterns
/patterns         Learned patterns
/mode <mode>      Switch mode (brainstorm/implement/review)
/export [file]    Save conversation
/help             Show all commands
```

</details>

<details>
<summary>Other commands</summary>

```bash
# MCP server
codesage mcp serve              # Start server
codesage mcp serve --global     # All indexed projects
codesage mcp test               # Test tools

# Developer profile
codesage profile show           # View profile
codesage profile patterns       # Learned patterns

# Configuration
codesage config features list   # Feature flags
codesage config storage info    # Storage details
codesage config hooks install   # Git pre-commit hook
```

</details>

## MCP Setup

Add to your MCP client config:

```json
{
  "mcpServers": {
    "codesage": {
      "command": "codesage",
      "args": ["mcp", "serve", "--global"]
    }
  }
}
```

<details>
<summary>MCP tools available</summary>

| Tool | Description |
|------|-------------|
| `search_code` | Semantic code search |
| `get_file_context` | File with dependencies |
| `get_task_context` | Implementation guidance |
| `review_code` | Code review |
| `analyze_security` | Vulnerability scan |
| `detect_code_smells` | Pattern deviations |
| `get_stats` | Index statistics |

</details>

<details>
<summary>Client-specific setup</summary>

**Claude Desktop:** Add config above to `claude_desktop_config.json`

**Cursor:** Settings → Features → MCP Servers → Add config

**Windsurf:** Settings → MCP → Add Server. Command: `codesage`, Args: `mcp serve --global`

</details>

## Configuration

Stored in `.codesage/config.yaml`:

```yaml
project_name: my-project
languages:
  - python
  - typescript

llm:
  provider: ollama
  model: qwen2.5-coder:7b
  embedding_model: mxbai-embed-large

exclude_dirs:
  - node_modules
  - venv
  - .git
```

<details>
<summary>All configuration options</summary>

```yaml
# LLM settings
llm:
  provider: ollama          # ollama, openai, anthropic
  model: qwen2.5-coder:7b
  embedding_model: mxbai-embed-large
  base_url: http://localhost:11434
  temperature: 0.3

# Storage
storage:
  vector_backend: lancedb
  use_graph: true

# Security scanning
security:
  enabled: true
  severity_threshold: medium

# Developer memory
memory:
  enabled: true
  global_dir: ~/.codesage/developer
  learn_on_index: true

# Feature flags
features:
  memory: true
  graph_storage: true
  code_smell_detection: false
```

</details>

## Language Support

- **Python** (built-in)
- JavaScript, TypeScript, Go, Rust (with `pycodesage[multi-language]`)

## Development

```bash
git clone https://github.com/keshavashiya/codesage.git
cd codesage
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
