# CodeSage

Local-first code intelligence CLI powered by Ollama. Search, analyze, and chat with your codebase using natural language.

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
pipx inject pycodesage "pycodesage[multi-language]"  # JS, TS, Go, Rust
pipx inject pycodesage "pycodesage[mcp]"             # MCP server
```

</details>

## Requirements

**Ollama** must be running:

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen3-embedding
ollama serve
```

## Usage

```bash
cd your-project
codesage init      # Initialize project
codesage index     # Build code index
codesage chat      # Interactive chat mode
```

## Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize project (detects languages, creates config) |
| `index` | Build or update the code index |
| `chat` | Interactive chat with code intelligence |
| `mcp serve` | Start MCP server for AI IDE integration |
| `mcp setup` | Show MCP configuration for your IDE |
| `mcp test` | Test MCP server functionality |

### Chat Commands

Inside `codesage chat`, use these slash commands:

**Search & Analysis**

| Command | Description |
|---------|-------------|
| `/search <query>` | Semantic code search |
| `/deep <query>` | Deep multi-agent analysis |
| `/similar <element>` | Find similar code |
| `/patterns [query]` | Show learned patterns |

**Planning & Review**

| Command | Description |
|---------|-------------|
| `/plan <task>` | Generate implementation plan |
| `/review [file]` | Review code changes |
| `/security [path]` | Security analysis |
| `/impact <element>` | Impact/blast radius analysis |

**Session**

| Command | Description |
|---------|-------------|
| `/mode <mode>` | Switch mode (`brainstorm` / `implement` / `review`) |
| `/context` | Show/modify context settings |
| `/stats` | Show index statistics |
| `/export [file]` | Export conversation |
| `/clear` | Clear chat history |
| `/help` | Show all commands |
| `/exit` or `Ctrl+D` | Exit chat |

## MCP Setup

CodeSage works as an MCP server for AI IDEs. Run `codesage mcp setup` to get the configuration, or add this to your MCP client config:

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
| `list_projects` | List all indexed projects (global mode) |
| `get_developer_profile` | Your coding patterns and conventions |
| `search_code` | Semantic code search with confidence scoring |
| `get_file_context` | File content with definitions and security analysis |
| `review_code` | Code review with static + LLM analysis |
| `analyze_security` | Security vulnerability scanning |
| `get_stats` | Index statistics and storage metrics |
| `explain_concept` | Understand how a concept is implemented |
| `suggest_approach` | Implementation guidance for a coding task |
| `trace_flow` | Trace callers/callees through the dependency graph |
| `find_examples` | Find usage examples of a pattern or function |
| `recommend_pattern` | Pattern recommendations from learned memory |

</details>

<details>
<summary>Client-specific setup</summary>

**Claude Desktop:** Add config above to `claude_desktop_config.json`

**Cursor:** Settings → Features → MCP Servers → Add config

**Windsurf:** Settings → MCP → Add Server. Command: `codesage`, Args: `mcp serve --global`

</details>

## Configuration

Stored in `.codesage/config.yaml` (created by `codesage init`):

```yaml
project_name: my-project
languages:
  - python
  - typescript

llm:
  provider: ollama
  model: qwen2.5-coder:7b
  embedding_model: qwen3-embedding

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
  embedding_model: qwen3-embedding
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
  learn_on_index: true

# Performance tuning
performance:
  embedding_batch_size: 200
  embedding_cache_size: 1000
  cache_enabled: true
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
