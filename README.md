# CodeSage

Local-first code intelligence CLI with MCP support for Claude Desktop, Cursor, and Windsurf.

Index your codebase and search it using natural language. Everything runs locally with Ollama.
Supports Python, JavaScript, TypeScript, Go, and Rust.

## Install

### Recommended: pipx

We strongly recommend installing CodeSage with [pipx](https://pypa.github.io/pipx/) to run it in an isolated environment.

1.  **Install pipx** (if not already installed):

    ```bash
    # macOS
    brew install pipx
    pipx ensurepath

    # Linux/Windows
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    ```

2.  **Install CodeSage**:

    ```bash
    # Install Python 3.10+ (Recommended)
    brew install python@3.11

    # Install CodeSage using the specific python version
    pipx install --python python3.11 pycodesage
    ```

    *Note: You can use any installed Python version >= 3.10 (e.g., `python3.10`, `python3.12`).*

    *Note: To add optional features later (e.g., multi-language support), use `pipx inject`:*
    ```bash
    pipx inject pycodesage mcp
    pipx inject pycodesage pycodesage[multi-language]
    ```

### Alternative: pip

You can also install via standard pip, though this may conflict with other packages:

```bash
pip install pycodesage
```

Or from source:

```bash
git clone https://github.com/keshavashiya/codesage.git
cd codesage
pip install -e .
```

## Requirements

**Ollama** must be running with these models:

```bash
ollama pull qwen2.5-coder:7b
ollama pull mxbai-embed-large
ollama serve
```

## Usage

```bash
# Initialize and index your project
cd your-project
codesage init
codesage index

# Search your code
codesage suggest "validate email"

# Check everything is working
codesage health
```

## MCP Setup

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
<summary>Other MCP clients (Cursor, Windsurf)</summary>

**Cursor:** Settings → Features → MCP Servers, add same config.

**Windsurf:** Settings → MCP → Add Server. Command: `codesage`, Args: `mcp serve --global`

</details>

## Commands

```bash
codesage init           # Initialize project
codesage index          # Index codebase
codesage suggest QUERY  # Search code
codesage stats          # Show stats
codesage health         # System check
codesage review         # AI code review
codesage chat           # Interactive mode
```

<details>
<summary>More commands</summary>

```bash
# MCP
codesage mcp serve          # Start server
codesage mcp serve --global # All projects
codesage mcp test           # Test tools

# Security
codesage security scan      # Scan vulnerabilities
codesage hooks install      # Pre-commit hook

# Storage
codesage storage info       # Backend details
codesage storage stats      # Metrics

# Profile
codesage profile show       # Developer profile
codesage profile patterns   # Learned patterns
```

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
  - venv
  - node_modules
  - .git
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
