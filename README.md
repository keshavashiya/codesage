# CodeSage

**Local-first CLI code intelligence tool with LangChain-powered RAG**

CodeSage indexes your codebase and enables semantic code search using AI. Everything runs locally on your machine with Ollama.

## Features

- **Semantic Code Search** - Find code using natural language queries
- **AI-Powered Suggestions** - Get intelligent code recommendations from your codebase
- **100% Local** - Everything runs on your machine with Ollama, no cloud required
- **Fast Indexing** - Incremental updates only re-index changed files
- **Python Support** - Full Python AST parsing

## Requirements

- **Python 3.9+**
- **Ollama** - Local LLM runtime ([Install Ollama](https://ollama.ai))

### Setting Up Ollama

```bash
# Install Ollama from https://ollama.ai, then pull required models:
ollama pull qwen2.5-coder:7b      # For code analysis (~4.4GB)
ollama pull mxbai-embed-large     # For embeddings (~670MB)
```

## Installation

### From PyPI

```bash
pip install codesage
```

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/keshavashiya/codesage.git
cd codesage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Optional: OpenAI/Anthropic Support

```bash
pip install codesage[openai]      # For OpenAI
pip install codesage[anthropic]   # For Anthropic Claude
```

## Quick Start

```bash
# Navigate to your project
cd your-project

# Initialize CodeSage
codesage init

# Index the codebase
codesage index

# Search for code
codesage suggest "function to validate email"
```

## Commands

### `codesage init [PATH]`

Initialize CodeSage in a project directory.

```bash
codesage init                              # Initialize in current directory
codesage init /path/to/project             # Initialize in specific directory
codesage init --model qwen2.5-coder:7b     # Specify LLM model
codesage init --embedding-model mxbai-embed-large  # Specify embedding model
```

### `codesage index [PATH]`

Index the codebase for semantic search.

```bash
codesage index                # Incremental index (only changed files)
codesage index --full         # Force full reindex
codesage index --clear        # Clear existing index before indexing
```

### `codesage suggest QUERY`

Search for code using natural language.

```bash
codesage suggest "error handling"           # Search with default settings
codesage suggest "database query" -n 10     # Return 10 results (default: 5)
codesage suggest "auth logic" -s 0.5        # Set minimum similarity threshold
codesage suggest "config" --no-explain      # Skip AI explanations (faster)
codesage suggest "utils" -p /path/to/proj   # Search in specific project
```

### `codesage stats`

Show index statistics.

```bash
codesage stats                # Show stats for current directory
codesage stats /path/to/proj  # Show stats for specific project
```

### `codesage health`

Check system health and dependencies.

```bash
codesage health               # Check Ollama, database, disk space
```

### `codesage version`

Show CodeSage version.

```bash
codesage version
```

## Configuration

Configuration is stored in `.codesage/config.yaml`:

```yaml
project_name: my-project
language: python

llm:
  provider: ollama
  model: qwen2.5-coder:7b
  embedding_model: mxbai-embed-large
  base_url: http://localhost:11434
  temperature: 0.3
  max_tokens: 500
  request_timeout: 30.0
  max_retries: 3

storage:
  vector_store: chromadb

exclude_dirs:
  - venv
  - node_modules
  - .git
  - __pycache__
```

### Using OpenAI or Anthropic

```bash
export CODESAGE_API_KEY="your-api-key"
```

```yaml
llm:
  provider: openai           # or anthropic
  model: gpt-4o-mini         # or claude-3-haiku-20240307
```

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black codesage tests

# Lint
ruff check codesage
```

## Troubleshooting

### "Ollama connection failed"

```bash
ollama serve  # Make sure Ollama is running
```

### "Model not found"

```bash
ollama pull qwen2.5-coder:7b
ollama pull mxbai-embed-large
```
