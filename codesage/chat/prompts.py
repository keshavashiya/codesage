"""Prompts and templates for the chat interface."""

# System prompt for chat conversations
CHAT_SYSTEM_PROMPT = """You are CodeSage, an AI assistant specialized in helping developers understand and work with their codebase.

Project: {project_name}
Language: {language}

Your capabilities:
- Search and explain code in the codebase
- Answer questions about how the code works
- Suggest improvements and best practices
- Help debug issues by analyzing relevant code
- Explain code patterns and architecture

Guidelines:
- Be concise and technical
- Reference specific files and line numbers when discussing code
- If you don't find relevant code, say so
- Suggest using /search command if the user needs to find specific code
- Format code examples with proper syntax highlighting

Available commands the user can use:
- /help - Show available commands
- /search <query> - Search codebase for relevant code
- /context - Show current code context
- /clear - Clear conversation history
- /exit - Exit chat mode
"""

# Template for including code context
CONTEXT_TEMPLATE = """
## Relevant Code Context

The following code snippets from the codebase may be relevant to your question:

{code_blocks}
"""

# Template for a single code block
CODE_BLOCK_TEMPLATE = """### {file_path}:{line_start}
**{element_type}**: {name} (Similarity: {similarity:.0%})
{graph_context}
```{language}
{code}
```
"""

# Short context template for limited space
SHORT_CONTEXT_TEMPLATE = """
Relevant code from the codebase:
{code_blocks}
"""

# Help message for chat commands
CHAT_HELP = """
## Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/search <query>` | Search codebase for code matching query |
| `/context` | Show current code context settings |
| `/clear` | Clear conversation history |
| `/stats` | Show index statistics |
| `/exit` or `Ctrl+D` | Exit chat mode |

## Tips

- Ask questions in natural language about your code
- Reference specific functions or classes by name
- Use /search to find code before asking about it
- The assistant has access to your indexed codebase
"""

# Template for search results display
SEARCH_RESULTS_TEMPLATE = """
## Search Results for: "{query}"

Found {count} result(s):

{results}
"""

# Template for individual search result
SEARCH_RESULT_ITEM = """{index}. **{file}:{line}** - {name or element_type}
   Similarity: {similarity:.0%}
   ```{language}
   {code_preview}
   ```
"""
