"""Prompts and templates for the chat interface."""

# ============================================================================
# Mode-Aware System Prompts
# ============================================================================

BRAINSTORM_PROMPT = """You are CodeSage, a senior software architect helping explore ideas for {project_name}.

Your role is to:
- Explore multiple approaches to problems
- Reference existing patterns in the codebase
- Suggest architectural considerations
- Ask clarifying questions to better understand requirements
- Consider trade-offs between different solutions
- Draw on cross-project insights when relevant

Guidelines:
- Be exploratory and creative
- Present multiple options with pros/cons
- Reference specific files and patterns from the codebase
- Encourage discussion and refinement of ideas
- Don't jump straight to implementation - help think through the problem

Available commands the user can use:
{commands_help}

Codebase context is provided below. Use it to ground your suggestions in reality.
"""

IMPLEMENT_PROMPT = """You are CodeSage, a pair programmer helping implement features in {project_name}.

Your role is to:
- Provide specific, actionable implementation steps
- Reference exact files and functions to modify
- Consider security implications
- Suggest test cases for the implementation
- Warn about potential breaking changes
- Follow existing code patterns and conventions

Guidelines:
- Be specific and practical
- Provide code examples that match existing style
- Reference line numbers and file paths
- Highlight dependencies and affected components
- Consider edge cases and error handling

Available commands the user can use:
{commands_help}

Implementation context is provided below. Be specific and practical.
"""

REVIEW_PROMPT = """You are CodeSage, a code reviewer analyzing changes in {project_name}.

Your role is to:
- Identify bugs and potential issues
- Check for security vulnerabilities
- Verify consistency with existing patterns
- Suggest improvements and best practices
- Prioritize critical issues over minor style concerns

Guidelines:
- Be constructive and actionable
- Explain WHY something is an issue, not just what
- Reference project-specific patterns when relevant
- Distinguish between must-fix and nice-to-have
- Consider performance and maintainability

Available commands the user can use:
{commands_help}

Review context including detected issues is provided below.
"""

# Default prompt (used when no mode is set or for general chat)
CHAT_SYSTEM_PROMPT = """You are CodeSage, an AI assistant specialized in helping developers understand and work with their codebase.

Project: {project_name}
Language: {language}

Your capabilities:
- Search and explain code in the codebase
- Answer questions about how the code works
- Suggest improvements and best practices
- Help debug issues by analyzing relevant code
- Explain code patterns and architecture
- Provide implementation guidance
- Review code for issues

Guidelines:
- Be concise and technical
- Reference specific files and line numbers when discussing code
- If you don't find relevant code, say so
- Suggest using commands if the user needs specific functionality
- Format code examples with proper syntax highlighting

Available commands the user can use:
{commands_help}
"""


# ============================================================================
# Context Templates
# ============================================================================

CONTEXT_TEMPLATE = """
## Relevant Code Context

The following code snippets from the codebase may be relevant to your question:

{code_blocks}
"""

CODE_BLOCK_TEMPLATE = """### {file_path}:{line_start}
**{element_type}**: {name} (Similarity: {similarity:.0%})
{graph_context}
```{language}
{code}
```
"""

SHORT_CONTEXT_TEMPLATE = """
Relevant code from the codebase:
{code_blocks}
"""


# ============================================================================
# Enhanced Context Templates (for different modes)
# ============================================================================

BRAINSTORM_CONTEXT_TEMPLATE = """
## Exploration Context

### Relevant Code
{code_blocks}

### Detected Patterns in This Codebase
{patterns}

### Architectural Considerations
{architecture_notes}

### Similar Approaches in Other Projects
{cross_project_insights}
"""

IMPLEMENT_CONTEXT_TEMPLATE = """
## Implementation Context

### Relevant Code to Reference
{code_blocks}

### Suggested Implementation Steps
{implementation_steps}

### Files Likely to be Affected
{affected_files}

### Dependencies to Consider
{dependencies}

### Security Requirements
{security_notes}

### Suggested Tests
{test_suggestions}
"""

REVIEW_CONTEXT_TEMPLATE = """
## Review Context

### Changes Under Review
{changes}

### Static Analysis Findings
{static_findings}

### Security Scan Results
{security_findings}

### Pattern Deviations
{pattern_deviations}

### Relevant Project Conventions
{conventions}
"""


# ============================================================================
# Help Messages
# ============================================================================

CHAT_HELP = """
## Chat Commands

### Search & Analysis
| Command | Description |
|---------|-------------|
| `/search <query>` | Semantic code search |
| `/deep <query>` | Deep multi-agent analysis |
| `/similar <element>` | Find similar code |
| `/patterns [query]` | Show learned patterns |

### Planning & Review
| Command | Description |
|---------|-------------|
| `/plan <task>` | Generate implementation plan |
| `/review [file]` | Review code changes |
| `/security [path]` | Security analysis |
| `/impact <element>` | Impact/blast radius analysis |

### Session
| Command | Description |
|---------|-------------|
| `/mode <mode>` | Switch mode (brainstorm/implement/review) |
| `/context` | Show context settings |
| `/stats` | Show index statistics |
| `/export [file]` | Export conversation |
| `/clear` | Clear history |
| `/help` | Show this help |
| `/exit` or `Ctrl+D` | Exit chat mode |

## Modes

- **brainstorm** (default): Open-ended exploration and idea generation
- **implement**: Task-focused with specific implementation guidance
- **review**: Code review focused on quality and security

## Tips

- Ask questions in natural language about your code
- Use `/deep` for thorough multi-agent analysis
- Use `/plan` to get a structured implementation approach
- Switch modes with `/mode` based on your current task
"""

CHAT_HELP_SHORT = """
**Commands:** /help, /search, /deep, /plan, /review, /security, /impact, /patterns, /similar, /mode, /stats, /context, /export, /clear, /exit

**Modes:** /mode brainstorm | implement | review

Type `/help` for full command reference.
"""


# ============================================================================
# Search Result Templates
# ============================================================================

SEARCH_RESULTS_TEMPLATE = """
## Search Results for: "{query}"

Found {count} result(s):

{results}
"""

SEARCH_RESULT_ITEM = """{index}. **{file}:{line}** - {name or element_type}
   Similarity: {similarity:.0%}
   ```{language}
   {code_preview}
   ```
"""


# ============================================================================
# Deep Analysis Templates
# ============================================================================

DEEP_ANALYSIS_TEMPLATE = """
## Deep Analysis: "{query}"

### Summary
{summary}

### Key Findings
{findings}

### Affected Components
{components}

### Recommendations
{recommendations}

---
*Analysis depth: {depth} | Components analyzed: {component_count}*

Ask me about any of these findings, or type 'explain <topic>' for more details.
"""


# ============================================================================
# Plan Templates
# ============================================================================

PLAN_TEMPLATE = """
## Implementation Plan: {task}

### Steps
{steps}

### Affected Files
{files}

### Dependencies
{dependencies}

### Security Considerations
{security}

### Test Suggestions
{tests}

---
**Refine this plan:**
- `add <requirement>` to add constraints
- `focus <file>` to focus on specific files
- `security` to add security requirements
- `approve` to finalize
"""


# ============================================================================
# Mode Transition Messages
# ============================================================================

MODE_CHANGED_TEMPLATE = """
**Mode changed to: {mode}**

{mode_description}

{mode_tips}
"""

MODE_DESCRIPTIONS = {
    "brainstorm": "You're now in exploration mode. I'll help you think through problems and explore multiple approaches.",
    "implement": "You're now in implementation mode. I'll provide specific, actionable implementation guidance.",
    "review": "You're now in review mode. I'll focus on code quality, security, and best practices.",
}

MODE_TIPS = {
    "brainstorm": "Try asking 'How would you approach...' or 'What are the trade-offs of...'",
    "implement": "Try '/plan <task>' for structured implementation steps",
    "review": "Try '/review' to review staged changes or '/security' for a security scan",
}
