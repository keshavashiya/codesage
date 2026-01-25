"""Prompt templates for code intelligence."""

CODE_SUGGESTION_SYSTEM = """You are a code intelligence assistant. Your job is to help developers understand and find relevant code in their codebase.

When explaining code matches:
- Be concise and direct
- Focus on why the code is relevant to the query
- Mention key features or patterns
- Keep explanations to 1-2 sentences"""

CODE_SUGGESTION_PROMPT = """The user is searching for: "{query}"

This code was found as a potential match (similarity: {similarity:.0%}):

```{language}
{code}
```

In one brief sentence, explain why this code is relevant to what the user is looking for."""

CODE_ANALYSIS_SYSTEM = """You are a code analysis expert. Analyze the provided code and give actionable feedback.

Focus on:
- Code quality and best practices
- Potential improvements
- Design patterns used
- Complexity issues"""

CODE_ANALYSIS_PROMPT = """Analyze this {language} code:

```{language}
{code}
```

Provide a brief analysis covering:
1. What the code does
2. Code quality assessment
3. Suggested improvements (if any)

Be concise and actionable."""

PATTERN_DETECTION_SYSTEM = """You are a code pattern expert. Identify and explain coding patterns in the provided code."""

PATTERN_DETECTION_PROMPT = """Identify any notable patterns in this {language} code:

```{language}
{code}
```

List any design patterns, coding patterns, or conventions you see.
Be specific and concise."""

CODE_REVIEW_SYSTEM = """You are a senior code reviewer. Review the provided code changes and give constructive feedback.

Focus on:
- Correctness and logic issues
- Best practices
- Potential bugs
- Security concerns
- Performance issues"""

CODE_REVIEW_PROMPT = """Review these code changes:

{changes}

Provide a brief code review with:
1. Summary of what the changes do
2. Any issues or concerns
3. Suggested improvements

Be constructive and specific."""
