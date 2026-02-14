"""Chat engine for interactive code conversations."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from codesage.llm.provider import LLMProvider
from codesage.storage.database import Database
from codesage.utils.config import Config
from codesage.utils.logging import get_logger
from codesage.core.context_provider import ContextProvider

from .commands import ChatCommand, ChatMode, CommandParser, ParsedCommand
from .context import CodeContextBuilder
from .models import ChatMessage, ChatSession
from .prompts import (
    BRAINSTORM_CONTEXT_TEMPLATE,
    BRAINSTORM_PROMPT,
    CHAT_HELP,
    CHAT_HELP_SHORT,
    CHAT_SYSTEM_PROMPT,
    IMPLEMENT_CONTEXT_TEMPLATE,
    IMPLEMENT_PROMPT,
    MODE_CHANGED_TEMPLATE,
    MODE_DESCRIPTIONS,
    MODE_TIPS,
    PLAN_TEMPLATE,
    REVIEW_CONTEXT_TEMPLATE,
    REVIEW_PROMPT,
)
from .query_expansion import (
    ConversationContext,
    ExpandedQuery,
    QueryExpander,
    UserIntent,
)

logger = get_logger("chat.engine")


@dataclass
class _LLMStreamRequest:
    """Returned by command handlers that want their LLM call streamed.

    Instead of calling self.llm.chat(messages) directly, a handler
    returns this object. The streaming framework then streams the
    LLM response token-by-token.
    """

    messages: List[Dict[str, str]]
    prefix: str = ""  # Text to show before LLM response
    suffix: str = ""  # Text to append after LLM response


class ChatEngine:
    """Main engine for chat interactions.

    Coordinates between user input, code context, and LLM
    to provide an interactive chat experience.

    Supports three modes:
    - brainstorm: Open-ended exploration
    - implement: Task-focused implementation guidance
    - review: Code review and quality analysis
    """

    # Maximum messages to include in LLM context
    MAX_HISTORY_MESSAGES = 20

    def __init__(
        self,
        config: Config,
        include_context: bool = True,
        max_context_results: int = 3,
        initial_mode: str = "brainstorm",
    ):
        """Initialize the chat engine.

        Args:
            config: CodeSage configuration
            include_context: Whether to include code context in prompts
            max_context_results: Maximum code snippets in context
            initial_mode: Starting interaction mode
        """
        self.config = config
        self.include_context = include_context
        self.max_context_results = max_context_results

        # Current interaction mode
        self._mode = ChatMode.BRAINSTORM
        if initial_mode in ("implement", "review"):
            self._mode = ChatMode(initial_mode)

        # Lazy-loaded components
        self._llm: Optional[LLMProvider] = None
        self._context_builder: Optional[CodeContextBuilder] = None
        self._db: Optional[Database] = None
        self._context_provider: Optional[ContextProvider] = None

        # Session management
        self._session = ChatSession(
            project_path=config.project_path,
            project_name=config.project_name,
        )

        # Command parser
        self._parser = CommandParser()

        # Query expansion for better context understanding
        self._query_expander = QueryExpander()
        self._conversation_context = ConversationContext(current_mode=initial_mode)

        # State for follow-up interactions
        self._last_deep_result: Optional[Dict[str, Any]] = None
        self._current_plan: Optional[Any] = None
        self._last_review: Optional[Dict[str, Any]] = None

        # Track discussed files and topics for context continuity
        self._discussed_files: Set[str] = set()
        self._discussed_topics: List[str] = []

        # Add system message
        self._session.add_message(ChatMessage.system(self._build_system_prompt()))

    @property
    def mode(self) -> ChatMode:
        """Get current interaction mode."""
        return self._mode

    @property
    def llm(self) -> LLMProvider:
        """Lazy-load the LLM provider."""
        if self._llm is None:
            self._llm = LLMProvider(self.config.llm)
        return self._llm

    @property
    def context_builder(self) -> CodeContextBuilder:
        """Lazy-load the context builder."""
        if self._context_builder is None:
            self._context_builder = CodeContextBuilder(self.config)
        return self._context_builder

    @property
    def context_provider(self) -> ContextProvider:
        """Lazy-load the context provider."""
        if self._context_provider is None:
            self._context_provider = ContextProvider(self.config)
        return self._context_provider

    @property
    def db(self) -> Database:
        """Lazy-load the database."""
        if self._db is None:
            self._db = Database(self.config.storage.db_path)
        return self._db

    @property
    def session(self) -> ChatSession:
        """Get the current chat session."""
        return self._session

    def _get_commands_help(self) -> str:
        """Get formatted commands help for prompts."""
        return CHAT_HELP_SHORT

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM based on current mode.

        Returns:
            Formatted system prompt string
        """
        commands_help = self._get_commands_help()

        if self._mode == ChatMode.BRAINSTORM:
            return BRAINSTORM_PROMPT.format(
                project_name=self.config.project_name,
                commands_help=commands_help,
            )
        elif self._mode == ChatMode.IMPLEMENT:
            return IMPLEMENT_PROMPT.format(
                project_name=self.config.project_name,
                commands_help=commands_help,
            )
        elif self._mode == ChatMode.REVIEW:
            return REVIEW_PROMPT.format(
                project_name=self.config.project_name,
                commands_help=commands_help,
            )
        else:
            return CHAT_SYSTEM_PROMPT.format(
                project_name=self.config.project_name,
                language=self.config.language,
                commands_help=commands_help,
            )

    def _format_mode_context(self, mode: str, **kwargs) -> str:
        """Format context using mode-specific templates.

        Args:
            mode: The chat mode (brainstorm, implement, review)
            **kwargs: Template variables

        Returns:
            Formatted context string
        """
        templates = {
            "brainstorm": BRAINSTORM_CONTEXT_TEMPLATE,
            "implement": IMPLEMENT_CONTEXT_TEMPLATE,
            "review": REVIEW_CONTEXT_TEMPLATE,
        }
        template = templates.get(mode, BRAINSTORM_CONTEXT_TEMPLATE)
        # Fill missing keys with empty strings
        for key in [
            "code_blocks",
            "patterns",
            "architecture_notes",
            "cross_project_insights",
            "implementation_steps",
            "affected_files",
            "dependencies",
            "security_notes",
            "test_suggestions",
            "changes",
            "static_findings",
            "security_findings",
            "pattern_deviations",
            "conventions",
        ]:
            kwargs.setdefault(key, "")
        return template.format(**kwargs)

    def process_input(self, user_input: str) -> Tuple[str, bool]:
        """Process user input and return response.

        Args:
            user_input: Raw user input string

        Returns:
            Tuple of (response_text, should_continue)
            should_continue is False if user wants to exit
        """
        # Parse the input
        parsed = self._parser.parse(user_input)

        # Handle commands
        handlers = {
            ChatCommand.EXIT: lambda _: ("Goodbye!", False),
            ChatCommand.HELP: lambda _: (self._handle_help(), True),
            ChatCommand.CLEAR: lambda _: (self._handle_clear(), True),
            ChatCommand.STATS: lambda _: (self._handle_stats(), True),
            ChatCommand.CONTEXT: lambda p: (self._handle_context(p.args), True),
            ChatCommand.SEARCH: lambda p: (self._handle_search(p.args), True),
            ChatCommand.UNKNOWN: lambda _: (
                "Unknown command. Type /help for available commands.",
                True,
            ),
            # Enhanced commands
            ChatCommand.DEEP: lambda p: (self._handle_deep(p.args), True),
            ChatCommand.PLAN: lambda p: (self._handle_plan(p.args), True),
            ChatCommand.REVIEW: lambda p: (self._handle_review(p.args), True),
            ChatCommand.SECURITY: lambda p: (self._handle_security(p.args), True),
            ChatCommand.IMPACT: lambda p: (self._handle_impact(p.args), True),
            ChatCommand.PATTERNS: lambda p: (self._handle_patterns(p.args), True),
            ChatCommand.SIMILAR: lambda p: (self._handle_similar(p.args), True),
            ChatCommand.MODE: lambda p: (self._handle_mode(p.args), True),
            ChatCommand.EXPORT: lambda p: (self._handle_export(p.args), True),
        }

        handler = handlers.get(parsed.command)
        if handler:
            result = handler(parsed)
            # Resolve LLM stream requests in non-streaming mode
            if isinstance(result, tuple) and len(result) == 2:
                response, cont = result
                if isinstance(response, _LLMStreamRequest):
                    llm_response = self.llm.chat(response.messages)
                    return response.prefix + llm_response + response.suffix, cont
            return result

        # Regular message - process with LLM
        if parsed.command == ChatCommand.MESSAGE and parsed.args:
            return self._process_message(parsed.args), True

        return "Please enter a message or command.", True

    def _process_message(self, message: str) -> str:
        """Process a regular chat message with query expansion.

        Args:
            message: User message text

        Returns:
            Assistant response text
        """
        # Update conversation context
        self._conversation_context.add_query(message)

        # Expand query for better understanding
        expanded = self._query_expander.expand(message, self._conversation_context)

        # Check if query needs clarification
        if expanded.is_ambiguous and expanded.confidence_score < 0.6:
            clarification_msg = self._generate_clarification_prompt(expanded)
            return clarification_msg

        # Context provider mode: return structured guidance instead of LLM output
        if self.config.features.context_provider_mode:
            # Use expanded query for better context
            context = self.context_provider.get_implementation_context(
                expanded.enhanced_query
            )
            return self.context_provider.to_markdown(context)

        # Add user message to session
        self._session.add_user_message(message)

        # Build context based on mode using EXPANDED query
        context = ""
        code_refs = []
        if self.include_context:
            try:
                # Use enhanced query for context retrieval
                context = self._build_mode_context(expanded.enhanced_query)
                code_refs = self.context_builder.get_code_refs(
                    expanded.enhanced_query, limit=5
                )

                # Track discussed files and topics
                self._update_conversation_context(code_refs, expanded)
            except Exception as e:
                logger.warning(f"Failed to build context: {e}")

        # Build messages for LLM with expanded context awareness
        messages = self._build_llm_messages(context)

        # Call LLM
        try:
            response = self.llm.chat(messages)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Sorry, I encountered an error: {e}"

        # Add assistant response to session
        self._session.add_assistant_message(response, code_refs=code_refs)

        return response

    def _generate_clarification_prompt(self, expanded: ExpandedQuery) -> str:
        """Generate a prompt asking user for clarification.

        Args:
            expanded: Expanded query with ambiguity info

        Returns:
            Clarification prompt string
        """
        lines = [
            "I'm not quite sure I understand what you're looking for. Could you clarify?",
            "",
        ]

        if expanded.ambiguity_hints:
            lines.append(
                f"I noticed these potentially ambiguous terms: {', '.join(expanded.ambiguity_hints[:3])}"
            )
            lines.append("")

        if expanded.suggested_clarifications:
            lines.append("Did you mean:")
            for i, suggestion in enumerate(expanded.suggested_clarifications[:4], 1):
                lines.append(f"{i}. {suggestion}")
            lines.append("")

        lines.append(
            "Or try being more specific, e.g., mention a file name, function, or add more context."
        )

        return "\n".join(lines)

    def _update_conversation_context(
        self, code_refs: List[Any], expanded: ExpandedQuery
    ):
        """Update conversation context with new information.

        Args:
            code_refs: Code references from search
            expanded: Expanded query information
        """
        # Extract files from code references
        files = []
        topics = []

        for ref in code_refs:
            if hasattr(ref, "file"):
                files.append(str(ref.file))
            elif isinstance(ref, dict) and "file" in ref:
                files.append(str(ref["file"]))

            if hasattr(ref, "name") and ref.name:
                topics.append(ref.name)
            elif isinstance(ref, dict) and ref.get("name"):
                topics.append(ref["name"])

        # Add expanded terms as topics
        if expanded.expanded_terms:
            topics.extend(expanded.expanded_terms[:3])

        # Add intent as topic
        if expanded.intent != UserIntent.UNKNOWN:
            topics.append(expanded.intent.name.lower())

        # Update conversation context
        self._conversation_context.add_discussion(files, topics)

        # Update local tracking
        self._discussed_files.update(files)
        self._discussed_topics.extend(topics)
        self._discussed_topics = self._discussed_topics[-20:]  # Keep last 20

    def _build_mode_context(self, query: str) -> str:
        """Build context appropriate for current mode.

        Args:
            query: User query

        Returns:
            Formatted context string
        """
        if self._mode == ChatMode.BRAINSTORM:
            # Rich exploratory context
            return self.context_builder.build_context(
                query,
                limit=self.max_context_results + 2,  # More context for exploration
            )
        elif self._mode == ChatMode.IMPLEMENT:
            # Focused implementation context
            impl_context = self.context_provider.get_implementation_context(query)
            return self.context_provider.to_markdown(impl_context)
        elif self._mode == ChatMode.REVIEW:
            # Review-focused context (code + patterns)
            return self.context_builder.build_context(
                query,
                limit=self.max_context_results,
            )
        else:
            return self.context_builder.build_context(
                query,
                limit=self.max_context_results,
            )

    def _build_llm_messages(self, context: str = "") -> List[Dict[str, str]]:
        """Build message list for LLM API.

        Args:
            context: Optional code context to include

        Returns:
            List of message dicts
        """
        messages = []

        # System message (always first)
        messages.append({"role": "system", "content": self._build_system_prompt()})

        # Get conversation history (excluding system messages)
        history = [msg for msg in self._session.messages if msg.role != "system"]

        # Limit history
        if len(history) > self.MAX_HISTORY_MESSAGES:
            history = history[-self.MAX_HISTORY_MESSAGES :]

        # Add history messages
        for msg in history[:-1]:  # All except last user message
            messages.append(msg.to_dict())

        # For the last user message, include context
        if history and history[-1].role == "user":
            last_msg = history[-1]
            if context:
                content = f"{context}\n\n## User Question\n\n{last_msg.content}"
            else:
                content = last_msg.content
            messages.append({"role": "user", "content": content})

        return messages

    # =========================================================================
    # Streaming Methods
    # =========================================================================

    def stream_message(self, message: str) -> Generator[Tuple[str, str], None, None]:
        """Stream a chat response with progressive context + LLM tokens.

        Yields:
            Tuples of (chunk_type, content):
            - ("context", markdown_str) - Retrieved code context summary
            - ("token", str) - Individual LLM response token
            - ("done", full_response) - Complete response for session recording
        """
        # 1. Query expansion
        yield ("status", "Analyzing your question...")
        self._conversation_context.add_query(message)
        expanded = self._query_expander.expand(message, self._conversation_context)

        if expanded.is_ambiguous and expanded.confidence_score < 0.6:
            yield ("done", self._generate_clarification_prompt(expanded))
            return

        # Context provider mode: return structured guidance (no streaming)
        if self.config.features.context_provider_mode:
            context = self.context_provider.get_implementation_context(
                expanded.enhanced_query
            )
            yield ("done", self.context_provider.to_markdown(context))
            return

        self._session.add_user_message(message)

        # 2. Build context
        yield ("status", "Searching codebase...")
        context = ""
        code_refs = []
        if self.include_context:
            try:
                context = self._build_mode_context(expanded.enhanced_query)
                code_refs = self.context_builder.get_code_refs(
                    expanded.enhanced_query, limit=5
                )
                self._update_conversation_context(code_refs, expanded)
            except Exception as e:
                logger.warning(f"Failed to build context: {e}")

        # 3. Yield context summary so CLI can show it immediately
        if code_refs:
            context_summary = self._format_context_summary(code_refs)
            yield ("context", context_summary)

        # 4. Stream LLM response
        messages = self._build_llm_messages(context)
        full_response = []
        try:
            for token in self.llm.stream_chat(messages):
                full_response.append(token)
                yield ("token", token)
        except Exception:
            # Fallback to non-streaming
            try:
                response = self.llm.chat(messages)
                full_response = [response]
                yield ("token", response)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                full_response = [f"Sorry, I encountered an error: {e}"]
                yield ("token", full_response[0])

        # 5. Record complete response in session
        complete = "".join(full_response)
        self._session.add_assistant_message(complete, code_refs=code_refs)
        yield ("done", complete)

    # Commands that call LLM and should stream their response
    _LLM_COMMANDS = {
        ChatCommand.DEEP,
        ChatCommand.PLAN,
        ChatCommand.REVIEW,
        ChatCommand.SECURITY,
        ChatCommand.IMPACT,
        ChatCommand.PATTERNS,
        ChatCommand.SIMILAR,
    }

    # Status messages shown while gathering context for each command
    _COMMAND_STATUS = {
        ChatCommand.DEEP: "Running deep analysis...",
        ChatCommand.PLAN: "Generating implementation plan...",
        ChatCommand.REVIEW: "Reviewing code changes...",
        ChatCommand.SECURITY: "Running security scan...",
        ChatCommand.IMPACT: "Analyzing impact...",
        ChatCommand.PATTERNS: "Loading patterns...",
        ChatCommand.SIMILAR: "Searching for similar code...",
    }

    def _stream_llm_response(
        self, messages: List[Dict[str, str]], collected: List[str]
    ) -> Generator[Tuple[str, str], None, None]:
        """Stream LLM response tokens. Appends chunks to collected list.

        Args:
            messages: LLM messages to send
            collected: List to append tokens to (for building full response)

        Yields:
            ("token", chunk) tuples
        """
        try:
            for token in self.llm.stream_chat(messages):
                collected.append(token)
                yield ("token", token)
        except Exception:
            # Fallback to non-streaming
            response = self.llm.chat(messages)
            collected.append(response)
            yield ("token", response)

    def process_input_stream(
        self, user_input: str
    ) -> Generator[Tuple[str, str], None, None]:
        """Streaming version of process_input. Yields (chunk_type, content) tuples.

        LLM-calling commands stream their responses progressively.
        Simple commands yield a single ("done", response) tuple.
        Yields ("exit", reason) if exiting.
        """
        parsed = self._parser.parse(user_input)

        # Exit command
        if parsed.command == ChatCommand.EXIT:
            yield ("exit", "Goodbye!")
            return

        # LLM-calling commands: run handler but stream the LLM part
        if parsed.command in self._LLM_COMMANDS:
            yield from self._stream_command(parsed)
            return

        # Non-LLM commands: fast, no streaming needed
        handler = self._get_command_handler(parsed)
        if handler:
            result = handler(parsed)
            yield ("done", result)
            return

        # Regular messages are streamed
        if parsed.command == ChatCommand.MESSAGE and parsed.args:
            yield from self.stream_message(parsed.args)
            return

        yield ("done", "Please enter a message or command.")

    def _stream_command(
        self, parsed: ParsedCommand
    ) -> Generator[Tuple[str, str], None, None]:
        """Stream an LLM-calling command with status updates and token streaming.

        If the handler returns an _LLMStreamRequest, streams the LLM
        response token-by-token. Otherwise yields the result as-is.
        """
        status = self._COMMAND_STATUS.get(parsed.command, "Processing...")
        yield ("status", status)

        handler = self._get_command_handler(parsed)
        if not handler:
            yield ("done", "Unknown command.")
            return

        try:
            result = handler(parsed)

            if isinstance(result, _LLMStreamRequest):
                # Stream the LLM response
                yield ("status", "Generating response...")

                collected = []
                if result.prefix:
                    collected.append(result.prefix)
                    yield ("token", result.prefix)

                yield from self._stream_llm_response(result.messages, collected)

                if result.suffix:
                    collected.append(result.suffix)
                    yield ("token", result.suffix)

                yield ("done", "".join(collected))
            else:
                # Non-LLM result (early return like "No changes found")
                yield ("done", result)
        except Exception as e:
            yield ("done", f"Command failed: {e}")

    def _get_command_handler(self, parsed: ParsedCommand):
        """Get command handler for a parsed command, or None."""
        handlers = {
            ChatCommand.HELP: lambda _: self._handle_help(),
            ChatCommand.CLEAR: lambda _: self._handle_clear(),
            ChatCommand.STATS: lambda _: self._handle_stats(),
            ChatCommand.CONTEXT: lambda p: self._handle_context(p.args),
            ChatCommand.SEARCH: lambda p: self._handle_search(p.args),
            ChatCommand.UNKNOWN: lambda _: "Unknown command. Type /help for available commands.",
            ChatCommand.DEEP: lambda p: self._handle_deep(p.args),
            ChatCommand.PLAN: lambda p: self._handle_plan(p.args),
            ChatCommand.REVIEW: lambda p: self._handle_review(p.args),
            ChatCommand.SECURITY: lambda p: self._handle_security(p.args),
            ChatCommand.IMPACT: lambda p: self._handle_impact(p.args),
            ChatCommand.PATTERNS: lambda p: self._handle_patterns(p.args),
            ChatCommand.SIMILAR: lambda p: self._handle_similar(p.args),
            ChatCommand.MODE: lambda p: self._handle_mode(p.args),
            ChatCommand.EXPORT: lambda p: self._handle_export(p.args),
        }
        return handlers.get(parsed.command)

    def _format_context_summary(self, code_refs: list) -> str:
        """Format a brief context summary for streaming display."""
        lines = ["**Found relevant code:**\n"]
        for ref in code_refs[:5]:
            if hasattr(ref, "file"):
                file_str = str(ref.file)
                line_str = str(getattr(ref, "line", "?"))
                name_str = getattr(ref, "name", "?")
            elif isinstance(ref, dict):
                file_str = ref.get("file", "?")
                line_str = str(ref.get("line", "?"))
                name_str = ref.get("name", "?")
            else:
                continue
            lines.append(f"- `{file_str}:{line_str}` - {name_str}")
        return "\n".join(lines)

    # =========================================================================
    # Core Command Handlers
    # =========================================================================

    def _handle_help(self) -> str:
        """Handle /help command."""
        return CHAT_HELP

    def _handle_clear(self) -> str:
        """Handle /clear command."""
        self._session.clear_history()
        self._session.add_message(ChatMessage.system(self._build_system_prompt()))
        self._last_deep_result = None
        self._current_plan = None
        self._last_review = None
        return "Conversation history cleared."

    def _handle_stats(self) -> str:
        """Handle /stats command."""
        try:
            stats = self.db.get_stats()
            return (
                f"**Index Statistics**\n\n"
                f"- Files indexed: {stats['files']}\n"
                f"- Code elements: {stats['elements']}\n"
                f"- Last indexed: {stats['last_indexed'] or 'Never'}\n"
                f"- Project: {self.config.project_name}\n"
                f"- Current mode: {self._mode.value}"
            )
        except Exception as e:
            return f"Could not get stats: {e}"

    def _handle_context(self, args: Optional[str]) -> str:
        """Handle /context command."""
        if not args:
            return (
                f"**Context Settings**\n\n"
                f"- Code context enabled: {self.include_context}\n"
                f"- Max context results: {self.max_context_results}\n"
                f"- Current mode: {self._mode.value}\n\n"
                f"**Modify:** `/context code on|off` | `/context limit <n>`"
            )

        parts = args.lower().split()
        if len(parts) >= 2:
            setting, value = parts[0], parts[1]

            if setting == "code":
                if value in ("on", "true", "yes", "1"):
                    self.include_context = True
                    return "Code context **enabled**."
                elif value in ("off", "false", "no", "0"):
                    self.include_context = False
                    return "Code context **disabled**."

            elif setting == "limit":
                try:
                    limit = int(value)
                    if 1 <= limit <= 20:
                        self.max_context_results = limit
                        return f"Context limit set to **{limit}**."
                    else:
                        return "Limit must be between 1 and 20."
                except ValueError:
                    return "Invalid number. Use: `/context limit <n>`"

        return "Unknown setting. Use: `/context code on|off` or `/context limit <n>`"

    def _handle_search(self, query: Optional[str]) -> str:
        """Handle /search command with query expansion and multiple strategies."""
        if not query:
            return "Please provide a search query: /search <query>"

        try:
            import re

            # Try multiple query strategies like /deep does
            search_queries = [query]

            # 1. Try lowercase
            if query.lower() != query:
                search_queries.append(query.lower())

            # 2. Try space-separated (for camelCase/PascalCase)
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", query)
            if spaced != query:
                search_queries.append(spaced.lower())

            # 3. Try individual significant terms
            terms = query.replace("_", " ").replace("-", " ").split()
            if len(terms) > 1:
                for term in terms:
                    if len(term) > 3:
                        search_queries.append(term.lower())

            # Perform multiple searches
            all_results = []
            seen_files = set()

            # Search with all query variations
            for sq in search_queries:
                results = self.context_builder.search_code(sq, limit=5)
                for r in results:
                    key = f"{r.get('file')}:{r.get('line')}"
                    if key not in seen_files:
                        seen_files.add(key)
                        all_results.append(r)

            # Also try query expansion for additional results
            expanded = self._query_expander.expand(query, self._conversation_context)
            if expanded.confidence_score >= 0.5 and expanded.expanded_terms:
                expanded_query = " ".join(expanded.expanded_terms[:5])
                expanded_results = self.context_builder.search_code(
                    expanded_query, limit=3
                )
                for r in expanded_results:
                    key = f"{r.get('file')}:{r.get('line')}"
                    if key not in seen_files:
                        seen_files.add(key)
                        all_results.append(r)

            # Re-sort by similarity (highest first)
            all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            results = all_results[:8]  # Limit total results
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            return f"No results found for: {query}"

        # Build output
        output_lines = [f"**Search Results for:** {query}\n"]

        # Show expansion info if helpful
        if expanded.expanded_terms and len(expanded.expanded_terms) > 2:
            output_lines.append(
                f"*Expanded with: {', '.join(expanded.expanded_terms[:4])}*\n"
            )

        output_lines.append("")

        for i, r in enumerate(results, 1):
            name_part = f" ({r['name']})" if r.get("name") else ""
            output_lines.append(
                f"\n**{i}. {r['file']}:{r['line']}**{name_part}\n"
                f"Similarity: {r['similarity']:.0%} | Type: {r['type']}\n"
                f"```{r['language']}\n{r['code'][:300]}{'...' if len(r['code']) > 300 else ''}\n```"
            )

        return "\n".join(output_lines)

    # =========================================================================
    # Enhanced Command Handlers
    # =========================================================================

    def _handle_deep(self, query: Optional[str]) -> str:
        """Handle /deep command for multi-agent analysis."""
        if not query:
            return "Please provide a query: /deep <query>"

        try:
            # Try multiple query strategies to find relevant code
            search_queries = [query]

            # Track best context (most relevant results)
            best_context = None
            best_context_has_exact_match = False

            # Add variations that might help semantic search
            # 1. Try lowercase
            if query.lower() != query:
                search_queries.append(query.lower())

            # 2. Try space-separated (for camelCase/PascalCase)
            import re

            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", query)
            if spaced != query:
                search_queries.append(spaced.lower())

            # 3. Try key terms from query
            terms = query.replace("_", " ").replace("-", " ").split()
            if len(terms) > 1:
                # Add individual significant terms
                for term in terms:
                    if len(term) > 3:  # Skip short terms
                        search_queries.append(term.lower())

            # 4. Try expanded terms from query expander (but only if useful)
            try:
                expanded = self._query_expander.expand(
                    query, self._conversation_context
                )
                if expanded and expanded.expanded_terms:
                    for term in expanded.expanded_terms[:3]:
                        if term.lower() not in [q.lower() for q in search_queries]:
                            search_queries.append(term.lower())
            except Exception:
                pass  # Don't fail if expansion fails

            # Try each query and keep the best result
            for sq in search_queries:
                context = self.context_provider.get_implementation_context(
                    sq,
                    include_cross_project=self.config.features.cross_project_recommendations,
                )

                # Check if we found an exact name match
                has_exact_match = any(
                    ref.name and query.lower() in ref.name.lower()
                    for ref in context.relevant_code
                )

                # Prefer context with exact match, otherwise prefer more results
                if best_context is None:
                    best_context = context
                    best_context_has_exact_match = has_exact_match
                elif has_exact_match and not best_context_has_exact_match:
                    # This context has exact match, use it
                    best_context = context
                    best_context_has_exact_match = True
                elif not has_exact_match and not best_context_has_exact_match:
                    # Neither has exact match, prefer more results
                    if len(context.relevant_code) > len(best_context.relevant_code):
                        best_context = context

            # Use best context found
            context = best_context

            # If nothing found, try a broader search
            if context is None or not context.relevant_code:
                # Try keyword-based search as fallback
                context = self.context_provider.get_implementation_context(
                    f"token bucket {query}",
                    include_cross_project=False,
                )

            # Store for follow-up
            self._last_deep_result = {
                "query": query,
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }

            # Build rich context for LLM
            context_sections = []

            # Add relevant code with more detail
            if context.relevant_code:
                context_sections.append("## Relevant Code Found\n")
                for ref in context.relevant_code[:10]:  # Increased from 5 to 10
                    snippet = (
                        getattr(ref, "snippet", "")
                        or getattr(ref, "content", "")
                        or getattr(ref, "code", "")
                    )
                    # Include full path for clarity
                    file_path = ref.file
                    context_sections.append(
                        f"### {ref.name or ref.element_type}\n"
                        f"**Location:** `{file_path}:{ref.line}`\n"
                        f"**Relevance:** {ref.similarity:.0%}\n"
                    )
                    if snippet:
                        # Include more context (500 chars instead of 300)
                        context_sections.append(
                            f"```{self.config.language}\n{snippet[:500]}\n```\n"
                        )

            # Add patterns
            if context.patterns:
                context_sections.append("## Detected Patterns\n")
                for p in context.patterns[:5]:
                    name = p.get("name", "Unknown")
                    desc = p.get("description", "")
                    conf = p.get("confidence_score", p.get("confidence", 0))
                    context_sections.append(
                        f"- **{name}** (confidence: {conf:.0%}): {desc[:150]}\n"
                    )

            # Add implementation plan if available
            if context.implementation_plan:
                context_sections.append("## Implementation Insights\n")
                if context.implementation_plan.steps:
                    context_sections.append("Suggested approach:\n")
                    for i, step in enumerate(context.implementation_plan.steps[:5], 1):
                        context_sections.append(f"{i}. {step}\n")

            # Add security considerations
            if context.security:
                if context.security.get("requirements"):
                    context_sections.append("## Security Considerations\n")
                    for req in context.security["requirements"][:3]:
                        context_sections.append(f"- {req}\n")

            # Add dependencies
            if context.dependencies:
                context_sections.append("## Related Dependencies\n")
                for dep in context.dependencies[:5]:
                    name = dep.get("name", "Unknown")
                    rel_type = dep.get("rel_type", "related")
                    context_sections.append(f"- {name} ({rel_type})\n")

            # Build prompt for LLM to generate human-readable response
            context_str = "\n".join(context_sections)

            # Check if we have meaningful context
            has_context = bool(context.relevant_code or context.patterns)

            # Get list of files in context for prompt
            files_in_context = (
                ", ".join(set(ref.file.split("/")[-1] for ref in context.relevant_code))
                if context.relevant_code
                else "the retrieved files"
            )

            prompt = f"""You are a senior code architect analyzing the codebase. Provide a detailed, technical analysis based ONLY on the code provided below.

The user asked: "{query}"

CODE RETRIEVED FROM CODEBASE:
---
{context_str if has_context else "NO CODE FOUND. Search the codebase for relevant code."}
---

Provide a comprehensive analysis that:
1. Explains what each code element does with specific details
2. Shows the full method signatures and their purposes
3. Explains relationships between components (imports, calls, inherits from)
4. Includes relevant file:line references for every claim
5. If asking about implementation, show the actual code that implements it
6. If the queried item doesn't exist in the code, state clearly: "No matching code found in the codebase"

Be specific and technical - this is for a developer who knows programming but needs to understand this specific codebase."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a precise code analysis assistant. NEVER invent code, methods, or files. Only discuss what's explicitly in the provided context. If something is not in the context, say so.",
                },
                {"role": "user", "content": prompt},
            ]

            return _LLMStreamRequest(
                messages=messages,
                prefix=f"## Analysis: {query}\n\n",
                suffix="\n\n---\n*Ask follow-up questions or use /plan to create an implementation plan.*",
            )

        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
            return f"Deep analysis failed: {e}"

    def _handle_plan(self, task: Optional[str]) -> str:
        """Handle /plan command for implementation planning."""
        if not task:
            return "Please provide a task: /plan <task description>"

        try:
            # Expand query for better context retrieval
            expanded = self._query_expander.expand(task, self._conversation_context)

            # Check confidence and ambiguity
            if expanded.is_ambiguous and expanded.confidence_score < 0.6:
                return self._generate_clarification_prompt(expanded)

            # Log expansion for debugging
            logger.info(
                f"Plan query expanded: '{task}' -> intent={expanded.intent.name}, "
                f"confidence={expanded.confidence_score:.2f}"
            )

            # Use EXPANDED query for context
            context = self.context_provider.get_implementation_context(
                expanded.enhanced_query
            )
            self._current_plan = context

            # Build rich context for LLM
            context_sections = [f"## Task: {task}\n"]

            # Add relevant code
            if context.relevant_code:
                context_sections.append("### Relevant Code References\n")
                for ref in context.relevant_code[:5]:
                    context_sections.append(
                        f"- **{ref.name or ref.element_type}** in `{ref.file}:{ref.line}` "
                        f"(relevance: {ref.similarity:.0%})\n"
                    )

            # Add implementation steps
            if context.implementation_plan and context.implementation_plan.steps:
                context_sections.append("\n### Implementation Steps\n")
                for i, step in enumerate(context.implementation_plan.steps, 1):
                    context_sections.append(f"{i}. {step}\n")

            # Add affected files
            if context.suggested_files:
                context_sections.append("\n### Affected Files\n")
                for f in context.suggested_files[:10]:
                    context_sections.append(f"- `{f}`\n")

            # Add dependencies
            if context.dependencies:
                context_sections.append("\n### Dependencies\n")
                for dep in context.dependencies[:5]:
                    name = dep.get("name", "Unknown")
                    rel_type = dep.get("rel_type", "related")
                    context_sections.append(f"- {name} ({rel_type})\n")

            # Add security considerations
            if context.security:
                if context.security.get("requirements"):
                    context_sections.append("\n### Security Requirements\n")
                    for req in context.security["requirements"]:
                        context_sections.append(f"- {req}\n")
                if context.security.get("test_suggestions"):
                    context_sections.append("\n### Test Suggestions\n")
                    for test in context.security["test_suggestions"][:5]:
                        context_sections.append(f"- {test}\n")

            # Build prompt for LLM
            context_str = "".join(context_sections)

            # Check if we have actual relevant code (>60% similarity)
            has_relevant_code = (
                any(ref.similarity >= 0.6 for ref in context.relevant_code)
                if context.relevant_code
                else False
            )

            if has_relevant_code:
                prompt = f"""You are CodeSage, an experienced pair programmer helping implement a feature.

The user wants to: {task}

Here's the context gathered from the codebase:

{context_str}

Please provide a clear, actionable implementation plan that:
1. Starts with a brief overview of the approach
2. Provides specific, step-by-step instructions that can be followed
3. References exact files and functions when discussing changes
4. Highlights potential gotchas or breaking changes
5. Suggests testing strategies
6. Maintains consistency with existing code patterns

Write in a practical, developer-friendly tone. Be specific about what needs to change and where."""
            else:
                prompt = f"""You are CodeSage, an experienced pair programmer helping implement a feature.

The user wants to: {task}

IMPORTANT: No sufficiently relevant code was found in the codebase for this task (highest similarity was {max([ref.similarity for ref in context.relevant_code], default=0):.0%} which is below the 60% threshold for relevance).
- Do NOT provide generic implementation guidance
- Do NOT suggest specific files or functions that don't have high relevance
- Do NOT make up code examples

Instead, respond with:
1. A brief acknowledgment that the codebase doesn't have relevant code for this task
2. Suggestions for what types of files might need to be created (e.g., 'you may need to create an auth module')
3. Keep it concise - maximum 3-4 sentences"""

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful senior developer creating implementation plans.",
                },
                {"role": "user", "content": prompt},
            ]

            return _LLMStreamRequest(
                messages=messages,
                prefix=f"## Implementation Plan: {task}\n\n",
                suffix="\n\n---\n*Refine this plan with: add <requirement> | focus <file> | security | approve*",
            )

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return f"Planning failed: {e}"

    def _handle_review(self, target: Optional[str]) -> str:
        """Handle /review command for code review."""
        try:
            # If no target, review all uncommitted changes
            review_target = target if target else "all changes"

            review_context = []

            # Run the hybrid analyzer if available
            try:
                from codesage.review.hybrid_analyzer import HybridReviewAnalyzer

                analyzer = HybridReviewAnalyzer(
                    config=self.config,
                    repo_path=self.config.project_path,
                )

                # Get changes
                if target:
                    all_changes = analyzer.get_all_changes()
                    changes = [c for c in all_changes if str(c.path) == target]
                    if not changes:
                        return f"No uncommitted changes found for: {target}"
                else:
                    changes = analyzer.get_all_changes()

                if not changes:
                    return (
                        "No uncommitted changes found. Make some changes and try again."
                    )

                # Run review
                result = analyzer.review_changes(
                    changes=changes,
                    use_llm_synthesis=False,
                )
                self._last_review = result

                # Build review context from ReviewResult
                review_context.append(f"## Code Review: {review_target}\n")
                review_context.append(f"**Summary:** {result.summary}\n")

                if result.issues:
                    # Group by severity
                    critical = [
                        i for i in result.issues if i.severity.name == "CRITICAL"
                    ]
                    warning = [i for i in result.issues if i.severity.name == "WARNING"]
                    info = [
                        i
                        for i in result.issues
                        if i.severity.name not in ("CRITICAL", "WARNING")
                    ]

                    if critical:
                        review_context.append(f"\n### Critical ({len(critical)})\n")
                        for issue in critical[:5]:
                            review_context.append(f"- **{issue.message}**")
                            if issue.file:
                                review_context.append(f"  `{issue.file}:{issue.line}`")
                            if issue.suggestion:
                                review_context.append(
                                    f"  Fix: {issue.suggestion[:100]}\n"
                                )

                    if warning:
                        review_context.append(f"\n### Warnings ({len(warning)})\n")
                        for issue in warning[:5]:
                            review_context.append(f"- {issue.message}")
                            if issue.file:
                                review_context.append(f"  `{issue.file}:{issue.line}`")

                    if info:
                        review_context.append(f"\n### Info ({len(info)})\n")
                        for issue in info[:3]:
                            review_context.append(f"- {issue.message}")
                else:
                    review_context.append("\nNo issues found!\n")

            except ImportError:
                review_context.append(
                    "*HybridReviewAnalyzer not available*\nRun `codesage review` from CLI for full analysis.\n"
                )

            # Build prompt for LLM to generate human-readable review
            context_str = "".join(review_context)

            # Check if we have actual review content
            has_content = (
                "No issues found" in context_str
                or "Summary:" in context_str
                or "Critical" in context_str
                or "Warning" in context_str
                or "Suggestion" in context_str
            )

            prompt = f"""You are a code reviewer. Summarize the review findings below.

Review target: {review_target}

FINDINGS:
{context_str if has_content else "No review data available."}

RULES:
1. Only discuss what's explicitly in the findings above
2. Do NOT mention files that aren't shown in the findings
3. Do NOT create example code
4. If no actual issues found, state that clearly

Keep the response concise."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a code reviewer. Summarize findings accurately. Do not invent files or issues.",
                },
                {"role": "user", "content": prompt},
            ]

            return _LLMStreamRequest(messages=messages)

        except Exception as e:
            logger.error(f"Review failed: {e}")
            return f"Review failed: {e}"

    def _handle_security(self, target: Optional[str]) -> str:
        """Handle /security command for security analysis."""
        try:
            from codesage.security.scanner import SecurityScanner
            from codesage.security.models import Severity

            scanner = SecurityScanner()

            if target:
                path = Path(target)
                if not path.is_absolute():
                    path = self.config.project_path / path
                if path.exists():
                    finding_list = scanner.scan_file(path)
                else:
                    return f"Path not found: {target}"
            else:
                report = scanner.scan_directory(self.config.project_path)
                finding_list = report.findings

            # Build security context for LLM
            security_context = []

            if finding_list:
                security_context.append(
                    f"## Security Analysis: {target or 'Project'}\n"
                )

                # Group by severity
                critical = [f for f in finding_list if f.severity == Severity.CRITICAL]
                high = [f for f in finding_list if f.severity == Severity.HIGH]
                medium = [f for f in finding_list if f.severity == Severity.MEDIUM]
                low = [f for f in finding_list if f.severity == Severity.LOW]

                if critical:
                    security_context.append(
                        f"\n### Critical Issues ({len(critical)})\n"
                    )
                    for f in critical[:5]:
                        security_context.append(
                            f"- **{f.rule.id}**: {f.rule.message[:100]}"
                        )
                        security_context.append(
                            f"  Location: `{f.file}:{f.line_number}`\n"
                        )

                if high:
                    security_context.append(
                        f"\n### High Priority Issues ({len(high)})\n"
                    )
                    for f in high[:5]:
                        security_context.append(
                            f"- **{f.rule.id}**: {f.rule.message[:80]}"
                        )
                        security_context.append(f"  `{f.file}:{f.line_number}`\n")

                if medium:
                    security_context.append(
                        f"\n### Medium Priority Issues ({len(medium)})\n"
                    )
                    for f in medium[:3]:
                        security_context.append(
                            f"- {f.rule.id}: {f.rule.message[:60]}\n"
                        )

                if low:
                    security_context.append(
                        f"\n### Low Priority ({len(low)} findings)\n"
                    )

                security_context.append(f"\n**Total findings: {len(finding_list)}**\n")
            else:
                scan_target = target if target else "Project-wide"
                security_context.append(
                    f"## Security Analysis: {scan_target}\n\n"
                    f"Security scan completed - no issues found.\n"
                    f"Scanned: {scan_target}\n"
                )

            # Build prompt for LLM
            context_str = "".join(security_context)

            # Check if there are actual findings
            has_findings = finding_list and len(finding_list) > 0

            if has_findings:
                prompt = f"""You are CodeSage, a security engineer reviewing code for vulnerabilities.

Target: {target or "Project-wide scan"}

Security scan results:

{context_str}

Please provide a security assessment that:
1. Gives an overall security posture summary
2. Prioritizes critical and high-severity issues with specific remediation steps
3. Explains the business/technical impact of each vulnerability
4. Groups related security concerns (e.g., all input validation issues together)
5. Provides code examples or specific fixes where possible
6. Suggests preventive measures for future development

Write in a clear, actionable tone. Focus on practical remediation."""
            else:
                prompt = f"""You are CodeSage, a security engineer reviewing code for vulnerabilities.

Target: {target or "Project-wide scan"}

Security scan results:

{context_str}

IMPORTANT: No security issues were found in this scan. Your response should ONLY confirm this finding - do NOT provide generic security advice, best practices, or recommendations. Keep it very brief (2-3 sentences max)."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful security engineer analyzing code vulnerabilities.",
                },
                {"role": "user", "content": prompt},
            ]

            return _LLMStreamRequest(messages=messages)

        except ImportError:
            return "Security scanner not available. Run `codesage review` from CLI for full analysis."
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return f"Security scan failed: {e}"

    def _handle_impact(self, element: Optional[str]) -> str:
        """Handle /impact command for blast radius analysis."""
        if not element:
            return "Please provide an element: /impact <function or class name>"

        try:
            import re
            import sqlite3

            # First, try to find exact name match in database
            db_path = str(self.config.storage.db_path)
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT name, file, type, line_start, code FROM code_elements WHERE name = ?",
                (element,),
            )
            exact_match = cursor.fetchone()
            conn.close()

            if exact_match:
                # Use exact match
                results = [
                    {
                        "name": exact_match[0],
                        "file": exact_match[1],
                        "type": exact_match[2],
                        "line": exact_match[3],
                        "code": exact_match[4] or "",
                        "similarity": 1.0,
                    }
                ]
            else:
                # Try multiple query strategies to find the element
                search_queries = [element]

                # 1. Try lowercase
                if element.lower() != element:
                    search_queries.append(element.lower())

                # 2. Try space-separated
                spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", element)
                if spaced != element:
                    search_queries.append(spaced.lower())

                # Search with multiple queries
                all_results = []
                seen = set()
                for sq in search_queries:
                    results = self.context_builder.search_code(sq, limit=5)
                    for r in results:
                        key = f"{r.get('file')}:{r.get('line')}"
                        if key not in seen:
                            seen.add(key)
                            all_results.append(r)

                # Re-sort by similarity (highest first)
                all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                results = all_results[:1] if all_results else []

            if not results:
                return f"Element not found: {element}"

            # Build impact context
            impact_context = []
            element_info = results[0]
            element_name = element_info.get("name", element)
            element_file = element_info.get("file", "Unknown")
            element_line = element_info.get("line", "")

            impact_context.append(f"## Impact Analysis: {element_name}\n")
            impact_context.append(f"**Location:** `{element_file}:{element_line}`\n")

            # Get graph context if available
            try:
                from codesage.storage.manager import StorageManager

                storage = StorageManager(self.config)
                if storage.graph_store:
                    element_id = element_info.get("id", element)

                    # Get dependents (what depends on this)
                    dependents = storage.graph_store.get_dependents(element_id)
                    callers = storage.graph_store.get_callers(element_id)

                    # Collect callers information
                    if callers:
                        impact_context.append(
                            f"\n### Direct Callers ({len(callers)})\n"
                        )
                        for c in callers[:10]:
                            caller_name = c.get("name", "Unknown")
                            caller_file = c.get("file", "")
                            impact_context.append(
                                f"- `{caller_name}` in `{caller_file}`\n"
                            )
                        if len(callers) > 10:
                            impact_context.append(
                                f"- *...and {len(callers) - 10} more*\n"
                            )

                    # Collect dependents information
                    if dependents:
                        impact_context.append(f"\n### Dependents ({len(dependents)})\n")
                        for d in dependents[:10]:
                            dep_name = d.get("name", "Unknown")
                            dep_type = d.get("rel_type", "related")
                            impact_context.append(f"- `{dep_name}` ({dep_type})\n")
                        if len(dependents) > 10:
                            impact_context.append(
                                f"- *...and {len(dependents) - 10} more*\n"
                            )

                    # Calculate impact metrics
                    impact_score = len(callers) + len(dependents)
                    impact_context.append(f"\n### Impact Metrics\n")
                    impact_context.append(f"- Total callers: {len(callers)}\n")
                    impact_context.append(f"- Total dependents: {len(dependents)}\n")
                    impact_context.append(f"- Combined impact score: {impact_score}\n")
                else:
                    impact_context.append(
                        "\n*Graph store not available. Run `codesage index` for full analysis.*\n"
                    )

            except Exception as e:
                impact_context.append(
                    f"\n*Could not analyze full dependency graph: {e}*\n"
                )

            # Build prompt for LLM
            context_str = "".join(impact_context)
            prompt = f"""You are CodeSage, a software architect analyzing the impact of changes to a code element.

{context_str}

Please provide an impact analysis that:
1. Summarizes the element's role in the codebase
2. Identifies the main categories of code that depend on it
3. Assesses the risk level of modifying this element (Low/Medium/High)
4. Suggests strategies for safe refactoring if needed
5. Highlights any potential breaking changes
6. Recommends testing approaches for changes to this element

Write in a clear, actionable tone suitable for developers planning changes."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful software architect analyzing code dependencies.",
                },
                {"role": "user", "content": prompt},
            ]

            return _LLMStreamRequest(messages=messages)

        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
            return f"Impact analysis failed: {e}"

    def _handle_patterns(self, query: Optional[str]) -> str:
        """Handle /patterns command to show learned patterns."""
        try:
            from codesage.memory.memory_manager import MemoryManager

            memory = MemoryManager(global_dir=self.config.memory.global_dir)

            # Get patterns
            if query:
                patterns = memory.find_similar_patterns(query, limit=10)
            else:
                patterns = memory.preference_store.get_patterns(limit=10)

            if not patterns:
                return "No patterns learned yet. Keep using CodeSage!"

            # Build patterns context for LLM
            patterns_context = []
            patterns_context.append(f"## Code Patterns in {self.config.project_name}\n")

            if query:
                patterns_context.append(f"*Showing patterns related to: {query}*\n\n")

            patterns_context.append(f"Found {len(patterns)} patterns:\n\n")

            for i, p in enumerate(patterns[:10], 1):
                if hasattr(p, "name"):
                    name = p.name
                    conf = getattr(p, "confidence_score", 0)
                    desc = getattr(p, "description", "")
                    patterns_context.append(
                        f"**{i}. {name}** (confidence: {conf:.0%})\n"
                    )
                    if desc:
                        patterns_context.append(f"   {desc[:120]}\n")
                elif isinstance(p, dict):
                    name = p.get("name", "Unknown")
                    patterns_context.append(f"**{i}. {name}**\n")
                    if p.get("description"):
                        patterns_context.append(f"   {p['description'][:120]}\n")
                patterns_context.append("")

            # Build prompt for LLM
            context_str = "".join(patterns_context)
            prompt = f"""You are CodeSage, analyzing coding patterns detected in a codebase.

{context_str}

Please provide an analysis that:
1. Summarizes the key patterns observed in this codebase
2. Groups related patterns into categories (e.g., error handling, data access, architecture)
3. Explains what these patterns reveal about the codebase's design philosophy
4. Highlights any notable or unique patterns
5. Suggests how developers should follow these patterns in new code
6. Notes any patterns that might be candidates for standardization or improvement

Write in an informative tone that helps developers understand and adopt these patterns."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful developer analyzing code patterns and conventions.",
                },
                {"role": "user", "content": prompt},
            ]

            return _LLMStreamRequest(messages=messages)

        except Exception as e:
            logger.warning(f"Could not load patterns: {e}")
            return f"Could not load patterns: {e}"

    def _handle_similar(self, element: Optional[str]) -> str:
        """Handle /similar command to find similar code."""
        if not element:
            return "Please provide an element: /similar <function or class name>"

        try:
            import re

            # Try multiple query strategies
            search_queries = [element]

            # 1. Try lowercase
            if element.lower() != element:
                search_queries.append(element.lower())

            # 2. Try space-separated
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", element)
            if spaced != element:
                search_queries.append(spaced.lower())

            # Search with multiple queries and collect all results
            all_results = []
            seen = set()
            for sq in search_queries:
                results = self.context_builder.search_code(sq, limit=10)
                for r in results:
                    key = f"{r.get('file')}:{r.get('line')}"
                    if key not in seen:
                        seen.add(key)
                        all_results.append(r)

            # Re-sort by similarity
            all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            results = all_results

            if not results:
                return f"No matches found for: {element}"

            # Skip first if it's exact match
            similar = results[1:] if results[0].get("similarity", 0) > 0.95 else results

            # Filter to only show reasonably similar results (>50%)
            similar = [r for r in similar if r.get("similarity", 0) > 0.5]

            if not similar:
                return f"No similar code found to: {element}"

            # Build similar code context for LLM
            similar_context = []
            similar_context.append(f"## Code Similar to: {element}\n")

            # Get reference element info
            ref_element = results[0]
            similar_context.append(
                f"**Reference:** `{ref_element.get('name', element)}` at `{ref_element['file']}:{ref_element['line']}`\n"
            )
            similar_context.append(
                f"**Type:** {ref_element.get('type', 'Unknown')}\n\n"
            )
            similar_context.append(f"Found {len(similar)} similar code elements:\n\n")

            for i, r in enumerate(similar[:5], 1):
                name = r.get("name", "Unknown")
                file_path = r["file"]
                line = r["line"]
                similarity = r["similarity"]
                element_type = r.get("type", "Unknown")

                similar_context.append(f"**{i}. {name}** - {similarity:.0%} similar\n")
                similar_context.append(f"   Location: `{file_path}:{line}`\n")
                similar_context.append(f"   Type: {element_type}\n\n")

            if len(similar) > 5:
                similar_context.append(
                    f"*...and {len(similar) - 5} more similar elements*\n"
                )

            # Build prompt for LLM
            context_str = "".join(similar_context)
            prompt = f"""You are CodeSage, analyzing similar code elements to help with refactoring or standardization.

{context_str}

Please provide an analysis that:
1. Summarizes what these similar elements have in common
2. Identifies if they represent duplicated logic or different use cases
3. Suggests opportunities for code consolidation or abstraction
4. Highlights any variations that might indicate inconsistent implementation
5. Recommends whether to keep them separate or unify them
6. Suggests next steps for the developer

Write in a practical, developer-friendly tone focused on code quality and maintainability."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful developer analyzing code similarities for refactoring opportunities.",
                },
                {"role": "user", "content": prompt},
            ]

            return _LLMStreamRequest(messages=messages)

        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            return f"Search failed: {e}"

    def _handle_mode(self, mode_arg: Optional[str]) -> str:
        """Handle /mode command to switch interaction mode."""
        if not mode_arg:
            return (
                f"**Current mode:** {self._mode.value}\n\n"
                f"Switch with: `/mode brainstorm|implement|review`\n\n"
                f"- **brainstorm**: Open-ended exploration\n"
                f"- **implement**: Task-focused guidance\n"
                f"- **review**: Code review focus"
            )

        mode_str = mode_arg.lower().strip()
        valid_modes = {"brainstorm", "implement", "review"}

        if mode_str not in valid_modes:
            return f"Invalid mode. Choose: {', '.join(valid_modes)}"

        old_mode = self._mode
        self._mode = ChatMode(mode_str)

        # Update system prompt
        self._session.messages = [
            msg for msg in self._session.messages if msg.role != "system"
        ]
        self._session.add_message(ChatMessage.system(self._build_system_prompt()))

        return MODE_CHANGED_TEMPLATE.format(
            mode=mode_str,
            mode_description=MODE_DESCRIPTIONS.get(mode_str, ""),
            mode_tips=MODE_TIPS.get(mode_str, ""),
        )

    def _handle_export(self, filename: Optional[str]) -> str:
        """Handle /export command to save conversation."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{timestamp}.md"

        # Check if there's any conversation to export
        user_msgs = [
            m for m in self._session.messages if m.role in ("user", "assistant")
        ]
        if not user_msgs:
            return "No conversation to export yet. Start chatting first!"

        try:
            output_path = self.config.project_path / filename

            lines = [
                f"# Chat Export - {self.config.project_name}",
                f"*Exported: {datetime.now().isoformat()}*",
                f"*Mode: {self._mode.value}*",
                f"*Messages: {len(user_msgs)}*",
                "",
                "---",
                "",
            ]

            for msg in self._session.messages:
                if msg.role == "system":
                    continue
                if msg.role == "user":
                    lines.append(f"## You\n\n{msg.content}\n")
                else:
                    lines.append(f"## CodeSage\n\n{msg.content}\n")
                lines.append("---\n")

            output_path.write_text("\n".join(lines))
            return (
                f"Conversation exported ({len(user_msgs)} messages) to: `{output_path}`"
            )

        except Exception as e:
            return f"Export failed: {e}"

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            "session_id": self._session.session_id,
            "message_count": self._session.message_count,
            "project": self.config.project_name,
            "context_enabled": self.include_context,
            "mode": self._mode.value,
        }
