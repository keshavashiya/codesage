"""Chat engine for interactive code conversations."""

from typing import Any, Dict, List, Optional, Tuple

from codesage.llm.provider import LLMProvider
from codesage.storage.database import Database
from codesage.utils.config import Config
from codesage.utils.logging import get_logger

from .commands import ChatCommand, CommandParser, ParsedCommand
from .context import CodeContextBuilder
from .models import ChatMessage, ChatSession
from .prompts import CHAT_HELP, CHAT_SYSTEM_PROMPT

logger = get_logger("chat.engine")


class ChatEngine:
    """Main engine for chat interactions.

    Coordinates between user input, code context, and LLM
    to provide an interactive chat experience.
    """

    # Maximum messages to include in LLM context
    MAX_HISTORY_MESSAGES = 20

    def __init__(
        self,
        config: Config,
        include_context: bool = True,
        max_context_results: int = 3,
    ):
        """Initialize the chat engine.

        Args:
            config: CodeSage configuration
            include_context: Whether to include code context in prompts
            max_context_results: Maximum code snippets in context
        """
        self.config = config
        self.include_context = include_context
        self.max_context_results = max_context_results

        # Lazy-loaded components
        self._llm: Optional[LLMProvider] = None
        self._context_builder: Optional[CodeContextBuilder] = None
        self._db: Optional[Database] = None

        # Session management
        self._session = ChatSession(
            project_path=config.project_path,
            project_name=config.project_name,
        )

        # Command parser
        self._parser = CommandParser()

        # Add system message
        self._session.add_message(ChatMessage.system(self._build_system_prompt()))

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
    def db(self) -> Database:
        """Lazy-load the database."""
        if self._db is None:
            self._db = Database(self.config.storage.db_path)
        return self._db

    @property
    def session(self) -> ChatSession:
        """Get the current chat session."""
        return self._session

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM.

        Returns:
            Formatted system prompt string
        """
        return CHAT_SYSTEM_PROMPT.format(
            project_name=self.config.project_name,
            language=self.config.language,
        )

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
        if parsed.command == ChatCommand.EXIT:
            return "Goodbye!", False

        if parsed.command == ChatCommand.HELP:
            return self._handle_help(), True

        if parsed.command == ChatCommand.CLEAR:
            return self._handle_clear(), True

        if parsed.command == ChatCommand.STATS:
            return self._handle_stats(), True

        if parsed.command == ChatCommand.CONTEXT:
            return self._handle_context(parsed.args), True

        if parsed.command == ChatCommand.SEARCH:
            return self._handle_search(parsed.args), True

        if parsed.command == ChatCommand.UNKNOWN:
            return f"Unknown command. Type /help for available commands.", True

        # Regular message - process with LLM
        if parsed.command == ChatCommand.MESSAGE and parsed.args:
            return self._process_message(parsed.args), True

        return "Please enter a message or command.", True

    def _process_message(self, message: str) -> str:
        """Process a regular chat message.

        Args:
            message: User message text

        Returns:
            Assistant response text
        """
        # Add user message to session
        user_msg = self._session.add_user_message(message)

        # Build context from codebase
        context = ""
        code_refs = []
        if self.include_context:
            try:
                context = self.context_builder.build_context(
                    message,
                    limit=self.max_context_results,
                )
                code_refs = self.context_builder.get_code_refs(message, limit=5)
            except Exception as e:
                logger.warning(f"Failed to build context: {e}")

        # Build messages for LLM
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

    def _build_llm_messages(self, context: str = "") -> List[Dict[str, str]]:
        """Build message list for LLM API.

        Args:
            context: Optional code context to include

        Returns:
            List of message dicts
        """
        messages = []

        # System message (always first, without context)
        messages.append({"role": "system", "content": self._build_system_prompt()})

        # Get conversation history (excluding system messages)
        history = [
            msg for msg in self._session.messages
            if msg.role != "system"
        ]

        # Limit history
        if len(history) > self.MAX_HISTORY_MESSAGES:
            history = history[-self.MAX_HISTORY_MESSAGES:]

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

    def _handle_help(self) -> str:
        """Handle /help command.

        Returns:
            Help text
        """
        return CHAT_HELP

    def _handle_clear(self) -> str:
        """Handle /clear command.

        Returns:
            Confirmation message
        """
        # Keep only system message
        self._session.clear_history()
        self._session.add_message(ChatMessage.system(self._build_system_prompt()))
        return "Conversation history cleared."

    def _handle_stats(self) -> str:
        """Handle /stats command.

        Returns:
            Index statistics
        """
        try:
            stats = self.db.get_stats()
            return (
                f"**Index Statistics**\n\n"
                f"- Files indexed: {stats['files']}\n"
                f"- Code elements: {stats['elements']}\n"
                f"- Last indexed: {stats['last_indexed'] or 'Never'}\n"
                f"- Project: {self.config.project_name}"
            )
        except Exception as e:
            return f"Could not get stats: {e}"

    def _handle_context(self, args: Optional[str]) -> str:
        """Handle /context command.

        Args:
            args: Optional arguments

        Returns:
            Context information or settings update
        """
        if args:
            # Could implement context settings here
            return f"Context settings not yet implemented."

        # Show current context settings
        return (
            f"**Context Settings**\n\n"
            f"- Code context enabled: {self.include_context}\n"
            f"- Max context results: {self.max_context_results}\n"
            f"- Max context chars: {self.context_builder.MAX_CONTEXT_CHARS}"
        )

    def _handle_search(self, query: Optional[str]) -> str:
        """Handle /search command.

        Args:
            query: Search query

        Returns:
            Search results
        """
        if not query:
            return "Please provide a search query: /search <query>"

        try:
            results = self.context_builder.search_code(query, limit=5)
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            return f"No results found for: {query}"

        # Format results
        output = [f"**Search Results for:** {query}\n"]

        for i, r in enumerate(results, 1):
            name_part = f" ({r['name']})" if r.get('name') else ""
            output.append(
                f"\n**{i}. {r['file']}:{r['line']}**{name_part}\n"
                f"Similarity: {r['similarity']:.0%} | Type: {r['type']}\n"
                f"```{r['language']}\n{r['code'][:300]}{'...' if len(r['code']) > 300 else ''}\n```"
            )

        return "\n".join(output)

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session.

        Returns:
            Session information dictionary
        """
        return {
            "session_id": self._session.session_id,
            "message_count": self._session.message_count,
            "project": self.config.project_name,
            "context_enabled": self.include_context,
        }
