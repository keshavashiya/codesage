"""Data models for the chat interface."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ChatMessage:
    """Represents a single message in a chat conversation.

    Attributes:
        role: Message role - 'user', 'assistant', or 'system'
        content: Message content text
        timestamp: When the message was created
        code_refs: Optional list of code references mentioned
        metadata: Additional metadata
    """

    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    code_refs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM API."""
        return {
            "role": self.role,
            "content": self.content,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary including metadata."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "code_refs": self.code_refs,
            "metadata": self.metadata,
        }

    @classmethod
    def user(cls, content: str, **kwargs) -> "ChatMessage":
        """Create a user message."""
        return cls(role="user", content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role="assistant", content=content, **kwargs)

    @classmethod
    def system(cls, content: str, **kwargs) -> "ChatMessage":
        """Create a system message."""
        return cls(role="system", content=content, **kwargs)


@dataclass
class ChatSession:
    """Represents a chat session with conversation history.

    Attributes:
        session_id: Unique session identifier
        messages: List of messages in the conversation
        project_path: Path to the project being discussed
        project_name: Name of the project
        created_at: When the session started
        metadata: Additional session metadata
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: List[ChatMessage] = field(default_factory=list)
    project_path: Optional[Path] = None
    project_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)

    def add_user_message(self, content: str, **kwargs) -> ChatMessage:
        """Add a user message and return it."""
        message = ChatMessage.user(content, **kwargs)
        self.add_message(message)
        return message

    def add_assistant_message(self, content: str, **kwargs) -> ChatMessage:
        """Add an assistant message and return it."""
        message = ChatMessage.assistant(content, **kwargs)
        self.add_message(message)
        return message

    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Get message history for LLM API.

        Args:
            max_messages: Maximum number of messages to include

        Returns:
            List of message dicts with role and content
        """
        messages = self.messages
        if max_messages and len(messages) > max_messages:
            messages = messages[-max_messages:]

        return [msg.to_dict() for msg in messages]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []

    @property
    def message_count(self) -> int:
        """Get number of messages in session."""
        return len(self.messages)

    @property
    def last_message(self) -> Optional[ChatMessage]:
        """Get the last message in the session."""
        return self.messages[-1] if self.messages else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_full_dict() for msg in self.messages],
            "project_path": str(self.project_path) if self.project_path else None,
            "project_name": self.project_name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
