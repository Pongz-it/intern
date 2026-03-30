"""Session and Memory data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MemoryType(str, Enum):
    """Types of user memory."""
    FACT = "fact"  # Factual information (e.g., "user works at company X")
    PREFERENCE = "preference"  # User preferences (e.g., "user prefers concise answers")
    HABIT = "habit"  # User habits (e.g., "user often asks about technical details")
    GOAL = "goal"  # User goals (e.g., "user is learning Python")
    RELATIONSHIP = "relationship"  # Relationships between entities


class MessageType(str, Enum):
    """Message type in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    id: str
    session_id: str
    type: MessageType
    content: str
    role: str = "user"  # For LLM compatibility
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llm_message(self) -> dict[str, Any]:
        """Convert to LLM message format."""
        msg: dict[str, Any] = {
            "role": self.type.value if self.type != MessageType.ASSISTANT else "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "type": self.type.value,
            "content": self.content,
            "role": self.role,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ConversationSession:
    """A conversation session containing multiple messages."""
    id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(
        self,
        msg_type: MessageType,
        content: str,
        id: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Add a message to the session."""
        import uuid
        message = ConversationMessage(
            id=id or str(uuid.uuid4()),
            session_id=self.id,
            type=msg_type,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_messages_for_llm(self) -> list[dict[str, Any]]:
        """Get messages formatted for LLM."""
        return [msg.to_llm_message() for msg in self.messages]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class UserMemory:
    """A piece of user memory extracted from conversations."""
    id: str
    session_id: str
    user_id: Optional[str]
    memory_type: MemoryType
    content: str
    embedding: Optional[list[float]] = None
    importance: float = 0.5  # 0-1 scale
    source_message_id: Optional[str] = None
    source_conversation: Optional[str] = None  # Context of extraction
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None  # Search relevance score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "importance": self.importance,
            "source_message_id": self.source_message_id,
            "source_conversation": self.source_conversation,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_search_result(self) -> dict[str, Any]:
        """Convert to search result format."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "metadata": self.metadata,
        }
