"""Session storage implementation using messages table in database."""

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, String, Text, DateTime, Index, Integer
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from agent_rag.core.database import Base
from agent_rag.core.session_memory_models import (
    ConversationMessage,
    ConversationSession,
    MessageType,
)
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


def _sanitize_db_text(value: Optional[str]) -> str:
    """Remove characters PostgreSQL text fields cannot store."""
    if not value:
        return ""
    return value.replace("\x00", "")


class MessageModel(Base, AsyncAttrs):
    """Database model for conversation messages.

    This table stores all conversation messages for sessions.
    Messages are stored with session_id for grouping.
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(100), index=True, nullable=True)
    role: Mapped[str] = mapped_column(String(50), default="user")
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_json: Mapped[Optional[str]] = mapped_column("metadata_json", Text, nullable=True)

    __table_args__ = (
        Index("idx_messages_session_created", "session_id", "created_at"),
        Index("idx_messages_user_created", "user_id", "created_at"),
    )

    @property
    def msg_metadata(self) -> dict:
        """Get metadata as dict."""
        if self.extra_json:
            return json.loads(self.extra_json)
        return {}

    @msg_metadata.setter
    def msg_metadata(self, value: dict) -> None:
        """Set metadata from dict."""
        self.extra_json = json.dumps(value) if value else None

    def to_model(self) -> ConversationMessage:
        """Convert to domain model."""
        msg_type = MessageType.USER
        if self.role == "assistant":
            msg_type = MessageType.ASSISTANT
        elif self.role == "system":
            msg_type = MessageType.SYSTEM

        return ConversationMessage(
            id=self.id,
            session_id=self.session_id,
            type=msg_type,
            content=self.content,
            role=self.role,
            created_at=self.created_at,
            metadata=self.msg_metadata,
        )

    def to_llm_message(self) -> dict[str, Any]:
        """Convert to LLM message format."""
        msg: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        return msg


class SessionManager:
    """Manager for conversation sessions using messages table.

    Features:
    - Uses single 'messages' table for all messages
    - session_id groups messages into conversations
    - Direct LLM message format output
    """

    def __init__(
        self,
        database_url: str,
        table_prefix: str = "",
    ):
        """Initialize session manager.

        Args:
            database_url: Database connection URL
            table_prefix: Prefix for table names (not used, table is fixed as 'messages')
        """
        self.database_url = database_url
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        self._engine = create_async_engine(self.database_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info(f"[SessionManager] Initialized with database: {self.database_url}")

    async def close(self) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def create_session(self, session_id: Optional[str] = None) -> str:
        """Create/return a session ID.

        Args:
            session_id: Optional existing session ID

        Returns:
            Session ID (generated or provided)
        """
        if not self._session_factory:
            await self.initialize()

        sid = session_id or str(uuid.uuid4())
        logger.debug(f"[SessionManager] Session: {sid}")
        return sid

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session with its messages.

        Args:
            session_id: Session ID

        Returns:
            ConversationSession object or None
        """
        messages = await self.get_session_messages(session_id)
        return ConversationSession(
            id=session_id,
            title=None,
            messages=messages
        )

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: Optional[int] = None,
        metadata: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> ConversationMessage:
        """Add a message to a session.

        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            token_count: Optional token count
            metadata: Optional metadata
            user_id: Optional user ID

        Returns:
            Created message
        """
        if not self._session_factory:
            await self.initialize()

        message_id = str(uuid.uuid4())
        safe_content = _sanitize_db_text(content)
        db_message = MessageModel(
            id=message_id,
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=safe_content,
            token_count=token_count,
            extra_json=_sanitize_db_text(json.dumps(metadata)) if metadata else None,
        )

        async with self._session_factory() as db_session:
            db_session.add(db_message)
            await db_session.commit()

        logger.debug(f"[SessionManager] Added message to session {session_id}: {message_id}")
        return db_message.to_model()

    async def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[ConversationMessage]:
        """Get messages for a session.

        Args:
            session_id: Session ID
            limit: Optional limit
            offset: Offset

        Returns:
            List of messages
        """
        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as db_session:
            from sqlalchemy import select, desc

            query = (
                select(MessageModel)
                .where(MessageModel.session_id == session_id)
                .order_by(MessageModel.created_at)
            )

            if offset > 0:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            result = await db_session.execute(query)
            messages = result.scalars().all()

        return [msg.to_model() for msg in messages]

    async def _get_messages_by_sessions(
        self,
        session_ids: list[str],
        limit_per_session: int = 10,
    ) -> dict[str, list[ConversationMessage]]:
        """Batch get messages for multiple sessions.

        Args:
            session_ids: List of session IDs
            limit_per_session: Max messages per session

        Returns:
            Dictionary mapping session_id to list of messages
        """
        if not session_ids:
            return {}

        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as db_session:
            from sqlalchemy import select

            query = (
                select(MessageModel)
                .where(MessageModel.session_id.in_(session_ids))
                .order_by(MessageModel.session_id, MessageModel.created_at)
                .limit(limit_per_session * len(session_ids))
            )

            result = await db_session.execute(query)
            messages = result.scalars().all()

        messages_by_session: dict[str, list[ConversationMessage]] = {}
        for msg in messages:
            if msg.session_id not in messages_by_session:
                messages_by_session[msg.session_id] = []
            messages_by_session[msg.session_id].append(msg.to_model())

        return messages_by_session

    async def get_llm_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get messages formatted for LLM.

        Args:
            session_id: Session ID
            limit: Optional limit
            offset: Offset

        Returns:
            List of messages in LLM format
        """
        messages = await self.get_session_messages(session_id, limit, offset)
        return [msg.to_llm_message() for msg in messages]

    async def get_recent_messages(
        self,
        session_id: str,
        num_messages: int = 10,
    ) -> list[ConversationMessage]:
        """Get recent messages from a session.

        Args:
            session_id: Session ID
            num_messages: Number of recent messages to get

        Returns:
            List of recent messages
        """
        messages = await self.get_session_messages(session_id)
        return messages[-num_messages:] if num_messages > 0 else messages

    async def delete_session(self, session_id: str) -> bool:
        """Delete all messages for a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted
        """
        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as db_session:
            from sqlalchemy import delete

            await db_session.execute(
                delete(MessageModel).where(MessageModel.session_id == session_id)
            )
            await db_session.commit()

        logger.info(f"[SessionManager] Deleted session: {session_id}")
        return True

    async def get_user_messages(
        self,
        user_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[ConversationMessage]:
        """Get all messages for a user (across all sessions).

        Args:
            user_id: User ID
            limit: Optional limit
            offset: Offset

        Returns:
            List of messages ordered by creation time
        """
        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as db_session:
            from sqlalchemy import select, desc

            query = (
                select(MessageModel)
                .where(MessageModel.user_id == user_id)
                .order_by(MessageModel.created_at)
            )

            if offset > 0:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            result = await db_session.execute(query)
            messages = result.scalars().all()
            return [msg.to_model() for msg in messages]

    async def count_messages(self, session_id: str) -> int:
        """Count messages in a session.

        Args:
            session_id: Session ID

        Returns:
            Number of messages
        """
        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as db_session:
            from sqlalchemy import select, func

            result = await db_session.execute(
                select(func.count(MessageModel.id)).where(MessageModel.session_id == session_id)
            )
            return result.scalar() or 0

    async def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """List all session IDs.

        Args:
            limit: Maximum sessions to return
            offset: Offset

        Returns:
            List of session IDs
        """
        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as db_session:
            from sqlalchemy import select, distinct

            result = await db_session.execute(
                select(distinct(MessageModel.session_id))
                .order_by(MessageModel.session_id)
                .offset(offset)
                .limit(limit)
            )
            return [row[0] for row in result.all()]

    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ConversationSession]:
        """Get all sessions for a user.

        Args:
            user_id: User ID
            limit: Maximum sessions to return
            offset: Offset

        Returns:
            List of sessions with messages
        """
        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as db_session:
            from sqlalchemy import select, func

            subquery = (
                select(
                    MessageModel.session_id,
                    func.max(MessageModel.created_at).label('max_created')
                )
                .where(MessageModel.user_id == user_id)
                .group_by(MessageModel.session_id)
                .order_by(func.max(MessageModel.created_at).desc())
                .offset(offset)
                .limit(limit)
                .subquery()
            )

            result = await db_session.execute(
                select(subquery.c.session_id)
            )
            session_ids = [row[0] for row in result.all()]

        if not session_ids:
            return []

        messages_by_session = await self._get_messages_by_sessions(session_ids)

        sessions = []
        for session_id in session_ids:
            messages = messages_by_session.get(session_id, [])
            sessions.append(ConversationSession(
                id=session_id,
                title=None,
                messages=messages
            ))

        return sessions

    async def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title.

        Args:
            session_id: Session ID
            title: New title

        Returns:
            True if updated
        """
        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as db_session:
            from sqlalchemy import select

            result = await db_session.execute(
                select(SessionModel).where(SessionModel.id == session_id)
            )
            session = result.scalar_one_or_none()

            if session:
                session.title = title
                await db_session.commit()
                return True

            return False
