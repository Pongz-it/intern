"""Session and Memory enabled Chat Agent."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

from agent_rag.agent.chat_agent import ChatAgent
from agent_rag.core.callbacks import AgentCallback
from agent_rag.core.config import AgentConfig
from agent_rag.core.memory_extractor import MemoryExtractor, MemoryRetriever
from agent_rag.core.memory_store import MemoryStore
from agent_rag.core.models import AgentResponse, Message, ToolCall
from agent_rag.core.session_memory_models import (
    ConversationMessage,
    ConversationSession,
    MemoryType,
    MessageType,
    UserMemory,
)
from agent_rag.core.session_manager import SessionManager
from agent_rag.embedding.interface import Embedder
from agent_rag.llm.interface import LLM
from agent_rag.tools.registry import ToolRegistry
from agent_rag.tools.runner import ToolRunner
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class SessionMemoryChatAgent:
    """Chat agent with session and memory capabilities.

    Features:
    - Session management: Tracks conversation history
    - Memory retrieval: Retrieves relevant user memories before processing
    - Memory extraction: Extracts and stores new memories after conversations
    """

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        session_manager: SessionManager,
        memory_store: MemoryStore,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        extract_memories_after_conversation: bool = True,
        retrieve_memories_for_query: bool = True,
        max_memories_per_query: int = 5,
        memory_types_to_retrieve: Optional[list[str]] = None,
        agent_callback: Optional[AgentCallback] = None,
        memory_extraction_interval: int = 600,
    ):
        """Initialize session memory chat agent.

        Args:
            llm: LLM for responses and memory extraction
            embedder: Embedder for memory storage and retrieval
            session_manager: Session manager for conversation storage
            memory_store: Memory store for user memories
            config: Agent configuration
            tool_registry: Registry of available tools
            system_prompt: Optional custom system prompt
            user_id: User ID for memory association
            extract_memories_after_conversation: Whether to extract memories after each conversation
            retrieve_memories_for_query: Whether to retrieve memories before processing queries
            max_memories_per_query: Maximum memories to retrieve per query
            memory_types_to_retrieve: Types of memories to retrieve
            agent_callback: Callback for agent lifecycle events
            memory_extraction_interval: Interval in seconds between memory extractions (default 600s = 10min)
        """
        self.llm = llm
        self.embedder = embedder
        self.session_manager = session_manager
        self.memory_store = memory_store
        self.user_id = user_id
        self.extract_memories = extract_memories_after_conversation
        self.retrieve_memories = retrieve_memories_for_query
        self.max_memories = max_memories_per_query
        self.memory_types_filter = (
            [MemoryType(t) for t in memory_types_to_retrieve]
            if memory_types_to_retrieve
            else None
        )

        self.base_agent = ChatAgent(
            llm=llm,
            config=config,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            agent_callback=agent_callback,
        )

        self.memory_extractor = MemoryExtractor(
            llm=llm,
            memory_store=memory_store,
        )
        self.memory_retriever = MemoryRetriever(memory_store=memory_store)

        self._current_session: Optional[ConversationSession] = None
        self._cached_messages: list[Message] = []
        self._cached_memories: list[UserMemory] = []
        self._last_extraction_time: dict[str, datetime] = {}
        self._last_extracted_message_id: dict[str, str] = {}
        self._extraction_interval = timedelta(seconds=memory_extraction_interval)
        logger.info(f"[SessionMemoryChatAgent] Initialized (memory extraction every {memory_extraction_interval}s)")

    async def start_session(
        self,
        title: Optional[str] = None,
    ) -> ConversationSession:
        """Start a new conversation session.

        Args:
            title: Optional session title

        Returns:
            Created session
        """
        self._current_session = await self.session_manager.create_session(
            user_id=self.user_id,
            title=title,
        )
        logger.info(f"[SessionMemoryChatAgent] Started session: {self._current_session.id}")
        return self._current_session

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing session.

        Args:
            session_id: Session ID

        Returns:
            Session if found
        """
        return await self.session_manager.get_session(session_id)

    async def resume_session(self, session_id: str) -> Optional[ConversationSession]:
        """Resume an existing session.

        Args:
            session_id: Session ID

        Returns:
            Session if found
        """
        session = await self.session_manager.get_session(session_id)
        if session:
            self._current_session = session
            if not self._cached_messages:
                self._cached_messages = await self.session_manager.get_session_messages(session_id)
                logger.info(f"[SessionMemoryChatAgent] Loaded {len(self._cached_messages)} messages from cache for session {session_id[:8]}...")
            logger.info(f"[SessionMemoryChatAgent] Resumed session: {session_id}")
        return session

    def _get_current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session.id if self._current_session else None

    async def _retrieve_memories(self, query: str) -> list[str]:
        """Retrieve relevant memories for query.

        Args:
            query: User query

        Returns:
            Formatted memory context
        """
        if not self.retrieve_memories:
            return []

        if self._cached_memories:
            context = self.memory_retriever.format_memories_for_context(self._cached_memories)
            logger.debug(f"[SessionMemoryChatAgent] Using {len(self._cached_memories)} cached memories")
            return context

        try:
            memories = await self.memory_retriever.retrieve_for_query(
                query=query,
                user_id=self.user_id,
                memory_types=self.memory_types_filter,
                limit=self.max_memories,
            )
        except Exception as e:
            logger.warning(
                f"[SessionMemoryChatAgent] Memory retrieval unavailable, continuing without memories: {e}"
            )
            return []

        if memories:
            self._cached_memories = memories
            context = self.memory_retriever.format_memories_for_context(memories)
            logger.debug(f"[SessionMemoryChatAgent] Retrieved {len(memories)} memories")
            return context

        return []

    def _get_conversation_history(self) -> str:
        """Get cached conversation history for context.

        Returns:
            Formatted conversation history string
        """
        if not self._cached_messages:
            return ""
        return "\n".join([f"{m.type.value}: {m.content}" for m in self._cached_messages])

    def _resolve_session_query(
        self,
        query: str,
        context: Optional[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Resolve the user query to persist separately from the prompt sent to the base agent."""
        enhanced_context = dict(context) if context else {}
        session_query = enhanced_context.pop("session_persist_query", query)
        return session_query, enhanced_context

    async def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResponse:
        """Run the agent with session and memory support.

        Args:
            query: User query
            context: Optional context for tools

        Returns:
            AgentResponse with result
        """
        session_id = self._get_current_session_id()
        session_query, enhanced_context = self._resolve_session_query(query, context)

        if session_id and self._current_session:
            await self.session_manager.add_message(
                session_id=session_id,
                role=MessageType.USER.value,
                content=session_query,
            )
            self._cached_messages.append(
                ConversationMessage(
                    id="temp_user",
                    session_id=session_id,
                    type=MessageType.USER,
                    content=session_query
                )
            )

        memories_context = await self._retrieve_memories(session_query)

        if memories_context:
            memory_key = "user_memories"
            if memory_key in enhanced_context:
                enhanced_context[memory_key] = (
                    enhanced_context[memory_key] + "\n\n" + memories_context
                )
            else:
                enhanced_context[memory_key] = memories_context

        conversation_history = self._get_conversation_history()
        if conversation_history:
            history_key = "conversation_history"
            if history_key in enhanced_context:
                enhanced_context[history_key] = (
                    enhanced_context[history_key] + "\n\n" + conversation_history
                )
            else:
                enhanced_context[history_key] = conversation_history

        context = enhanced_context

        response = self.base_agent.run(query=query, context=context)

        if session_id and self._current_session:
            last_msg = response.messages[-1] if response.messages else None
            if last_msg and last_msg.content:
                await self.session_manager.add_message(
                    session_id=session_id,
                    role=MessageType.ASSISTANT.value,
                    content=last_msg.content,
                )
                self._cached_messages.append(
                    ConversationMessage(
                        id="temp_assistant",
                        session_id=session_id,
                        type=MessageType.ASSISTANT,
                        content=last_msg.content
                    )
                )

            if self.extract_memories:
                await self._extract_session_memories()

        return response

    async def run_stream(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Run the agent with streaming and memory support.

        Args:
            query: User query
            context: Optional context for tools

        Yields:
            Response tokens
        """
        session_id = self._get_current_session_id()
        session_query, enhanced_context = self._resolve_session_query(query, context)

        if session_id and self._current_session:
            self._cached_messages.append(
                ConversationMessage(
                    id="temp_user",
                    session_id=session_id,
                    type=MessageType.USER,
                    content=session_query
                )
            )

        memories_context = await self._retrieve_memories(session_query)

        if memories_context:
            memory_key = "user_memories"
            if memory_key in enhanced_context:
                enhanced_context[memory_key] = (
                    enhanced_context[memory_key] + "\n\n" + memories_context
                )
            else:
                enhanced_context[memory_key] = memories_context

        conversation_history = self._get_conversation_history()
        if conversation_history:
            history_key = "conversation_history"
            if history_key in enhanced_context:
                enhanced_context[history_key] = (
                    enhanced_context[history_key] + "\n\n" + conversation_history
                )
            else:
                enhanced_context[history_key] = conversation_history

        context = enhanced_context

        if session_id and self._current_session:
            asyncio.create_task(
                self.session_manager.add_message(
                    session_id=session_id,
                    role=MessageType.USER.value,
                    content=session_query,
                )
            )

        for token in self.base_agent.run_stream(query=query, context=context):
            yield token

        if session_id and self._current_session:
            final_content = self.base_agent._get_final_content()
            if final_content:
                self._cached_messages.append(
                    ConversationMessage(
                        id="temp_assistant",
                        session_id=session_id,
                        type=MessageType.ASSISTANT,
                        content=final_content
                    )
                )

                asyncio.create_task(self._finalize_session_turn(session_id, final_content))

    async def _finalize_session_turn(self, session_id: str, content: str) -> None:
        """Background task to save message and extract memories."""
        try:
            # 1. Save assistant message to DB
            await self.session_manager.add_message(
                session_id=session_id,
                role=MessageType.ASSISTANT.value,
                content=content,
            )
            
            # 2. Extract memories if enabled
            if self.extract_memories:
                await self._extract_session_memories()
                
        except Exception as e:
            logger.error(f"[SessionMemoryChatAgent] Background finalization failed: {e}")

    async def _extract_session_memories(self) -> None:
        """Extract memories from current session (only new messages since last extraction).

        Trigger conditions:
        - Time interval reached (default 10 minutes), OR
        - New messages >= 5 rounds
        """
        if not self._current_session:
            return

        session_id = self._current_session.id
        now = datetime.utcnow()

        try:
            messages = await self.session_manager.get_session_messages(session_id=session_id)

            if len(messages) < 2:
                return

            last_extracted_id = self._last_extracted_message_id.get(session_id)
            new_messages = []
            if last_extracted_id:
                found = False
                for msg in messages:
                    if msg.id == last_extracted_id:
                        found = True
                        continue
                    if found:
                        new_messages.append(msg)
            else:
                new_messages = list(messages)

            new_count = len(new_messages)
            if new_count < 2:
                logger.debug(f"[SessionMemoryChatAgent] No new messages for {session_id[:8]}...")
                return

            last_time = self._last_extraction_time.get(session_id)
            time_elapsed = (now - last_time).total_seconds() if last_time else float('inf')

            should_extract = False
            trigger_reason = ""

            if new_count >= 5:
                should_extract = True
                trigger_reason = f"{new_count} new messages (threshold: 5)"
            elif time_elapsed >= self._extraction_interval.total_seconds():
                should_extract = True
                if last_time is None:
                    trigger_reason = "initial extraction"
                else:
                    trigger_reason = (
                        f"time interval reached ({int(time_elapsed)}s / "
                        f"{int(self._extraction_interval.total_seconds())}s)"
                    )
            else:
                elapsed_display = "first run" if last_time is None else f"{int(time_elapsed)}s"
                logger.debug(
                    f"[SessionMemoryChatAgent] Skipping extraction for {session_id[:8]}... "
                    f"({new_count} new messages, {elapsed_display} since last)"
                )

            if not should_extract:
                return

            messages_for_extraction = [
                {"role": msg.type.value, "content": msg.content}
                for msg in new_messages
            ]

            extracted = await self.memory_extractor.extract_from_messages(
                messages=messages_for_extraction,
                session_id=session_id,
                user_id=self.user_id,
            )

            if new_messages:
                last_msg = new_messages[-1]
                self._last_extracted_message_id[session_id] = last_msg.id
            self._last_extraction_time[session_id] = now
            logger.info(f"[SessionMemoryChatAgent] Extracted {len(extracted)} memories ({trigger_reason}) for session {session_id[:8]}...")

        except Exception as e:
            logger.error(f"[SessionMemoryChatAgent] Failed to extract memories: {e}")

    async def close(self) -> None:
        """Close connections and extract any pending memories."""
        if self._current_session:
            session_id = self._current_session.id
            try:
                messages = await self.session_manager.get_session_messages(session_id=session_id)
                if messages:
                    last_extracted_id = self._last_extracted_message_id.get(session_id)
                    has_pending = False
                    if last_extracted_id:
                        for msg in messages:
                            if msg.id == last_extracted_id:
                                has_pending = True
                                break
                            if has_pending:
                                break
                    else:
                        has_pending = len(messages) >= 2

                    if has_pending:
                        logger.info(f"[SessionMemoryChatAgent] Extracting pending memories before close...")
                        messages_for_extraction = [
                            {"role": msg.type.value, "content": msg.content}
                            for msg in messages
                        ]
                        extracted = await self.memory_extractor.extract_from_messages(
                            messages=messages_for_extraction,
                            session_id=session_id,
                            user_id=self.user_id,
                        )
                        logger.info(f"[SessionMemoryChatAgent] Extracted {len(extracted)} pending memories before close")
            except Exception as e:
                logger.error(f"[SessionMemoryChatAgent] Failed to extract pending memories: {e}")

        await self.session_manager.close()
        self.memory_store.close()
        logger.info("[SessionMemoryChatAgent] Closed")

    async def get_memory_summary(self, user_id: Optional[str] = None) -> dict[str, Any]:
        """Get memory summary for user.

        Args:
            user_id: User ID (uses agent's user_id if not provided)

        Returns:
            Memory summary
        """
        uid = user_id or self.user_id
        if not uid:
            return {"error": "No user ID provided"}

        return await self.memory_retriever.get_user_summary(uid)

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[ConversationSession]:
        """List user's sessions.

        Args:
            user_id: User ID (uses agent's user_id if not provided)
            limit: Maximum sessions to return

        Returns:
            List of sessions
        """
        uid = user_id or self.user_id
        sessions, _ = await self.session_manager.list_sessions(
            user_id=uid,
            limit=limit,
        )
        return sessions

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its memories.

        Args:
            session_id: Session ID

        Returns:
            True if deleted
        """
        deleted = await self.session_manager.delete_session(session_id)
        if deleted:
            await self.memory_store.delete_memories_by_session(session_id)
        return deleted


class MemoryContextBuilder:
    """Build memory context for prompts."""

    def __init__(self, memory_retriever: MemoryRetriever):
        """Initialize context builder.

        Args:
            memory_retriever: Memory retriever
        """
        self.retriever = memory_retriever

    async def build_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        include_all_types: bool = True,
    ) -> str:
        """Build memory context for a query.

        Args:
            query: User query
            user_id: User ID
            include_all_types: Include all memory types or just relevant

        Returns:
            Formatted memory context
        """
        memory_types = None if include_all_types else [
            MemoryType.FACT,
            MemoryType.PREFERENCE,
        ]

        memories = await self.retriever.retrieve_for_query(
            query=query,
            user_id=user_id,
            memory_types=memory_types,
            limit=5,
        )

        return self.retriever.format_memories_for_context(memories)

    def build_few_shot_context(
        self,
        memories: list[dict[str, Any]],
    ) -> str:
        """Build few-shot examples from memories.

        Args:
            memories: List of memory dicts

        Returns:
            Formatted few-shot context
        """
        if not memories:
            return ""

        parts = ["基于用户历史交互的模式："]
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "")
            mem_type = mem.get("type", "信息")
            parts.append(f"  {i}. [{mem_type}] {content}")

        return "\n".join(parts)
