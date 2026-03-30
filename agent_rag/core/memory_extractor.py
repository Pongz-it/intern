"""Memory extraction using LLM."""

import json
from typing import Any, Optional

from pydantic import BaseModel

from agent_rag.core.session_memory_models import MemoryType, UserMemory
from agent_rag.core.memory_store import MemoryStore
from agent_rag.llm.interface import LLM
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ExtractedMemory(BaseModel):
    """A memory extracted by LLM."""
    content: str
    memory_type: str
    importance: float
    reasoning: str


class MemoryExtractor:
    """Extract memories from conversations using LLM."""

    EXTRACTION_PROMPT = """You are a memory extraction assistant. Your task is to analyze the conversation and extract important user information that should be remembered for future interactions.

Analyze the conversation and extract memories in the following categories:

1. FACTS: Factual information about the user
   - Work information (company, role, projects)
   - Personal information (location, interests)
   - Technical knowledge level
   - Preferences explicitly stated

2. PREFERENCES: User preferences for interactions
   - Communication style preferences
   - Format preferences (concise vs detailed)
   - Topic preferences or avoidances
   - Special requirements or constraints

3. HABITS: Behavioral patterns observed
   - Common query patterns
   - Information seeking behaviors
   - Typical use cases
   - Interaction rhythms

4. GOALS: User goals and objectives
   - Learning goals
   - Project objectives
   - Research interests
   - Career aspirations

5. RELATIONSHIPS: Entity relationships mentioned
   - People they work with
   - Tools they use
   - Systems they're familiar with
   - Concepts they relate

CONVERSATION:
{conversation}

Extract up to {max_memories} most important memories. For each memory:
- Make it specific and actionable
- Include the reasoning for extraction
- Rate importance from 0-1

Output as JSON array with the following structure:
[
  {{
    "content": "specific memory content",
    "memory_type": "fact|preference|habit|goal|relationship",
    "importance": 0.8,
    "reasoning": "why this is important to remember"
  }}
]

Only output valid JSON, no other text."""

    def __init__(
        self,
        llm: LLM,
        memory_store: MemoryStore,
        max_memories_per_extraction: int = 5,
        similarity_threshold: float = 0.85,
    ):
        """Initialize memory extractor.

        Args:
            llm: LLM for extraction
            memory_store: Memory store for checking duplicates
            max_memories_per_extraction: Maximum memories to extract per call
            similarity_threshold: Threshold for duplicate detection
        """
        self.llm = llm
        self.memory_store = memory_store
        self.max_memories = max_memories_per_extraction
        self.similarity_threshold = similarity_threshold

    async def extract_from_conversation(
        self,
        conversation_text: str,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[UserMemory]:
        """Extract memories from conversation text.

        Args:
            conversation_text: Full conversation text
            session_id: Session ID
            user_id: Optional user ID

        Returns:
            List of extracted memories
        """
        prompt = self.EXTRACTION_PROMPT.format(
            conversation=conversation_text,
            max_memories=self.max_memories,
        )

        try:
            from agent_rag.llm.interface import LLMMessage
            messages = [LLMMessage(role="user", content=prompt)]
            
            # Use async chat if available to avoid blocking event loop
            if hasattr(self.llm, "chat_async"):
                response = await self.llm.chat_async(messages)
            else:
                # Fallback to sync chat in thread pool for providers without async support
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                # Use a default executor
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, lambda: self.llm.chat(messages))

            content = response.content

            logger.debug(f"[MemoryExtractor] LLM response: {content[:200]}...")

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            extracted_data = json.loads(content.strip())
            
            logger.debug(f"[MemoryExtractor] Parsed {len(extracted_data)} memories from LLM response")

            if not isinstance(extracted_data, list):
                extracted_data = [extracted_data]

            memories = []
            for item in extracted_data:
                raw_content = item.get("content", "")
                
                if not raw_content or not raw_content.strip():
                    logger.warning(f"[MemoryExtractor] Skipping empty memory content")
                    continue
                
                if len(raw_content.strip()) < 5:
                    logger.warning(f"[MemoryExtractor] Skipping too short memory: '{raw_content[:20]}...'")
                    continue

                extracted = ExtractedMemory(
                    content=raw_content.strip(),
                    memory_type=item.get("memory_type", "fact"),
                    importance=float(item.get("importance", 0.5)),
                    reasoning=item.get("reasoning", ""),
                )

                logger.debug(f"[MemoryExtractor] Processing memory: '{extracted.content[:50]}...'")

                try:
                    is_duplicate = await self._check_duplicate(extracted.content, session_id)
                except Exception as dup_err:
                    logger.warning(f"[MemoryExtractor] Duplicate check failed: {dup_err}, proceeding without duplicate check")
                    is_duplicate = False
                
                if is_duplicate:
                    logger.debug(f"[MemoryExtractor] Skipping duplicate: {extracted.content[:50]}...")
                    continue

                logger.debug(f"[MemoryExtractor] Adding memory to store...")
                memory = await self.memory_store.add_memory(
                    content=extracted.content,
                    memory_type=MemoryType(extracted.memory_type),
                    session_id=session_id,
                    user_id=user_id,
                    importance=extracted.importance,
                    source_conversation=extracted.reasoning,
                )
                memories.append(memory)
                logger.debug(f"[MemoryExtractor] Successfully added memory: {memory.id}")

            logger.info(f"[MemoryExtractor] Extracted {len(memories)} new memories from conversation")
            return memories

        except json.JSONDecodeError as e:
            logger.error(f"[MemoryExtractor] Failed to parse extraction response: {e}")
            return []
        except Exception as e:
            logger.error(f"[MemoryExtractor] Extraction failed: {e}")
            return []

    async def _check_duplicate(self, content: str, session_id: str) -> bool:
        """Check if similar memory already exists.

        Args:
            content: Content to check
            session_id: Session ID

        Returns:
            True if duplicate exists
        """
        try:
            existing = await self.memory_store.get_memories_by_session(session_id, limit=20)

            for memory in existing:
                similarity = self._calculate_similarity(content, memory.content)
                if similarity >= self.similarity_threshold:
                    return True

            return False
        except Exception:
            return False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score 0-1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    async def extract_from_messages(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[UserMemory]:
        """Extract memories from formatted messages.

        Args:
            messages: List of message dicts with role and content
            session_id: Session ID
            user_id: Optional user ID

        Returns:
            List of extracted memories
        """
        conversation_parts = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role in ["user", "human"]:
                conversation_parts.append(f"用户: {content}")
            elif role in ["assistant", "ai"]:
                conversation_parts.append(f"助手: {content}")
            elif role == "system":
                conversation_parts.append(f"系统: {content}")

        conversation_text = "\n\n".join(conversation_parts)
        return await self.extract_from_conversation(conversation_text, session_id, user_id)


class MemoryRetriever:
    """Retrieve relevant memories for queries."""

    def __init__(self, memory_store: MemoryStore):
        """Initialize memory retriever.

        Args:
            memory_store: Memory store
        """
        self.memory_store = memory_store

    async def retrieve_for_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        memory_types: Optional[list[MemoryType]] = None,
        limit: int = 5,
    ) -> list[UserMemory]:
        """Retrieve memories relevant to query.

        Args:
            query: Query text
            user_id: Optional user ID
            memory_types: Optional memory type filters
            limit: Maximum results

        Returns:
            List of relevant memories
        """
        memories = await self.memory_store.search_memories(
            query=query,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
            score_threshold=0.3,
        )

        return memories

    def format_memories_for_context(
        self,
        memories: list[UserMemory],
        include_type: bool = True,
    ) -> str:
        """Format memories for inclusion in LLM context.

        Args:
            memories: List of memories
            include_type: Whether to include memory type

        Returns:
            Formatted memory context
        """
        if not memories:
            return ""

        parts = ["用户历史信息："]
        for i, memory in enumerate(memories, 1):
            if include_type:
                type_label = {
                    MemoryType.FACT: "事实",
                    MemoryType.PREFERENCE: "偏好",
                    MemoryType.HABIT: "习惯",
                    MemoryType.GOAL: "目标",
                    MemoryType.RELATIONSHIP: "关系",
                }.get(memory.memory_type, "信息")

                parts.append(f"{i}. [{type_label}] {memory.content}")
            else:
                parts.append(f"{i}. {memory.content}")

        return "\n".join(parts)

    async def get_user_summary(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """Get summary of all user memories.

        Args:
            user_id: User ID

        Returns:
            Summary dict
        """
        all_memories = await self.memory_store.get_memories_by_user(user_id)

        summary = {
            "total_count": len(all_memories),
            "by_type": {},
            "high_importance": [],
        }

        for memory in all_memories:
            mtype = memory.memory_type.value
            if mtype not in summary["by_type"]:
                summary["by_type"][mtype] = 0
            summary["by_type"][mtype] += 1

            if memory.importance >= 0.8:
                summary["high_importance"].append({
                    "content": memory.content,
                    "type": mtype,
                })

        return summary
