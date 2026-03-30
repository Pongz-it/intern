"""Query expansion utilities for search."""

from datetime import datetime
from typing import Any, Optional

from agent_rag.core.config import ReasoningEffort
from agent_rag.core.models import Message
from agent_rag.llm.interface import LLM, LLMMessage
from agent_rag.tools.builtin.search.prompts import (
    KEYWORD_REPHRASE_SYSTEM_PROMPT,
    KEYWORD_REPHRASE_USER_PROMPT,
    REPHRASE_CONTEXT_PROMPT,
    SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT,
    SEMANTIC_QUERY_REPHRASE_USER_PROMPT,
)
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


def _history_to_messages(history: Optional[Any]) -> list[LLMMessage]:
    """Convert history to LLMMessage list."""
    if not history:
        return []

    messages = []
    for item in history:
        if isinstance(item, Message):
            messages.append(LLMMessage(role=item.role, content=item.content))
        elif isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            messages.append(LLMMessage(role=role, content=content))
        else:
            messages.append(LLMMessage(role="user", content=str(item)))
    return messages


def _build_additional_context(
    user_info: Optional[str] = None,
    memories: Optional[list[str]] = None,
) -> str:
    """Build additional context section for rephrasing."""
    has_user_info = bool(user_info and user_info.strip())
    has_memories = bool(memories and any(m.strip() for m in memories))
    if not has_user_info and not has_memories:
        return ""

    formatted_user_info = user_info if has_user_info else "N/A"
    formatted_memories = (
        "\n".join(f"- {memory}" for memory in memories)
        if has_memories and memories
        else "N/A"
    )
    return REPHRASE_CONTEXT_PROMPT.format(
        user_info=formatted_user_info,
        memories=formatted_memories,
    )


def semantic_query_rephrase(
    query: str,
    llm: LLM,
    history: Optional[Any] = None,
    user_info: Optional[str] = None,
    memories: Optional[list[str]] = None,
) -> Optional[str]:
    """Generate a single semantic rephrase for hybrid search."""
    current_date = datetime.now().strftime("%B %d, %Y %H:%M")
    system_prompt = SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT.format(
        current_date=current_date
    )
    additional_context = _build_additional_context(user_info, memories)
    user_prompt = SEMANTIC_QUERY_REPHRASE_USER_PROMPT.format(
        additional_context=additional_context,
        user_query=query,
    )
    messages = [LLMMessage(role="system", content=system_prompt)]
    messages.extend(_history_to_messages(history))
    messages.append(LLMMessage(role="user", content=user_prompt))

    try:
        # Use ReasoningEffort.OFF for fast query rephrasing (auxiliary task)
        response = llm.chat(messages, max_tokens=100, reasoning_effort=ReasoningEffort.OFF)
        return response.content.strip() if response.content else None
    except Exception as exc:
        logger.warning(f"Semantic rephrase failed: {exc}")
        return None


def keyword_query_expansion(
    query: str,
    llm: LLM,
    max_queries: int = 3,
    history: Optional[Any] = None,
    user_info: Optional[str] = None,
    memories: Optional[list[str]] = None,
) -> list[str]:
    """Generate keyword-heavy query variants."""
    current_date = datetime.now().strftime("%B %d, %Y %H:%M")
    system_prompt = KEYWORD_REPHRASE_SYSTEM_PROMPT.format(
        current_date=current_date
    )
    additional_context = _build_additional_context(user_info, memories)
    user_prompt = KEYWORD_REPHRASE_USER_PROMPT.format(
        additional_context=additional_context,
        user_query=query,
    )
    messages = [LLMMessage(role="system", content=system_prompt)]
    messages.extend(_history_to_messages(history))
    messages.append(LLMMessage(role="user", content=user_prompt))

    try:
        # Use ReasoningEffort.OFF for fast keyword expansion (auxiliary task)
        response = llm.chat(messages, max_tokens=150, reasoning_effort=ReasoningEffort.OFF)
        lines = [line.strip() for line in response.content.split("\n") if line.strip()]
        return lines[:max_queries]
    except Exception as exc:
        logger.warning(f"Keyword expansion failed: {exc}")
        return []
