"""Structured logging for LLM calls."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from threading import Lock
from uuid import uuid4

from agent_rag.core.config import ToolChoice
from agent_rag.llm.interface import LLMMessage, LLMResponse


def _truncate(text: str, limit: int = 400) -> str:
    if text is None:
        return ""
    return text[:limit]


def _messages_preview(messages: list[LLMMessage]) -> str:
    parts = []
    for msg in messages:
        content = msg.content or ""
        parts.append(f"{msg.role}: {content}")
    return "\n".join(parts)


def _tool_names(tools: Optional[list[dict[str, Any]]]) -> list[str]:
    if not tools:
        return []
    names = []
    for tool in tools:
        func = tool.get("function") if isinstance(tool, dict) else None
        name = func.get("name") if isinstance(func, dict) else None
        if name:
            names.append(name)
    return names


def _response_preview(response: Optional[LLMResponse]) -> str:
    if not response:
        return ""
    parts = [response.content or ""]
    if response.tool_calls:
        tool_names = [tc.name for tc in response.tool_calls]
        parts.append(f"[tool_calls] {', '.join(tool_names)}")
    return "\n".join(p for p in parts if p)


_CALL_INDEX_LOCK = Lock()
_GLOBAL_CALL_INDEX = 0


def _get_log_path() -> Path:
    log_file = os.environ.get("AGENT_RAG_LLM_LOG_FILE", "logs/llm_calls/llm_calls.jsonl")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def _next_call_index() -> int:
    global _GLOBAL_CALL_INDEX
    with _CALL_INDEX_LOCK:
        _GLOBAL_CALL_INDEX += 1
        return _GLOBAL_CALL_INDEX


def create_llm_log_file(
    *,
    agent_name: str,
) -> Path:
    return _get_log_path()


def log_llm_call(
    *,
    agent_name: str,
    mode: str,
    model: str,
    provider: str,
    messages: list[LLMMessage],
    tools: Optional[list[dict[str, Any]]],
    tool_choice: ToolChoice,
    response: Optional[LLMResponse],
    error: Optional[str] = None,
    log_path: Optional[Path] = None,
    call_index: Optional[int] = None,
) -> Path:
    if log_path is None:
        log_path = _get_log_path()
    elif isinstance(log_path, str):
        log_path = Path(log_path)

    if call_index is None:
        call_index = _next_call_index()

    request_preview = _truncate(_messages_preview(messages), 400)
    response_preview = _truncate(_response_preview(response), 400)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "mode": mode,
        "model": model,
        "provider": provider,
        "tool_choice": tool_choice.value if isinstance(tool_choice, ToolChoice) else str(tool_choice),
        "tool_names": _tool_names(tools),
        "call_index": call_index,
        "request": {
            "message_count": len(messages),
            "preview_head_400": request_preview,
        },
        "response": {
            "preview_head_400": response_preview,
            "finish_reason": response.finish_reason if response else None,
            "usage": response.usage if response else None,
        },
        "error": error,
    }

    with log_path.open("a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=True)
        f.write("\n")

    return log_path
