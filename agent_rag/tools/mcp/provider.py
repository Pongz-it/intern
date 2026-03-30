"""MCP tool provider for Agent RAG."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from agent_rag.tools.interface import Tool, ToolResponse
from agent_rag.tools.registry import ToolRegistry
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MCPToolSpec:
    """Specification for an MCP tool."""
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPTool(Tool[None]):
    """Tool wrapper that calls an MCP tool."""

    def __init__(
        self,
        provider: "MCPToolProvider",
        spec: MCPToolSpec,
        id: Optional[int] = None,
    ) -> None:
        super().__init__(id)
        self._provider = provider
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    def tool_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._spec.name,
                "description": self._spec.description,
                "parameters": self._spec.input_schema or {"type": "object", "properties": {}},
            },
        }

    def run(
        self,
        override_kwargs: Optional[None] = None,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        result_text, payload = self._provider.call_tool(self._spec.name, llm_kwargs)
        return ToolResponse(
            llm_response=result_text,
            rich_response=payload,
        )


class MCPToolProvider:
    """Discover and call MCP tools via streamable HTTP."""

    def __init__(
        self,
        server_url: str,
        auth_token: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.transport_url = f"{self.server_url}/?transportType=streamable-http"
        self.headers = headers.copy() if headers else {}
        if auth_token:
            self.headers.setdefault("Authorization", f"Bearer {auth_token}")

    def _require_mcp(self) -> None:
        try:
            import mcp  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "mcp is not installed. Install with: pip install mcp"
            ) from exc

    def _run_with_session(self, action: Any) -> Any:
        self._require_mcp()

        async def _runner() -> Any:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(self.transport_url, headers=self.headers) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    return await action(session)

        return asyncio.run(_runner())

    def discover_tools(self) -> list[MCPToolSpec]:
        """Discover tools from MCP server."""

        async def _action(session: Any) -> list[MCPToolSpec]:
            await session.initialize()
            tools_result = await session.list_tools()
            specs: list[MCPToolSpec] = []
            for tool in tools_result.tools:
                input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
                specs.append(MCPToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=input_schema,
                ))
            return specs

        return self._run_with_session(_action)

    def build_registry(self) -> ToolRegistry:
        """Build a ToolRegistry populated with MCP tools."""
        registry = ToolRegistry()
        for spec in self.discover_tools():
            registry.register(MCPTool(self, spec))
        return registry

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, Any]:
        """Call an MCP tool and return text + parsed payload."""

        async def _action(session: Any) -> Any:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)

        result = self._run_with_session(_action)
        text_blocks = []
        for block in getattr(result, "content", []):
            text = getattr(block, "text", None)
            if text:
                text_blocks.append(text)

        text = text_blocks[-1] if text_blocks else ""
        payload = None
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"text": text} if text else None

        return text, payload
