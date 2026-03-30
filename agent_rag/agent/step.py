"""Agent step execution for Agent RAG."""

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from agent_rag.core.models import Message, ToolCall
from agent_rag.core.config import ToolChoice
from agent_rag.llm.interface import LLM, LLMMessage, LLMResponse
from agent_rag.tools.registry import ToolRegistry
from agent_rag.tools.runner import ToolRunner
from agent_rag.tools.interface import ToolResponse
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StepResult:
    """Result of a single agent step."""
    content: str
    tool_calls: list[ToolCall]
    should_continue: bool
    tokens_used: int = 0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentStep:
    """
    Executes a single step in the agent loop.

    A step consists of:
    1. Calling LLM with current messages
    2. Processing any tool calls
    3. Returning result with continuation decision
    """

    def __init__(
        self,
        llm: LLM,
        tool_registry: ToolRegistry,
        tool_runner: Optional[ToolRunner] = None,
        *,
        log_llm_calls: bool = False,
        agent_name: str = "",
    ) -> None:
        """
        Initialize the step executor.

        Args:
            llm: LLM provider
            tool_registry: Registry of available tools
            tool_runner: Optional tool runner (created if not provided)
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.tool_runner = tool_runner or ToolRunner(tool_registry)
        self.log_llm_calls = log_llm_calls
        self.agent_name = agent_name or "agent"
        self.log_path = None
        self.llm_call_index = 0

    def _messages_to_llm_format(
        self,
        messages: list[Message],
    ) -> list[LLMMessage]:
        """Convert Message objects to LLMMessage format."""
        return [
            LLMMessage(
                role=msg.role,
                content=msg.content,
                tool_calls=[tc.to_dict() for tc in (msg.tool_calls or [])] if msg.tool_calls else None,
                tool_call_id=msg.tool_call_id,
            )
            for msg in messages
        ]

    def execute(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        force_tool: Optional[str] = None,
    ) -> StepResult:
        """
        Execute a single step.

        Args:
            messages: Conversation messages
            tools: Optional tool definitions (uses registry if not provided)
            force_tool: Optional tool name to force

        Returns:
            StepResult with content and tool calls
        """
        # Get tool definitions
        if tools is None:
            tools = self.tool_registry.get_all_definitions()

        # Convert messages
        llm_messages = self._messages_to_llm_format(messages)

        # Build tool choice
        tool_choice = ToolChoice.NONE
        if force_tool:
            if tools:
                tools = [t for t in tools if t.get("function", {}).get("name") == force_tool]
            tool_choice = ToolChoice.REQUIRED
        elif tools:
            tool_choice = ToolChoice.AUTO

        # Call LLM
        response: Optional[LLMResponse] = None
        try:
            self.llm_call_index += 1
            response = self.llm.chat(
                messages=llm_messages,
                tools=tools if tools else None,
                tool_choice=tool_choice,
            )
        except Exception as e:
            if self.log_llm_calls:
                from agent_rag.utils.llm_logging import log_llm_call

                log_llm_call(
                    agent_name=self.agent_name,
                    mode="chat",
                    model=self.llm.config.model,
                    provider=self.llm.config.provider,
                    messages=llm_messages,
                    tools=tools if tools else None,
                    tool_choice=tool_choice,
                    response=response,
                    error=str(e),
                    log_path=self.log_path,
                    call_index=self.llm_call_index,
                )
            raise

        if self.log_llm_calls:
            from agent_rag.utils.llm_logging import log_llm_call

            log_llm_call(
                agent_name=self.agent_name,
                mode="chat",
                model=self.llm.config.model,
                provider=self.llm.config.provider,
                messages=llm_messages,
                tools=tools if tools else None,
                tool_choice=tool_choice,
                response=response,
                log_path=self.log_path,
                call_index=self.llm_call_index,
            )

        # Extract tool calls
        tool_calls: list[ToolCall] = []
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                ))

        # Determine if we should continue
        should_continue = len(tool_calls) > 0

        # Handle content - DeepSeek V3 puts answer in reasoning instead of content
        final_content = response.content or response.reasoning or ""

        return StepResult(
            content=final_content,
            tool_calls=tool_calls,
            should_continue=should_continue,
            tokens_used=response.usage.get("total_tokens", 0) if response.usage else 0,
        )

    def execute_stream(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Iterator[tuple[str, Optional[StepResult]]]:
        """
        Execute a step with streaming.

        Args:
            messages: Conversation messages
            tools: Optional tool definitions

        Yields:
            Tuples of (token, final_result) where final_result is None until complete
        """
        if tools is None:
            tools = self.tool_registry.get_all_definitions()

        llm_messages = self._messages_to_llm_format(messages)

        tool_choice = ToolChoice.AUTO if tools else ToolChoice.NONE

        # Accumulate response
        full_content = ""
        tool_calls: list[ToolCall] = []
        tokens_used = 0

        self.llm_call_index += 1
        stream = self.llm.chat_stream(
            messages=llm_messages,
            tools=tools if tools else None,
            tool_choice=tool_choice,
        )

        final_response: Optional[LLMResponse] = None
        try:
            while True:
                chunk = next(stream)
                # Handle content - DeepSeek V3 puts answer in reasoning instead of content
                text_to_yield = chunk.content or chunk.reasoning
                if text_to_yield:
                    full_content += text_to_yield
                    yield (text_to_yield, None)
        except StopIteration as exc:
            final_response = exc.value
        except Exception as e:
            if self.log_llm_calls:
                from agent_rag.utils.llm_logging import log_llm_call

                log_llm_call(
                    agent_name=self.agent_name,
                    mode="chat_stream",
                    model=self.llm.config.model,
                    provider=self.llm.config.provider,
                    messages=llm_messages,
                    tools=tools if tools else None,
                    tool_choice=tool_choice,
                    response=final_response,
                    error=str(e),
                    log_path=self.log_path,
                    call_index=self.llm_call_index,
                )
            raise

        if self.log_llm_calls:
            from agent_rag.utils.llm_logging import log_llm_call

            log_llm_call(
                agent_name=self.agent_name,
                mode="chat_stream",
                model=self.llm.config.model,
                provider=self.llm.config.provider,
                messages=llm_messages,
                tools=tools if tools else None,
                tool_choice=tool_choice,
                response=final_response,
                log_path=self.log_path,
                call_index=self.llm_call_index,
            )

        if final_response and final_response.tool_calls:
            for tc in final_response.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                ))

        if final_response and final_response.usage:
            tokens_used = final_response.usage.get("total_tokens", 0)

        # Yield final result
        result = StepResult(
            content=full_content,
            tool_calls=tool_calls,
            should_continue=len(tool_calls) > 0,
            tokens_used=tokens_used,
        )
        yield ("", result)

    def execute_tools(
        self,
        tool_calls: list[ToolCall],
        context: Optional[dict[str, Any]] = None,
    ) -> list[tuple[ToolCall, ToolResponse]]:
        """
        Execute tool calls and return results.

        Args:
            tool_calls: Tool calls to execute
            context: Optional context for tools

        Returns:
            List of (tool_call, result_string) tuples
        """
        results: list[tuple[ToolCall, ToolResponse]] = []

        for tool_call in tool_calls:
            try:
                result = self.tool_runner.run(
                    tool_call,
                    context=context,
                )
                results.append((tool_call, result))
            except Exception as e:
                logger.error(f"Tool execution failed: {tool_call.name}: {e}")
                results.append((tool_call, ToolResponse(
                    llm_response=f"Error: {str(e)}",
                )))

        return results

    def execute_tools_parallel(
        self,
        tool_calls: list[ToolCall],
        context: Optional[dict[str, Any]] = None,
        max_workers: int = 5,
    ) -> list[tuple[ToolCall, ToolResponse]]:
        """
        Execute tool calls in parallel.

        Args:
            tool_calls: Tool calls to execute
            context: Optional context for tools
            max_workers: Maximum parallel workers

        Returns:
            List of (tool_call, result_string) tuples
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[tuple[ToolCall, ToolResponse]] = []

        def execute_single(tc: ToolCall) -> tuple[ToolCall, ToolResponse]:
            try:
                result = self.tool_runner.run(
                    tc,
                    context=context,
                )
                return (tc, result)
            except Exception as e:
                logger.error(f"Tool execution failed: {tc.name}: {e}")
                return (tc, ToolResponse(
                    llm_response=f"Error: {str(e)}",
                ))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(execute_single, tc): tc
                for tc in tool_calls
            }

            for future in as_completed(futures):
                results.append(future.result())

        # Sort by original order
        tc_order = {tc.id: i for i, tc in enumerate(tool_calls)}
        results.sort(key=lambda x: tc_order.get(x[0].id, 0))

        return results
