"""Chat agent implementation for Agent RAG."""

from typing import Any, Iterator, Optional

from agent_rag.agent.base import AgentState, BaseAgent
from agent_rag.agent.step import AgentStep
from agent_rag.citation.processor import DynamicCitationProcessor
from agent_rag.citation.utils import chunks_to_citations, format_citation_list
from agent_rag.core.callbacks import AgentCallback, StreamCallback, ToolCallback
from agent_rag.core.config import AgentConfig
from agent_rag.core.models import AgentResponse, Chunk, Citation, Message, ToolCall
from agent_rag.llm.interface import LLM
from agent_rag.tools.registry import ToolRegistry
from agent_rag.tools.interface import ToolResponse
from agent_rag.tools.runner import ToolRunner
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a highly capable, thoughtful, and precise assistant for the organization's knowledge base. Your goal is to deeply understand the user's intent, think step-by-step through complex problems, and provide clear, accurate answers with proper citations.

# Critical Instruction - ALWAYS Search First
You are a RAG (Retrieval-Augmented Generation) assistant connected to an internal knowledge base. For EVERY question, you MUST search the knowledge base first to find relevant information. Only use your general knowledge as a supplement if the search results are insufficient.

Never answer questions directly from your training data when they could be answered from the knowledge base. The knowledge base contains the most accurate and up-to-date information for this organization.

# Response Style
You use different text styles, bolding, and other formatting to make your responses more readable and engaging.
For code you prefer to use Markdown and specify the language.

# Tools
When using search tools:
- Always search the knowledge base first for every question
- Stay as faithful to the user's query as possible
- If initial results cannot fully answer the query, try again with DIFFERENT queries or tools
- Do NOT repeat the same or very similar queries that already have been run
- Never provide more than 3 queries at once

## internal_search
Use the `internal_search` tool to search the internal knowledge base for information. This is your PRIMARY tool for answering questions. Use it for:
- Any factual question about the organization, products, projects, or processes
- Questions about company policies, procedures, or documentation
- Technical questions about systems, APIs, or codebases
- Any topic that might be covered in indexed documents

# Citations
CRITICAL: When referencing knowledge from searches, cite relevant statements INLINE using the format [1], [2], [3], etc. based on the "citation_id" field of documents.
DO NOT provide any links following the citations. Cite inline as opposed to leaving all citations until the very end."""


# Reminder message added after search tool results
CITATION_REMINDER = """
Remember to provide inline citations in the format [1], [2], [3], etc. based on the "citation_id" field of the documents.
If the search results provide enough information to answer the question, answer directly with citations.
Do not search again unless the results are insufficient.

Do not acknowledge this hint in your response.
""".strip()


class ChatAgent(BaseAgent):
    """
    Chat agent for conversational RAG.

    Implements the agent loop:
    1. Receive user query
    2. Optionally call tools (search, etc.)
    3. Generate response with citations
    4. Repeat until done or max steps reached
    """

    def __init__(
        self,
        llm: LLM,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        tool_runner: Optional[ToolRunner] = None,
        system_prompt: Optional[str] = None,
        stream_callback: Optional[StreamCallback] = None,
        tool_callback: Optional[ToolCallback] = None,
        agent_callback: Optional[AgentCallback] = None,
    ) -> None:
        """
        Initialize the chat agent.

        Args:
            llm: LLM provider
            config: Agent configuration
            tool_registry: Registry of available tools
            tool_runner: Optional tool runner
            system_prompt: Optional custom system prompt
            stream_callback: Callback for streaming tokens
            tool_callback: Callback for tool execution events
            agent_callback: Callback for agent lifecycle events
        """
        super().__init__(
            llm=llm,
            config=config,
            tool_registry=tool_registry,
            stream_callback=stream_callback,
            tool_callback=tool_callback,
            agent_callback=agent_callback,
        )

        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.tool_runner = tool_runner or ToolRunner(self.tool_registry)
        self.step_executor = AgentStep(
            llm=llm,
            tool_registry=self.tool_registry,
            tool_runner=self.tool_runner,
            agent_name="chat_agent",
        )

        # Track retrieved chunks for citations
        self._retrieved_chunks: list[Chunk] = []
        self._citation_processor: Optional[DynamicCitationProcessor] = None

        # Track if search tool has been called in current turn (for skip_query_expansion optimization)
        self._has_called_search_tool: bool = False

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._retrieved_chunks = []
        self._citation_processor = None
        self._has_called_search_tool = False

    def _initialize_conversation(self, query: str) -> None:
        """Initialize conversation with system prompt and user query."""
        self.reset()
        self.add_system_message(self.system_prompt)
        self.add_user_message(query)

    def _process_tool_results(
        self,
        tool_results: list[tuple[ToolCall, ToolResponse]],
    ) -> None:
        """Process tool results and extract chunks."""
        for tool_call, result in tool_results:
            self.add_tool_result(tool_call.id, result.llm_response)

            # Track tool call
            self.state.tool_calls.append(tool_call)

            # Extract chunks from search results if available
            if result.rich_response:
                chunks = result.rich_response.get("chunks", [])
                if isinstance(chunks, list):
                    for chunk in chunks:
                        if isinstance(chunk, Chunk):
                            self._retrieved_chunks.append(chunk)

    def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Run the chat agent.

        Args:
            query: User query
            context: Optional context for tools

        Returns:
            AgentResponse with result
        """
        self._notify_start(query)
        self._initialize_conversation(query)

        try:
            while self.should_continue():
                self.state.step_count += 1

                # Following Onyx pattern: on the last cycle, no tools allowed
                # This forces the LLM to answer directly without tool calls
                if self.is_last_cycle():
                    logger.info(
                        f"[ChatAgent] Last cycle ({self.state.step_count}), forcing answer without tools"
                    )
                    tools = []  # No tools, LLM must answer
                else:
                    tools = self.get_tool_definitions()

                # Execute step
                step_result = self.step_executor.execute(
                    messages=self.state.messages,
                    tools=tools,
                )

                self.state.total_tokens += step_result.tokens_used

                # Add assistant message
                if step_result.content or step_result.tool_calls:
                    self.add_assistant_message(
                        content=step_result.content,
                        tool_calls=step_result.tool_calls if step_result.tool_calls else None,
                    )
                    self._notify_step(
                        self.state.step_count,
                        self.state.messages[-1],
                    )

                # Execute tool calls if any
                if step_result.tool_calls:
                    for tc in step_result.tool_calls:
                        self._notify_tool_start(tc)

                    # Build tool context with skip_query_expansion optimization
                    # If search was already called this turn, skip query expansion for subsequent calls
                    tool_context = dict(context) if context else {}
                    tool_context["skip_query_expansion"] = self._has_called_search_tool

                    tool_results = self.step_executor.execute_tools(
                        step_result.tool_calls,
                        context=tool_context,
                    )

                    # Track if any search tool was called
                    ran_search = False
                    for tc, result in tool_results:
                        if tc.name == "internal_search":
                            self._has_called_search_tool = True
                            ran_search = True
                        self._notify_tool_end(tc, result)

                    self._process_tool_results(tool_results)

                    # Add reminder message after search tool results (Onyx pattern)
                    # This guides the model to use search results and cite properly
                    if ran_search:
                        self.add_user_message(CITATION_REMINDER)

                # Check if we should stop
                if not step_result.should_continue:
                    self.state.should_stop = True
                    break

            # Build response
            final_content = self._get_final_content()
            citations = self._build_citations()

            response = AgentResponse(
                content=final_content,
                citations=citations,
                tool_calls=self.state.tool_calls,
                messages=self.state.messages,
                metadata={
                    "steps": self.state.step_count,
                    "tokens": self.state.total_tokens,
                },
            )

            self._notify_end(response)
            return response

        except Exception as e:
            self._notify_error(e)
            raise

    def run_stream(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Run the chat agent with streaming.

        Args:
            query: User query
            context: Optional context for tools

        Yields:
            Response tokens
        """
        self._notify_start(query)
        self._initialize_conversation(query)

        try:
            while self.should_continue():
                self.state.step_count += 1

                accumulated_content = ""
                step_result = None

                # Following Onyx pattern: on the last cycle, no tools allowed
                if self.is_last_cycle():
                    logger.info(
                        f"[ChatAgent] Last cycle ({self.state.step_count}), forcing answer without tools"
                    )
                    tools = []
                else:
                    tools = self.get_tool_definitions()

                # Stream step execution
                for token, result in self.step_executor.execute_stream(
                    messages=self.state.messages,
                    tools=tools,
                ):
                    if token:
                        accumulated_content += token
                        self._stream_token(token)
                        yield token

                    if result:
                        step_result = result

                if step_result is None:
                    break

                self.state.total_tokens += step_result.tokens_used

                # Add assistant message
                if accumulated_content or step_result.tool_calls:
                    self.add_assistant_message(
                        content=accumulated_content,
                        tool_calls=step_result.tool_calls if step_result.tool_calls else None,
                    )

                # Execute tool calls if any
                if step_result.tool_calls:
                    for tc in step_result.tool_calls:
                        self._notify_tool_start(tc)

                    # Build tool context with skip_query_expansion optimization
                    tool_context = dict(context) if context else {}
                    tool_context["skip_query_expansion"] = self._has_called_search_tool

                    tool_results = self.step_executor.execute_tools(
                        step_result.tool_calls,
                        context=tool_context,
                    )

                    # Track if any search tool was called
                    ran_search = False
                    for tc, result in tool_results:
                        if tc.name == "internal_search":
                            self._has_called_search_tool = True
                            ran_search = True
                        self._notify_tool_end(tc, result)

                    self._process_tool_results(tool_results)

                    # Add reminder message after search tool results (Onyx pattern)
                    if ran_search:
                        self.add_user_message(CITATION_REMINDER)

                    # Stream tool execution indicator
                    yield "\n[Searching...]\n"

                if not step_result.should_continue:
                    self.state.should_stop = True
                    break

            # Stream citations if any
            citations = self._build_citations()
            if citations:
                citation_text = "\n\n" + format_citation_list(citations)
                yield citation_text

        except Exception as e:
            self._notify_error(e)
            raise

    def _get_final_content(self) -> str:
        """Get final response content from messages."""
        # Find last assistant message without tool calls
        for msg in reversed(self.state.messages):
            if msg.role == "assistant" and not msg.tool_calls:
                return msg.content
            elif msg.role == "assistant" and msg.content:
                return msg.content

        return ""

    def _build_citations(self) -> list[Citation]:
        """Build citations from retrieved chunks."""
        if not self._retrieved_chunks:
            return []

        # Deduplicate by document_id
        seen_docs: set[str] = set()
        unique_chunks: list[Chunk] = []

        for chunk in self._retrieved_chunks:
            if chunk.document_id not in seen_docs:
                seen_docs.add(chunk.document_id)
                unique_chunks.append(chunk)

        return chunks_to_citations(unique_chunks)

    def continue_conversation(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Continue an existing conversation.

        Args:
            message: New user message
            context: Optional context for tools

        Returns:
            AgentResponse with result
        """
        self.add_user_message(message)
        self.state.should_stop = False
        self.state.step_count = 0

        # Clear retrieved chunks for new turn
        self._retrieved_chunks = []
        # Reset search tracking for new turn
        self._has_called_search_tool = False

        return self._run_loop(context)

    def _run_loop(
        self,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResponse:
        """Run the agent loop (internal helper)."""
        while self.should_continue():
            self.state.step_count += 1

            # Following Onyx pattern: on the last cycle, no tools allowed
            if self.is_last_cycle():
                logger.info(
                    f"[ChatAgent] Last cycle ({self.state.step_count}), forcing answer without tools"
                )
                tools = []
            else:
                tools = self.get_tool_definitions()

            step_result = self.step_executor.execute(
                messages=self.state.messages,
                tools=tools,
            )

            self.state.total_tokens += step_result.tokens_used

            if step_result.content or step_result.tool_calls:
                self.add_assistant_message(
                    content=step_result.content,
                    tool_calls=step_result.tool_calls if step_result.tool_calls else None,
                )

            if step_result.tool_calls:
                # Build tool context with skip_query_expansion optimization
                tool_context = dict(context) if context else {}
                tool_context["skip_query_expansion"] = self._has_called_search_tool

                tool_results = self.step_executor.execute_tools(
                    step_result.tool_calls,
                    context=tool_context,
                )

                # Track if any search tool was called
                ran_search = False
                for tc, result in tool_results:
                    if tc.name == "internal_search":
                        self._has_called_search_tool = True
                        ran_search = True

                self._process_tool_results(tool_results)

                # Add reminder message after search tool results (Onyx pattern)
                if ran_search:
                    self.add_user_message(CITATION_REMINDER)

            if not step_result.should_continue:
                self.state.should_stop = True
                break

        final_content = self._get_final_content()
        citations = self._build_citations()

        return AgentResponse(
            content=final_content,
            citations=citations,
            tool_calls=self.state.tool_calls,
            messages=self.state.messages,
            metadata={
                "steps": self.state.step_count,
                "tokens": self.state.total_tokens,
            },
        )
