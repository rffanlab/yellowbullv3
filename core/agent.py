"""Agent core — main orchestration loop with tool calling."""

import json
import uuid
from dataclasses import dataclass, field

from config.settings import Settings
from core.context_builder import ContextBuilder
from core.logging_setup import get_logger
from core.session_manager import SessionManager
from llm.base import BaseLLM
from models.message import Message, MessageRole
from models.session import Session
from tools.registry import ToolRegistry

logger = get_logger(__name__)


@dataclass
class ChatRequest:
    session_id: str | None  # None means create new session
    user_id: str
    message: str


@dataclass
class ChatResponse:
    content: str
    session_id: str
    tool_results: list[dict] = field(default_factory=list)
    needs_clarification: bool = False
    usage: dict | None = None


class Agent:
    """Main agent orchestrator with ReAct-style tool calling loop."""

    def __init__(self, settings: Settings, llm: BaseLLM):
        self.settings = settings
        self.llm = llm
        self.session_manager = SessionManager()
        self.context_builder = ContextBuilder(settings.agent.system_prompt)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Main entry point: process user request with tool calling loop."""
        session = self._get_or_create_session(request)
        logger.info("Processing request", extra={
            "session_id": session.session_id,
            "user_id": request.user_id,
        })

        # Add user message to session
        user_msg = Message(
            id=str(uuid.uuid4()),
            role=MessageRole.USER,
            content=request.message,
        )
        session.add_message(user_msg)

        tool_results_log: list[dict] = []

        # Main loop: LLM -> tools -> LLM until final answer
        max_depth = self.settings.agent.max_chain_depth
        for depth in range(max_depth):
            session.state.chain_depth += 1
            logger.debug(f"Chain depth {depth + 1}/{max_depth}", extra={
                "session_id": session.session_id,
            })

            # Build context and call LLM
            context = self.context_builder.build(
                session, self.settings.agent.context_window
            )
            tools = ToolRegistry.to_function_definitions()
            response = await self.llm.chat(messages=context, tools=tools)

            # No tool calls -> return final answer
            if not response.tool_calls:
                assistant_msg = Message(
                    id=str(uuid.uuid4()),
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                )
                session.add_message(assistant_msg)
                logger.info("Final response generated", extra={
                    "session_id": session.session_id,
                    "depth": depth + 1,
                })
                return ChatResponse(
                    content=response.content or "",
                    session_id=session.session_id,
                    usage=response.usage,
                )

            # Record assistant message with tool calls
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
            session.add_message(assistant_msg)

            # Execute tools in parallel
            results = await self._execute_tools(response.tool_calls, session)
            tool_results_log.extend(results)

            # Add tool results back to context for next iteration
            for tr in results:
                tool_msg = Message(
                    id=str(uuid.uuid4()),
                    role=MessageRole.TOOL,
                    content=tr["content"],
                    tool_call_id=tr["tool_call_id"],
                )
                session.add_message(tool_msg)

        # Exceeded max depth — return fallback response
        logger.warning("Max chain depth exceeded", extra={
            "session_id": session.session_id,
        })
        fallback = "抱歉，任务处理步骤过多，请简化您的请求。"
        session.add_message(Message(
            id=str(uuid.uuid4()),
            role=MessageRole.ASSISTANT,
            content=fallback,
        ))
        return ChatResponse(content=fallback, session_id=session.session_id)

    async def _execute_tools(self, tool_calls: list[dict], session: Session) -> list[dict]:
        """Execute multiple tool calls in parallel."""
        import asyncio

        max_calls = self.settings.agent.max_tool_calls_per_turn
        tasks = [self._run_single_tool(tc) for tc in tool_calls[:max_calls]]
        return await asyncio.gather(*tasks)

    async def _run_single_tool(self, tool_call: dict) -> dict:
        """Execute a single tool call with retry logic."""
        func_name = tool_call["function"]["name"]
        try:
            func_args = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            return {
                "tool_call_id": tool_call.get("id", ""),
                "content": f"工具 '{func_name}' 参数解析失败",
            }

        tool = ToolRegistry.get(func_name)
        if not tool:
            return {
                "tool_call_id": tool_call.get("id", ""),
                "content": f"未知工具: {func_name}",
            }

        # Retry loop
        retry_limit = self.settings.agent.tool_retry_limit
        for attempt in range(retry_limit):
            result = await tool.execute(**func_args)
            if result.success:
                return {
                    "tool_call_id": tool_call.get("id", ""),
                    "content": result.content,
                }
            logger.warning(
                f"Tool '{func_name}' attempt {attempt + 1} failed: {result.content}",
                extra={"session_id": None},
            )

        # All retries exhausted — return last error
        return {
            "tool_call_id": tool_call.get("id", ""),
            "content": f"工具 '{func_name}' 执行失败: {result.content}",
        }

    def _get_or_create_session(self, request: ChatRequest) -> Session:
        if request.session_id:
            session = self.session_manager.get(request.session_id)
            if session:
                return session
        return self.session_manager.create(request.user_id)
