"""Agent core — main orchestration loop with tool calling."""

import json
import uuid
from dataclasses import dataclass, field

from config.settings import Settings
from core.context_builder import ContextBuilder
from core.logging_setup import get_logger
from core.session_manager import SessionManager
from core.tool_executor import ExecutionResult, ExecutionStatus, ToolCallRequest, ToolExecutor
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

        # Initialize tool executor from config
        tc = settings.agent.tools
        self.tool_executor = ToolExecutor(
            registry=ToolRegistry(),
            default_timeout=tc.tool_timeout_seconds,
            max_retries=settings.agent.tool_retry_limit,
            retry_backoff_factor=tc.retry_backoff_factor,
            parallel=True,
            enable_cache=tc.enable_cache,
        )

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

        tool_results_log: list[ExecutionResult] = []

        # Main loop: LLM -> tools -> LLM until final answer
        for depth in range(self.settings.agent.max_chain_depth):
            session.state.chain_depth += 1
            logger.debug(f"Chain depth {depth + 1}", extra={
                "session_id": session.session_id,
            })

            # Build context and call LLM
            context = self.context_builder.build(
                session, self.settings.agent.context_window, work_dir=session.work_dir
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

            # Execute tools via ToolExecutor (parallel by default)
            requests = self._build_tool_requests(response.tool_calls)
            batch_result = await self.tool_executor.execute_batch(requests)
            tool_results_log.extend(batch_result.results)

            # Add tool results back to context for next iteration
            for idx, tc in enumerate(response.tool_calls):
                exec_result = batch_result.results[idx] if idx < len(batch_result.results) else None
                content = exec_result.content if exec_result else "工具执行结果不可用"
                tool_msg = Message(
                    id=str(uuid.uuid4()),
                    role=MessageRole.TOOL,
                    content=content,
                    tool_call_id=tc.get("id", ""),
                )
                session.add_message(tool_msg)

        # Exceeded max depth — summarize what was accomplished so far
        logger.warning("Max chain depth exceeded", extra={
            "session_id": session.session_id,
        })

        # Collect tool results that were successful for a summary
        success_results = [r for r in tool_results_log if r.status == ExecutionStatus.SUCCESS]
        failed_results = [r for r in tool_results_log if r.status != ExecutionStatus.SUCCESS]

        parts = []
        if success_results:
            done = [f"- {r.tool_name}: {r.content[:100]}" for r in success_results]
            parts.append("已完成的操作：\n" + "\n".join(done))
        if failed_results:
            errors = [f"- {r.tool_name}: {r.error or 'unknown'}" for r in failed_results]
            parts.append("失败的操作：\n" + "\n".join(errors))

        fallback = "任务处理已达到最大步骤限制。"
        if parts:
            fallback += "\n\n" + "\n\n".join(parts)
        fallback += "\n\n请继续对话让我完成剩余部分，或分步提交子任务。"

        session.add_message(Message(
            id=str(uuid.uuid4()),
            role=MessageRole.ASSISTANT,
            content=fallback,
        ))
        return ChatResponse(content=fallback, session_id=session.session_id)

    def _build_tool_requests(self, tool_calls: list[dict]) -> list[ToolCallRequest]:
        """Convert LLM tool call dicts to ToolCallRequest objects."""
        max_calls = self.settings.agent.max_tool_calls_per_turn
        requests = []
        for tc in tool_calls[:max_calls]:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                func_args = {}
            requests.append(ToolCallRequest(
                tool_name=func_name,
                arguments=func_args,
            ))
        return requests

    def _get_or_create_session(self, request: ChatRequest) -> Session:
        if request.session_id:
            session = self.session_manager.get(request.session_id)
            if session:
                return session
        return self.session_manager.create(request.user_id)
