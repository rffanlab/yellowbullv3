"""Anthropic Claude LLM provider."""

import json
from typing import Any, AsyncIterator

import httpx
from anthropic import AsyncAnthropic

from llm.base import BaseLLM, LLMResponse, Message, Role, StreamChunk


class AnthropicLLM(BaseLLM):
    """Anthropic API provider with streaming support."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        client_kwargs = {
            "api_key": api_key or "dummy-key",
            "timeout": httpx.Timeout(timeout),
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**client_kwargs)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[StreamChunk]:
        """Chat with Anthropic API."""
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            elif msg.role == Role.TOOL:
                # Anthropic uses tool_result content block
                if not anthropic_messages or anthropic_messages[-1]["role"] != "user":
                    anthropic_messages.append({"role": "user", "content": []})
                anthropic_messages[-1]["content"].append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "",
                })
            elif msg.role == Role.ASSISTANT:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fc = tc.get("function", {})
                        content.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fc.get("name", ""),
                            "input": self._parse_args(fc.get("arguments", "{}")),
                        })
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                anthropic_messages.append({
                    "role": msg.role.value,
                    "content": msg.content or "",
                })

        if stream:
            return self._stream_chat(system_prompt, anthropic_messages, tools)

        kwargs = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        response = await self.client.messages.create(**kwargs)
        return self._parse_response(response)

    async def _stream_chat(
        self,
        system_prompt: str | None,
        messages: list[dict],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat with Anthropic API."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "text") and delta.text:
                        yield StreamChunk(delta=delta.text)
                elif event.type == "content_block_start":
                    block = event.content_block
                    if hasattr(block, "type") and block.type == "tool_use":
                        tc = {
                            "id": getattr(block, "id", ""),
                            "function": {
                                "name": getattr(block, "name", ""),
                                "arguments": "",
                            },
                        }
                        yield StreamChunk(tool_call=tc)

        yield StreamChunk(done=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic tokenizer."""
        import anthropic

        # Use a simple estimation if tokenizer not available
        return len(text.encode("utf-8")) // 4

    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict]:
        """Convert OpenAI-style tools to Anthropic format."""
        result = []
        for tool in tools:
            anthropic_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
            }
            result.append(anthropic_tool)
        return result

    def _parse_args(self, args: str | dict) -> dict:
        """Parse function arguments."""
        if isinstance(args, dict):
            return args
        try:
            return json.loads(args) or {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _parse_response(self, response) -> LLMResponse:
        """Parse Anthropic response to LLMResponse."""
        content = ""
        tool_calls = []

        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input or {}),
                        },
                    })

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": (
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                ),
            }

        return LLMResponse(content=content or None, tool_calls=tool_calls, usage=usage)
