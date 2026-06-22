"""OpenAI-compatible LLM provider."""

import json
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI

from llm.base import BaseLLM, LLMResponse, Message, Role, StreamChunk


class OpenAILLM(BaseLLM):
    """OpenAI API provider with streaming support."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
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
        self.client = AsyncOpenAI(**client_kwargs)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[StreamChunk]:
        """Chat with OpenAI API."""
        formatted = self._format_messages(messages)

        if stream:
            return self._stream_chat(formatted, tools)

        kwargs = {
            "model": self.model_name,
            "messages": formatted,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    async def _stream_chat(
        self,
        messages: list[dict],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat with OpenAI API."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)

        accumulated_tool_calls: dict[str, Any] = {}
        has_tool_call = False

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta

            # Handle tool calls
            if delta.tool_calls:
                has_tool_call = True
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    call = accumulated_tool_calls[idx]
                    if tc.id:
                        call["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            call["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            call["function"]["arguments"] += tc.function.arguments

            # Handle text content
            text_delta = delta.content or ""
            yield StreamChunk(delta=text_delta)

        # Yield final tool calls
        if has_tool_calls:
            for call in accumulated_tool_calls.values():
                yield StreamChunk(tool_call=call)

        yield StreamChunk(done=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        import tiktoken

        enc = tiktoken.encoding_for_model(self.model_name)
        return len(enc.encode(text))

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal Message to OpenAI format."""
        result = []
        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role.value}

            if msg.role == Role.TOOL:
                entry["content"] = msg.content or ""
                entry["tool_call_id"] = msg.tool_call_id
                result.append(entry)
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                entry["content"] = msg.content
                entry["tool_calls"] = msg.tool_calls
                result.append(entry)
            else:
                entry["content"] = msg.content
                result.append(entry)

        return result

    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI response to LLMResponse."""
        choice = response.choices[0] if response.choices else None
        if not choice:
            return LLMResponse()

        content = choice.message.content
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type or "function",
                    "function": {
                        "name": tc.function.name if tc.function else "",
                        "arguments": tc.function.arguments if tc.function else "{}",
                    },
                })

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)
