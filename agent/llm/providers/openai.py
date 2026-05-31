"""OpenAI Chat Completions adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..types import (
    CompletionResponse,
    ContentBlock,
    Message,
    MessageStop,
    StreamDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
    ToolUseStart,
    ToolUseStop,
)
from .base import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI's Chat Completions API (the `openai` SDK, >= 1.50)."""

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        import openai  # lazy

        self._client = openai.AsyncOpenAI(api_key=api_key, **client_kwargs)

    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> CompletionResponse:
        translated = self.translate_messages(messages, system=system)
        payload: dict[str, Any] = {
            "model": model,
            "messages": translated,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = self.translate_tools(tools)
        payload.update(kwargs)
        response = await self._client.chat.completions.create(**payload)
        return self.parse_response(response)

    @staticmethod
    def translate_messages(
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "system":
                content = msg.content if isinstance(msg.content, str) else ""
                out.append({"role": "system", "content": content})
                continue
            if isinstance(msg.content, str):
                out.append({"role": msg.role, "content": msg.content})
                continue
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results and msg.role == "user":
                for tr in tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": tr.content,
                        }
                    )
                remaining = [b for b in msg.content if not isinstance(b, ToolResultBlock)]
                if remaining:
                    text_parts = [b.text for b in remaining if isinstance(b, TextBlock)]
                    if text_parts:
                        out.append({"role": "user", "content": "\n".join(text_parts)})
                continue
            text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]
            entry: dict[str, Any] = {"role": msg.role}
            thinking_parts = [b.text for b in msg.content if isinstance(b, ThinkingBlock)]
            if thinking_parts:
                entry["reasoning_content"] = "".join(thinking_parts)
            if text_parts:
                entry["content"] = "\n".join(text_parts)
            else:
                entry["content"] = None
            if tool_uses:
                entry["tool_calls"] = [
                    {
                        "id": tu.id,
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(tu.input),
                        },
                    }
                    for tu in tool_uses
                ]
            out.append(entry)
        return out

    @staticmethod
    def translate_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    @staticmethod
    def parse_response(response: Any) -> CompletionResponse:
        choice = response.choices[0]
        message = choice.message
        content: list[ContentBlock] = []
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            content.append(ThinkingBlock(text=reasoning))
        text = getattr(message, "content", None)
        if text:
            content.append(TextBlock(text=text))
        for call in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments) if call.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": call.function.arguments}
            content.append(ToolUseBlock(id=call.id, name=call.function.name, input=args))
        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "input_tokens": getattr(raw_usage, "prompt_tokens", 0),
                "output_tokens": getattr(raw_usage, "completion_tokens", 0),
            }
        return CompletionResponse(
            content=content,
            stop_reason=choice.finish_reason or "stop",
            usage=usage,
            raw=response,
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        translated = self.translate_messages(messages, system=system)
        payload: dict[str, Any] = {
            "model": model,
            "messages": translated,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = self.translate_tools(tools)
        payload.update(kwargs)

        text_buf: list[str] = []
        thinking_buf: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        tool_call_order: list[int] = []
        stop_reason = "stop"
        usage: dict[str, int] | None = None

        stream = await self._client.chat.completions.create(**payload)
        async for chunk in stream:
            raw_usage = getattr(chunk, "usage", None)
            if raw_usage is not None:
                usage = {
                    "input_tokens": getattr(raw_usage, "prompt_tokens", 0),
                    "output_tokens": getattr(raw_usage, "completion_tokens", 0),
                }
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            finish = getattr(choice, "finish_reason", None)
            if finish:
                stop_reason = finish
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            text_piece = getattr(delta, "content", None)
            if text_piece:
                text_buf.append(text_piece)
                yield TextDelta(text=text_piece)
            reasoning_piece = getattr(delta, "reasoning_content", None)
            if reasoning_piece:
                thinking_buf.append(reasoning_piece)
                yield ThinkingDelta(text=reasoning_piece)
            for tc_delta in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc_delta, "index", 0)
                fn = getattr(tc_delta, "function", None)
                if idx not in tool_calls:
                    tc_id = getattr(tc_delta, "id", "") or ""
                    tc_name = getattr(fn, "name", "") or "" if fn is not None else ""
                    tool_calls[idx] = {"id": tc_id, "name": tc_name, "arguments": ""}
                    tool_call_order.append(idx)
                    yield ToolUseStart(id=tc_id, name=tc_name)
                args_piece = getattr(fn, "arguments", None) if fn is not None else None
                if args_piece:
                    tool_calls[idx]["arguments"] += args_piece
                    yield ToolUseDelta(id=tool_calls[idx]["id"], partial_json=args_piece)

        for idx in tool_call_order:
            yield ToolUseStop(id=tool_calls[idx]["id"])
        yield MessageStop(stop_reason=stop_reason, usage=usage)

        content: list[ContentBlock] = []
        thinking = "".join(thinking_buf)
        if thinking:
            content.append(ThinkingBlock(text=thinking))
        text = "".join(text_buf)
        if text:
            content.append(TextBlock(text=text))
        for idx in tool_call_order:
            tc = tool_calls[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": tc["arguments"]}
            content.append(ToolUseBlock(id=tc["id"], name=tc["name"], input=args))
        yield StreamDone(
            response=CompletionResponse(
                content=content,
                stop_reason=stop_reason,
                usage=usage,
                raw=None,
            )
        )

    @classmethod
    def default_retryable(cls) -> tuple[type[BaseException], ...]:
        import openai  # lazy

        return (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )
