from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from dataclasses import dataclass

from agent.llm import Agent
from agent.llm.errors import MaxIterationsExceeded
from agent.llm.events import EventBus, ThinkingChunkReceived, ToolExecutionStarted, UsageUpdated
from agent.llm.providers.openai import OpenAIAdapter
from agent.llm.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

from ..permissions import PermissionCallback, PermissionGate
from ..router import Session, SYSTEM_PROMPT
from ..subagent import DoneEvent, InferEndEvent, LogEvent, SubAgentEvent, SubAgentStartEvent, ThinkingTokenEvent
from ..tools import make_tools


@dataclass
class Artifact:
    content: str

_TOOL_INSTRUCTION = (
    "the <workspace> block contains verified metadata about this repository: proj_brief, tech_stack, primary_languages, branch, and domain_map; "
    "if these fields fully answer the question, respond directly without using tools; "
    "if the question requires file contents, implementation details, logic, or architecture depth, "
    "you MUST use tools to read actual files — do not guess or rely on training knowledge; when multiple targets are nearby, prefer one wider ranged read_file call over many individual reads"
)

_QUERY_COLOR = "#4169E1"


def _fmt_tool_call(call: ToolUseBlock) -> str:
    inp = call.input or {}
    if call.name == "read_file":
        return f"Read {inp.get('path', '')}"
    if call.name == "list_files":
        return f"List {inp.get('pattern', '')}"
    if call.name == "grep":
        pat = inp.get("pattern", "")
        path = inp.get("path", "")
        return f"Grep {pat}" + (f" in {path}" if path else "")
    return call.name.capitalize()


class QueryHandler:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._extra_params = extra_params or {}

    async def stream(
        self,
        session: Session,
        user_input: str,
        permission_callback: PermissionCallback | None = None,
    ) -> AsyncIterator[SubAgentEvent | str]:
        bus = EventBus()
        adapter = OpenAIAdapter(api_key=self._api_key, base_url=self._api_base)
        agent = Agent(
            provider=adapter,
            model=self._model,
            system=f"{SYSTEM_PROMPT}\n\n{_TOOL_INSTRUCTION}",
            event_bus=bus,
            extra_params=self._extra_params,
        )
        for t in make_tools(session.working_dir):
            agent.tools.register(t)
        agent.tools.set_gate(PermissionGate(
            permissions=session.permissions,
            on_request=permission_callback,
        ))

        queue: asyncio.Queue[LogEvent | InferEndEvent | ThinkingTokenEvent | None] = asyncio.Queue()

        async def _consume_bus() -> None:
            async for event in bus.stream():
                if isinstance(event, ToolExecutionStarted):
                    await queue.put(LogEvent(message=_fmt_tool_call(event.call), tool_name=event.call.name))
                elif isinstance(event, UsageUpdated) and event.delta:
                    await queue.put(InferEndEvent(
                        prompt_tokens=event.delta.get("input_tokens"),
                        completion_tokens=event.delta.get("output_tokens"),
                    ))
                elif isinstance(event, ThinkingChunkReceived):
                    await queue.put(ThinkingTokenEvent(text=event.text))
            await queue.put(None)

        yield SubAgentStartEvent(name="Query", description="Inspecting workspace", color=_QUERY_COLOR)

        prior = [Message(role=m["role"], content=m["content"]) for m in session.messages[1:-1]]
        prior.append(Message(role="user", content=user_input))
        agent_task: asyncio.Task = asyncio.create_task(agent.run(prior))
        bus_task: asyncio.Task = asyncio.create_task(_consume_bus())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

            try:
                history = await agent_task
            except MaxIterationsExceeded:
                yield DoneEvent(thinking_chars=0)
                return

            thinking_chars = sum(
                len(b.text)
                for msg in history
                if msg.role == "assistant" and isinstance(msg.content, list)
                for b in msg.content
                if isinstance(b, ThinkingBlock)
            )
            yield DoneEvent(thinking_chars=thinking_chars)

            last = history[-1]
            if isinstance(last.content, list):
                text = "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
            else:
                text = last.content or ""
            if text:
                yield text

            artifact = _synthesize_artifact(history)
            if artifact:
                yield Artifact(content=f"[artifact] {artifact}")
        finally:
            if not agent_task.done():
                agent_task.cancel()
            if not bus_task.done():
                bus_task.cancel()


def _synthesize_artifact(history: list) -> str | None:
    """Very conservative: list files actually read + one short fact from final answer."""
    files: list[str] = []
    for msg in history:
        if hasattr(msg, "content") and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolUseBlock) and block.name in ("read_file", "grep", "list_files"):
                    inp = block.input or {}
                    if block.name == "read_file" and inp.get("path"):
                        files.append(inp["path"])
                    elif block.name == "grep" and inp.get("path"):
                        files.append(inp["path"])
    files = list(dict.fromkeys(files))[:6]  # dedup, cap

    final = ""
    if history:
        last = history[-1]
        if isinstance(last.content, list):
            final = " ".join(b.text for b in last.content if isinstance(b, TextBlock))
        else:
            final = last.content or ""
    summary = (final[:140] + "…") if len(final) > 140 else final

    if not files and not summary:
        return None
    parts = []
    if files:
        parts.append("files: " + ", ".join(files))
    if summary:
        parts.append("summary: " + summary)
    return "; ".join(parts)
