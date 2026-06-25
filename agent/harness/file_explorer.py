from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from ..llm import Agent
from ..llm.errors import MaxIterationsExceeded
from ..llm.events import EventBus, ThinkingChunkReceived, ToolExecutionStarted, UsageUpdated
from ..llm.providers.openai import OpenAIAdapter
from ..llm.types import Message, TextBlock, ThinkingBlock
from ..persistence import append_debug
from ..persona import _IDENTITY_SUB, _SHARED_BODY, render_tool_instruction
from ..session import Session
from ..events import DoneEvent, InferEndEvent, LogEvent, MaxIterationsEvent, AgentEvent, SubAgentStartEvent, ThinkingTokenEvent
from ..shell import resolve_shell
from ..tools import HiddenGrantCallback, make_tools
from .core import _fmt_tool_call, _recency_turns

_EXPLORE_COLOR = "#20B2AA"
_RECENCY_N = 2

_ROLE = (
    "your specialization is read-only investigation of the workspace — "
    "you DISCOVER and ANSWER — you never edit, create, move, or delete files\n"
)


class FileExplorer:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        debug: bool = False,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._debug = debug

    async def stream(
        self,
        session: Session,
        user_input: str,
        hidden_grant_callback: HiddenGrantCallback | None = None,
    ) -> AsyncIterator[AgentEvent | str]:
        bus = EventBus()
        selected = [t for t in make_tools(session.working_dir, grant_cb=hidden_grant_callback) if t.is_read_only]
        system = (
            _IDENTITY_SUB + _ROLE + _SHARED_BODY
            + "\n<tools>\n" + render_tool_instruction([t.name for t in selected], shell_kind=resolve_shell().kind)
        )

        adapter = OpenAIAdapter(
            api_key=self._api_key,
            base_url=self._api_base,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0),
        )
        agent = Agent(
            provider=adapter,
            model=self._model,
            system=system,
            event_bus=bus,
        )
        for t in selected:
            agent.tools.register(t)

        if self._debug:
            append_debug(session, {"content": {"system": agent.system}})

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

        yield SubAgentStartEvent(name="explore", description="exploring", color=_EXPLORE_COLOR)

        prior = _recency_turns(session.messages, _RECENCY_N)
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
                yield MaxIterationsEvent()
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
        finally:
            if not agent_task.done():
                agent_task.cancel()
            if not bus_task.done():
                bus_task.cancel()
