from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from llmstitch import Agent
from llmstitch.events import EventBus, ToolExecutionStarted
from llmstitch.providers.openai import OpenAIAdapter
from llmstitch.types import Message, TextBlock, ToolUseBlock

from ..router import Session, SYSTEM_PROMPT
from ..subagent import DoneEvent, LogEvent, SubAgentEvent, SubAgentStartEvent
from ..tools import make_tools

_TOOL_INSTRUCTION = (
    "the <workspace> block contains verified metadata about this repository: proj_brief, tech_stack, primary_languages, branch, and domain_map; "
    "if these fields fully answer the question, respond directly without using tools; "
    "if the question requires file contents, implementation details, logic, or architecture depth, "
    "you MUST use tools to read actual files — do not guess or rely on training knowledge"
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
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    async def stream(self, session: Session, user_input: str) -> AsyncIterator[SubAgentEvent | str]:
        bus = EventBus()
        adapter = OpenAIAdapter(api_key=self._api_key, base_url=self._api_base)
        agent = Agent(
            provider=adapter,
            model=self._model,
            system=f"{SYSTEM_PROMPT}\n\n{_TOOL_INSTRUCTION}",
            event_bus=bus,
        )
        for t in make_tools(session.working_dir):
            agent.tools.register(t)

        queue: asyncio.Queue[LogEvent | None] = asyncio.Queue()

        async def _consume_bus() -> None:
            async for event in bus.stream():
                if isinstance(event, ToolExecutionStarted):
                    await queue.put(LogEvent(message=_fmt_tool_call(event.call), tool_name=event.call.name))
            await queue.put(None)

        yield SubAgentStartEvent(name="Query", description="Inspecting workspace", color=_QUERY_COLOR)

        prior = [Message(role=m["role"], content=m["content"]) for m in session.messages[1:-1]]
        prior.append(Message(role="user", content=user_input))
        agent_task = asyncio.create_task(agent.run(prior))
        asyncio.create_task(_consume_bus())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        history = await agent_task
        yield DoneEvent()

        last = history[-1]
        if isinstance(last.content, list):
            text = "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
        else:
            text = last.content or ""
        if text:
            yield text
