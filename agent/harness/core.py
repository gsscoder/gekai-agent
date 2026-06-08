from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from agent.llm import Agent
from agent.llm.errors import MaxIterationsExceeded
from agent.llm.events import EventBus, ThinkingChunkReceived, ToolExecutionCompleted, ToolExecutionStarted, UsageUpdated
from agent.llm.providers.openai import OpenAIAdapter
from agent.llm.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

from pathlib import Path

from ..permissions import PermissionCallback, PermissionGate
from ..persistence import append_debug
from ..persona import SYSTEM_PROMPT, render_tool_instruction
from ..session import Session
from ..settings import Permissions
from ..diff import build_diff
from ..events import DiffEvent, DoneEvent, InferEndEvent, LogEvent, MaxIterationsEvent, AgentEvent, SubAgentStartEvent, ThinkingTokenEvent
from ..subagents import Subagent
from ..tools import make_tools

_MAIN_COLOR = "#4169E1"
_RECENCY_N = 2


def _recency_turns(messages: list[dict], n: int) -> list[Message]:
    turns = [m for m in messages[:-1] if m["role"] in ("user", "assistant")]
    return [Message(role=m["role"], content=m["content"]) for m in turns[-(n * 2):]]


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
    if call.name == "edit_file":
        return f"Edit {inp.get('path', '')}"
    if call.name == "write_file":
        return f"Write {inp.get('path', '')}"
    if call.name == "move_file":
        return f"Move {inp.get('src', '')} → {inp.get('dst', '')}"
    if call.name == "copy_file":
        return f"Copy {inp.get('src', '')} → {inp.get('dst', '')}"
    if call.name == "delete_file":
        return f"Delete {inp.get('path', '')}"
    if call.name == "make_dir":
        return f"Mkdir {inp.get('path', '')}"
    if call.name == "run_command":
        return f"Run {inp.get('command', '')[:60]}"
    return call.name.capitalize()


def _build_agent(
    model: str,
    api_key: str | None,
    api_base: str | None,
    extra_params: dict,
    working_dir: Path,
    permissions: Permissions,
    permission_callback: PermissionCallback | None,
    system_base: str,
    bus: EventBus | None = None,
    subagent: Subagent | None = None,
) -> Agent:
    if subagent and subagent.permissions is not None:
        effective = Permissions(
            read=permissions.read and subagent.permissions.read,
            write=permissions.write and subagent.permissions.write,
            exec=permissions.exec and subagent.permissions.exec,
        )
    else:
        effective = permissions

    selected = []
    for t in make_tools(working_dir):
        if subagent and subagent.tools is not None and t.name not in subagent.tools:
            continue
        perm = t.required_permission
        if perm != "none" and not getattr(effective, perm, False) and permission_callback is None:
            continue
        selected.append(t)

    system = f"{system_base}\n<tools>\n{render_tool_instruction([t.name for t in selected])}"

    adapter = OpenAIAdapter(api_key=api_key, base_url=api_base)
    agent = Agent(
        provider=adapter,
        model=model,
        system=system,
        event_bus=bus,
        extra_params=extra_params,
    )
    for t in selected:
        agent.tools.register(t)
    agent.tools.set_gate(PermissionGate(
        permissions=effective,
        on_request=permission_callback,
    ))
    return agent


class Harness:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict | None = None,
        debug: bool = False,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._extra_params = extra_params or {}
        self._debug = debug

    async def stream(
        self,
        session: Session,
        user_input: str,
        permission_callback: PermissionCallback | None = None,
        subagent: Subagent | None = None,
    ) -> AsyncIterator[AgentEvent | str]:
        bus = EventBus()
        system_base = subagent.build_system_base() if subagent else SYSTEM_PROMPT
        agent = _build_agent(
            self._model, self._api_key, self._api_base, self._extra_params,
            session.working_dir, session.permissions, permission_callback, system_base, bus,
            subagent=subagent,
        )
        if self._debug:
            append_debug(session, {"content": agent.system})

        queue: asyncio.Queue[LogEvent | InferEndEvent | ThinkingTokenEvent | None] = asyncio.Queue()

        async def _consume_bus() -> None:
            async for event in bus.stream():
                if isinstance(event, ToolExecutionStarted):
                    await queue.put(LogEvent(message=_fmt_tool_call(event.call), tool_name=event.call.name))
                elif isinstance(event, ToolExecutionCompleted):
                    if event.call.name == "edit_file":
                        inp = event.call.input or {}
                        old_str = inp.get("old_str", "")
                        new_str = inp.get("new_str", "")
                        if old_str != new_str:
                            diff_lines = build_diff(old_str, new_str)
                            await queue.put(DiffEvent(path=inp.get("path", ""), diff_lines=diff_lines))
                elif isinstance(event, UsageUpdated) and event.delta:
                    await queue.put(InferEndEvent(
                        prompt_tokens=event.delta.get("input_tokens"),
                        completion_tokens=event.delta.get("output_tokens"),
                    ))
                elif isinstance(event, ThinkingChunkReceived):
                    await queue.put(ThinkingTokenEvent(text=event.text))
            await queue.put(None)

        yield SubAgentStartEvent(
            name=subagent.name if subagent else "main",
            description=subagent.description if subagent else "thinking",
            color=_MAIN_COLOR,
        )

        prior = [] if subagent else _recency_turns(session.messages, _RECENCY_N)
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
