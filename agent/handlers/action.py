from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from dataclasses import dataclass

from agent.llm import Agent
from agent.llm.errors import MaxIterationsExceeded
from agent.llm.events import EventBus, ThinkingChunkReceived, ToolExecutionCompleted, ToolExecutionStarted, UsageUpdated
from agent.llm.providers.openai import OpenAIAdapter
from agent.llm.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

from pathlib import Path

from ..permissions import PermissionCallback, PermissionGate
from ..profiles import AgentProfile
from ..router import Session, SYSTEM_PROMPT
from ..settings import Permissions
from ..diff import build_diff
from ..subagent import DiffEvent, DoneEvent, InferEndEvent, LogEvent, SubAgentEvent, SubAgentStartEvent, ThinkingTokenEvent
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

_ACTION_COLOR = "#4169E1"


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
    return call.name.capitalize()


def _build_agent(
    model: str,
    api_key: str | None,
    api_base: str | None,
    extra_params: dict,
    working_dir: Path,
    permissions: Permissions,
    permission_callback: PermissionCallback | None,
    bus: EventBus | None = None,
    profile: AgentProfile | None = None,
) -> Agent:
    adapter = OpenAIAdapter(api_key=api_key, base_url=api_base)
    directives = f"\n\n{profile.directives}" if profile else ""
    agent = Agent(
        provider=adapter,
        model=model,
        system=f"{SYSTEM_PROMPT}{directives}\n\n{_TOOL_INSTRUCTION}",
        event_bus=bus,
        extra_params=extra_params,
    )
    for t in make_tools(working_dir):
        if profile and profile.tools is not None and t.name not in profile.tools:
            continue
        agent.tools.register(t)
    if profile and profile.permissions is not None:
        effective = Permissions(
            read=permissions.read and profile.permissions.read,
            write=permissions.write and profile.permissions.write,
            exec=permissions.exec and profile.permissions.exec,
        )
    else:
        effective = permissions
    agent.tools.set_gate(PermissionGate(
        permissions=effective,
        on_request=permission_callback,
    ))
    return agent


class ActionHandler:
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
        profile: AgentProfile | None = None,
    ) -> AsyncIterator[SubAgentEvent | str]:
        bus = EventBus()
        agent = _build_agent(
            self._model, self._api_key, self._api_base, self._extra_params,
            session.working_dir, session.permissions, permission_callback, bus,
            profile=profile,
        )

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
            name=profile.name if profile else "Action",
            description=profile.description if profile else "Inspecting workspace",
            color=_ACTION_COLOR,
        )

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
