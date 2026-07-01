from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from agent.llm import Agent
from agent.llm.errors import MaxIterationsExceeded
from agent.llm.events import AgentStopped, Event as LlmEvent, EventBus, ThinkingChunkReceived, ToolExecutionCompleted, ToolExecutionStarted, UsageUpdated
from agent.llm.providers.openai import OpenAIAdapter
from agent.llm.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

from pathlib import Path

from ..permissions import PermissionCallback, PermissionGate
from ..persistence import append_debug
from ..persona import SYSTEM_PROMPT, render_tool_instruction
from ..session import Session
from ..settings import Permissions
from ..diff import build_diff
from ..events import BudgetExhaustedEvent, DiffEvent, DoneEvent, InferEndEvent, LogEvent, MaxIterationsEvent, AgentEvent, SubAgentStartEvent, ThinkingTokenEvent
from ..shell import resolve_shell
from ..subagents import SUBAGENTS, Subagent
from ..tools import HiddenGrantCallback, make_tools

_MAIN_COLOR = "#4169E1"
_RECENCY_N = 2

_MENTION_TRIGGER = r"(?:use|have|ask|let|delegate\s+to|call)"
_NEGATION_CUES = re.compile(
    r"\b(?:don't|do not|doesn't|does not|didn't|did not|never|avoid|shouldn't|should not|"
    r"isn't|isn’t|without|no need to|not)\b",
    re.IGNORECASE,
)
_NEGATION_LOOKBACK_CHARS = 20


def detect_subagent_mentions(user_input: str) -> list[str]:
    """Roster-order list of invocable subagent names the user explicitly asked
    to be used (e.g. "use code-expert", "have test-fixer solve..."), skipping
    matches preceded by a nearby negation cue ("don't use X"). Best-effort,
    not airtight — a deliberately conservative trigger-verb + lookback-window
    heuristic, not full NLP negation scope resolution.
    """
    roster = [s.name for s in SUBAGENTS if s.user_invocable]
    found: list[str] = []
    for name in roster:
        pattern = re.compile(rf"\b{_MENTION_TRIGGER}\s+{re.escape(name)}\b", re.IGNORECASE)
        match = pattern.search(user_input)
        if match is None:
            continue
        window = user_input[max(0, match.start() - _NEGATION_LOOKBACK_CHARS):match.start()]
        if _NEGATION_CUES.search(window):
            continue
        found.append(name)
    return found


def _subagents_request_block(names: list[str]) -> str:
    roster = "\n".join(names)
    return (
        "\n<subagents_request>\n"
        "the user explicitly named the following specialist(s) this turn — call `delegate` for "
        "each one instead of doing their work yourself, even if you judge the task simple enough "
        "to handle directly:\n"
        f"{roster}\n"
        "</subagents_request>"
    )


def _enrich_system_base(system_base: str, working_dir: Path) -> str:
    is_empty = not any(p for p in working_dir.iterdir() if p.name != ".gekai")
    return system_base + (
        f"\nworking root directory: {working_dir}"
        f"\nfile tool paths are relative to this root"
        f"\nthe directory name is only a label — do not infer requirements from it or use it to add unrequested features or complexity"
        + ("\nthis directory is empty — do not create a redundant wrapper subdirectory mirroring the project name; package layout (src/, tests/, etc.) is fine" if is_empty else "")
    )


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


_DEBUG_TRUNCATE_LIMIT = 1000


def _truncate_debug_text(text: str) -> str:
    if len(text) <= _DEBUG_TRUNCATE_LIMIT:
        return text
    return text[:_DEBUG_TRUNCATE_LIMIT] + f"…+{len(text) - _DEBUG_TRUNCATE_LIMIT} more chars"


def _fmt_debug_tool_input(call: ToolUseBlock) -> dict:
    """Debug-log representation of a tool call's input.

    write_file/edit_file inputs carry full file contents or old/new diff strings —
    these are logged as path-only (the path is never truncated; it's always short and
    is the only part of those inputs that's useful for debugging without bloating the
    debug log with entire file bodies). All other tools get their full input dict,
    JSON-serialized and truncated like any other debug text.
    """
    inp = call.input or {}
    if call.name in ("write_file", "edit_file"):
        return {"name": call.name, "path": inp.get("path", "")}
    return {"name": call.name, "input": _truncate_debug_text(json.dumps(inp, default=str))}


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
    hidden_grant_callback: HiddenGrantCallback | None = None,
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
    for t in make_tools(working_dir, grant_cb=hidden_grant_callback):
        if subagent and subagent.tools is not None and t.name not in subagent.tools:
            continue
        perm = t.required_permission
        if perm != "none" and not getattr(effective, perm, False) and permission_callback is None:
            continue
        selected.append(t)

    system = f"{system_base}\n<tools>\n{render_tool_instruction([t.name for t in selected], shell_kind=resolve_shell().kind)}"

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

    if subagent is None:
        from ..tools.delegate import make_delegate_tool
        delegate_t = make_delegate_tool(
            model=model,
            api_key=api_key,
            api_base=api_base,
            extra_params=extra_params,
            working_dir=working_dir,
            permissions=permissions,
            permission_callback=permission_callback,
            bus=bus,
            hidden_grant_callback=hidden_grant_callback,
        )
        agent.tools.register(delegate_t)

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
        extra_params: dict | None = None,
        hidden_grant_callback: HiddenGrantCallback | None = None,
    ) -> AsyncIterator[AgentEvent | str]:
        bus = EventBus()
        system_base = subagent.build_system_base() if subagent else SYSTEM_PROMPT
        system_base = _enrich_system_base(system_base, session.working_dir)
        if subagent is None:
            mentions = detect_subagent_mentions(user_input)
            if mentions:
                system_base += _subagents_request_block(mentions)
        effective_extra_params = self._extra_params if extra_params is None else extra_params
        agent = _build_agent(
            self._model, self._api_key, self._api_base, effective_extra_params,
            session.working_dir, session.permissions, permission_callback, system_base, bus,
            subagent=subagent,
            hidden_grant_callback=hidden_grant_callback,
        )
        if self._debug:
            append_debug(session, {"content": {"system": agent.system, "extra_params": effective_extra_params}})

        queue: asyncio.Queue[
            LogEvent | DiffEvent | InferEndEvent | ThinkingTokenEvent | BudgetExhaustedEvent | None
        ] = asyncio.Queue()
        files_touched: list[str] = []

        _SINGLE_PATH_TOOLS = ("write_file", "edit_file", "make_dir", "delete_file")
        _DUAL_PATH_TOOLS = ("move_file", "copy_file")

        def _on_event(event: LlmEvent) -> None:
            # Synchronous subscriber, not an asyncio task racing `agent_task`:
            # a task-based consumer isn't guaranteed to be scheduled before a
            # single-turn agent run completes and emits AgentStopped, which
            # silently drops every event and hangs `queue.get()` forever.
            if isinstance(event, ToolExecutionStarted):
                queue.put_nowait(LogEvent(message=_fmt_tool_call(event.call), tool_name=event.call.name))
                if self._debug:
                    append_debug(session, {"content": {"tool_call": _fmt_debug_tool_input(event.call)}})
            elif isinstance(event, ToolExecutionCompleted):
                if event.call.name == "edit_file":
                    inp = event.call.input or {}
                    old_str = inp.get("old_str", "")
                    new_str = inp.get("new_str", "")
                    if old_str != new_str:
                        diff_lines = build_diff(old_str, new_str)
                        queue.put_nowait(DiffEvent(path=inp.get("path", ""), diff_lines=diff_lines))
                elif event.call.name == "write_file":
                    inp = event.call.input or {}
                    content = inp.get("content", "")
                    if content:
                        diff_lines = build_diff("", content)
                        queue.put_nowait(DiffEvent(path=inp.get("path", ""), diff_lines=diff_lines))
                if not event.result.is_error:
                    inp = event.call.input or {}
                    if event.call.name in _SINGLE_PATH_TOOLS:
                        path = inp.get("path", "")
                        if path and path not in files_touched:
                            files_touched.append(path)
                    elif event.call.name in _DUAL_PATH_TOOLS:
                        for path in (inp.get("src", ""), inp.get("dst", "")):
                            if path and path not in files_touched:
                                files_touched.append(path)
                if self._debug:
                    append_debug(session, {
                        "content": {
                            "tool_result": {
                                "name": event.call.name,
                                "duration_s": round(event.duration_s, 3),
                                "is_error": event.result.is_error,
                                "result": _truncate_debug_text(event.result.content),
                            }
                        }
                    })
            elif isinstance(event, UsageUpdated) and event.delta:
                queue.put_nowait(InferEndEvent(
                    prompt_tokens=event.delta.get("input_tokens"),
                    completion_tokens=event.delta.get("output_tokens"),
                ))
            elif isinstance(event, ThinkingChunkReceived):
                queue.put_nowait(ThinkingTokenEvent(text=event.text))
            elif isinstance(event, AgentStopped):
                if event.budget_exhausted:
                    queue.put_nowait(BudgetExhaustedEvent())
                queue.put_nowait(None)

        yield SubAgentStartEvent(
            name=subagent.name if subagent else "main",
            description=subagent.description if subagent else "thinking",
            color=_MAIN_COLOR,
        )

        prior = [] if subagent else _recency_turns(session.messages, _RECENCY_N)
        prior.append(Message(role="user", content=user_input))
        unsubscribe = bus.subscribe(_on_event)
        agent_task: asyncio.Task = asyncio.create_task(agent.run(prior))

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
                yield DoneEvent(thinking_chars=0, files_touched=files_touched)
                return

            thinking_chars = sum(
                len(b.text)
                for msg in history
                if msg.role == "assistant" and isinstance(msg.content, list)
                for b in msg.content
                if isinstance(b, ThinkingBlock)
            )
            yield DoneEvent(thinking_chars=thinking_chars, files_touched=files_touched)

            text = ""
            for msg in reversed(history):
                if msg.role == "assistant":
                    if isinstance(msg.content, list):
                        text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
                    else:
                        text = msg.content or ""
                    if text:
                        break
            if text:
                yield text
        finally:
            if not agent_task.done():
                agent_task.cancel()
            unsubscribe()
