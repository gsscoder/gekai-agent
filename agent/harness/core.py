from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from agent.llm import Agent
from agent.llm.errors import MaxIterationsExceeded
from agent.llm.events import AgentStopped, DelegationCompleted, DelegationStarted, Event as LlmEvent, EventBus, ThinkingChunkReceived, ToolExecutionCompleted, ToolExecutionStarted, UsageUpdated
from agent.llm.providers.openai import OpenAIAdapter
from agent.llm.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

from pathlib import Path

from ..permissions import PermissionCallback, PermissionGate
from ..persistence import append_debug
from ..persona import SYSTEM_PROMPT, render_tool_instruction
from ..pipeline.estimate import Estimator
from ..pipeline.plan import Plan, PlanStep
from ..pipeline.planner import Planner
from ..harness.interpreter import PlanHalted, StepResult, run_plan
from ..session import Session
from ..settings import Permissions
from ..diff import build_diff
from ..events import BudgetExhaustedEvent, DelegationDoneEvent, DelegationStartEvent, DiffEvent, DoneEvent, EstimateEvent, InferEndEvent, LogEvent, MaxIterationsEvent, AgentEvent, PlanHaltedEvent, PlanStartedEvent, SubAgentStartEvent, ThinkingTokenEvent
from ..shell import resolve_shell
from ..subagents import SUBAGENTS, Subagent
from ..tools import HiddenGrantCallback, make_tools
from ..tools.delegate import run_subagent

_MAIN_COLOR = "#4169E1"
_RECENCY_N = 2


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

    # No `delegate` tool is registered — main is a pure work-operator; the
    # harness (planner + interpreter) owns all cross-agent control flow
    # (plan 27 decision 11, superseding plan 25's agents-as-tools).
    return agent


class Harness:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict | None = None,
        debug: bool = False,
        supp_model: str | None = None,
        supp_api_key: str | None = None,
        supp_api_base: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._extra_params = extra_params or {}
        self._debug = debug
        self._estimator = (
            Estimator(model=supp_model, api_key=supp_api_key, api_base=supp_api_base)
            if supp_model
            else None
        )
        # planner runs CORE thinking (same model/creds as main) — one call per
        # mutation turn (plan 27 decision 5)
        self._planner = Planner(
            model=model, api_key=api_key, api_base=api_base, extra_params=self._extra_params,
        )

    async def stream(
        self,
        session: Session,
        user_input: str,
        permission_callback: PermissionCallback | None = None,
        subagent: Subagent | None = None,
        extra_params: dict | None = None,
        hidden_grant_callback: HiddenGrantCallback | None = None,
        seed: str | None = None,
    ) -> AsyncIterator[AgentEvent | str]:
        bus = EventBus()
        system_base = subagent.build_system_base() if subagent else SYSTEM_PROMPT
        system_base = _enrich_system_base(system_base, session.working_dir)
        effective_extra_params = self._extra_params if extra_params is None else extra_params

        estimate_decision = "skipped"
        estimate_duration_ms = 0
        if subagent is None:
            if seed is not None:
                estimate_decision = "seeded"
            elif self._estimator is not None:
                t0 = time.monotonic()
                estimate = await self._estimator.estimate(user_input)
                estimate_duration_ms = round((time.monotonic() - t0) * 1000)
                estimate_decision = "mutate" if estimate.mutate else "trivial"
            else:
                estimate_decision = "trivial"  # no estimator wired — safest default, no planning
        yield EstimateEvent(decision=estimate_decision, specialists=[], duration_ms=estimate_duration_ms)

        if subagent is None and estimate_decision in ("mutate", "seeded"):
            async for item in self._stream_plan(
                session, user_input, seed, permission_callback, hidden_grant_callback, bus,
            ):
                yield item
            return

        agent = _build_agent(
            self._model, self._api_key, self._api_base, effective_extra_params,
            session.working_dir, session.permissions, permission_callback, system_base, bus,
            subagent=subagent,
            hidden_grant_callback=hidden_grant_callback,
        )
        if self._debug:
            append_debug(session, {"content": {"system": agent.system, "extra_params": effective_extra_params}})

        queue: asyncio.Queue[
            LogEvent | DiffEvent | InferEndEvent | ThinkingTokenEvent | BudgetExhaustedEvent
            | DelegationStartEvent | DelegationDoneEvent | None
        ] = asyncio.Queue()
        files_touched: list[str] = []
        main_run_id = uuid.uuid4().hex

        def _on_event(event: LlmEvent) -> None:
            _bridge_llm_event(event, queue, session, files_touched, self._debug, main_run_id)

        yield SubAgentStartEvent(
            name=subagent.name if subagent else "main",
            description=subagent.description if subagent else "thinking",
            color=_MAIN_COLOR,
        )

        prior = [] if subagent else _recency_turns(session.messages, _RECENCY_N)
        prior.append(Message(role="user", content=user_input))
        unsubscribe = bus.subscribe(_on_event)
        agent_task: asyncio.Task = asyncio.create_task(agent.run(prior, run_id=main_run_id))

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

            text = _last_assistant_text(history)
            if text:
                yield text
        finally:
            if not agent_task.done():
                agent_task.cancel()
            unsubscribe()

    async def _stream_plan(
        self,
        session: Session,
        user_input: str,
        seed: str | None,
        permission_callback: PermissionCallback | None,
        hidden_grant_callback: HiddenGrantCallback | None,
        bus: EventBus,
    ) -> AsyncIterator[AgentEvent | str]:
        """Case 3/4 mutation path: planner produces a validated Plan, the
        fixed interpreter walks it (plan 27 improvements 2-3). `main` steps
        run as a direct instruction (no spawn); subagent steps are a cold,
        fire-and-forget run whose start/outcome are traced on `bus` — the
        same bridge the single-agent path above uses, so DiffEvent/LogEvent/
        Delegation* rendering is shared, not reimplemented.
        """
        working_dir = session.working_dir
        permissions = session.permissions
        queue: asyncio.Queue = asyncio.Queue()
        files_touched: list[str] = []

        def _on_event(event: LlmEvent) -> None:
            _bridge_llm_event(event, queue, session, files_touched, self._debug, run_id=None)

        unsubscribe = bus.subscribe(_on_event)

        # SubAgentStartEvent is always the first event (AgentEvent contract) —
        # the TUI's per-step renderer setup relies on it, same as the
        # single-agent path; nested steps then render as delegation badges
        # underneath it via the same DelegationStart/DoneEvent bridge.
        yield SubAgentStartEvent(name="planner", description="planning", color=_MAIN_COLOR)

        try:
            plan = await self._planner.plan(user_input, seed=seed)
        except ValueError as exc:
            unsubscribe()
            yield PlanHaltedEvent(step_index=-1, agent="planner", reason=str(exc))
            yield DoneEvent(thinking_chars=0, files_touched=[])
            yield f"[planner failed to produce a valid plan: {exc}]"
            return

        yield PlanStartedEvent(
            step_count=len(plan),
            agents=[s.agent for s in plan],
            verify_placements=sum(1 for s in plan if s.verify),
        )

        async def dispatch(agent_name: str, task: str) -> str:
            if agent_name == "main":
                main_system = _enrich_system_base(SYSTEM_PROMPT, working_dir)
                agent_obj = _build_agent(
                    self._model, self._api_key, self._api_base, self._extra_params,
                    working_dir, permissions, permission_callback, main_system, bus,
                    subagent=None, hidden_grant_callback=hidden_grant_callback,
                )
                run_id = uuid.uuid4().hex
                bus.emit(DelegationStarted(agent="main", task=task, run_id=run_id))
                try:
                    history = await agent_obj.run(task, run_id=run_id)
                finally:
                    bus.emit(DelegationCompleted(agent="main", run_id=run_id))
                return _last_assistant_text(history)
            return await run_subagent(
                agent_name, task,
                model=self._model, api_key=self._api_key, api_base=self._api_base,
                extra_params=self._extra_params, working_dir=working_dir,
                permissions=permissions, permission_callback=permission_callback,
                bus=bus, hidden_grant_callback=hidden_grant_callback,
            )

        async def verify_agent(step: PlanStep, out: str) -> bool:
            # No dedicated verdict-emitting verify agent exists yet (open point 2,
            # hard problem 3) — "mechanical" or any unnamed check is a placeholder:
            # non-empty output passes. A named post-planning-only agent gets a
            # PASS/FAIL-prefixed review request; anything else is a fail-open pass.
            roster_names = {s.name for s in SUBAGENTS}
            if step.verify in roster_names:
                verdict = await dispatch(
                    step.verify,
                    f"verify this step's output against its task, then answer PASS or FAIL "
                    f"on the first line.\n\ntask: {step.task}\n\noutput:\n{out}",
                )
                return verdict.strip().upper().startswith("PASS")
            return bool(out.strip())

        plan_task: asyncio.Task[list[StepResult]] = asyncio.create_task(
            run_plan(plan, dispatch, verify_agent=verify_agent)
        )

        try:
            while not plan_task.done():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                yield item
            while not queue.empty():
                yield queue.get_nowait()

            try:
                await plan_task
            except PlanHalted as halted:
                yield PlanHaltedEvent(step_index=halted.index, agent=halted.step.agent, reason=halted.reason)
                yield DoneEvent(thinking_chars=0, files_touched=files_touched)
                yield _recap(plan, halted=halted)
                return

            yield DoneEvent(thinking_chars=0, files_touched=files_touched)
            yield _recap(plan, halted=None)
        finally:
            unsubscribe()


def _last_assistant_text(history: list[Message]) -> str:
    for msg in reversed(history):
        if msg.role == "assistant":
            if isinstance(msg.content, list):
                return "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
            return msg.content or ""
    return ""


def _recap(plan: Plan, *, halted: PlanHalted | None) -> str:
    """Mechanical (no narrator LLM call) recap of the plan's outcome — this
    is both the yielded assistant text (so it persists into session history
    for next-turn continuity, plan 27 hard problem 4) and the rendered
    checkpoint artifact (decision 15)."""
    lines = [f"plan ({len(plan)} step{'s' if len(plan) != 1 else ''}):"]
    for i, step in enumerate(plan):
        if halted is not None and i > halted.index:
            lines.append(f"  {i + 1}. {step.agent}: {step.task} — not run (plan halted earlier)")
        elif halted is not None and i == halted.index:
            lines.append(f"  {i + 1}. {step.agent}: {step.task} — HALTED ({halted.reason})")
        else:
            lines.append(f"  {i + 1}. {step.agent}: {step.task} — ok")
    if halted is not None:
        lines.append("prior steps' work is kept; nothing was rolled back.")
    return "\n".join(lines)


def _bridge_llm_event(
    event: LlmEvent,
    queue: asyncio.Queue,
    session: Session,
    files_touched: list[str],
    debug: bool,
    run_id: str | None,
) -> None:
    """Shared bus->queue bridge for both the single-agent path and the
    plan-interpreter path — tool/diff/thinking/delegation rendering is one
    implementation, not duplicated per path."""
    _SINGLE_PATH_TOOLS = ("write_file", "edit_file", "make_dir", "delete_file")
    _DUAL_PATH_TOOLS = ("move_file", "copy_file")

    if isinstance(event, ToolExecutionStarted):
        queue.put_nowait(LogEvent(message=_fmt_tool_call(event.call), tool_name=event.call.name))
        if debug:
            append_debug(session, {"content": {"tool_call": _fmt_debug_tool_input(event.call)}})
    elif isinstance(event, ToolExecutionCompleted):
        if not event.result.is_error:
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
            inp = event.call.input or {}
            if event.call.name in _SINGLE_PATH_TOOLS:
                path = inp.get("path", "")
                if path and path not in files_touched:
                    files_touched.append(path)
            elif event.call.name in _DUAL_PATH_TOOLS:
                for path in (inp.get("src", ""), inp.get("dst", "")):
                    if path and path not in files_touched:
                        files_touched.append(path)
        if debug:
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
    elif isinstance(event, DelegationStarted):
        queue.put_nowait(DelegationStartEvent(agent_name=event.agent, task=event.task))
    elif isinstance(event, DelegationCompleted):
        queue.put_nowait(DelegationDoneEvent(agent_name=event.agent))
    elif isinstance(event, AgentStopped) and run_id is not None:
        if event.run_id != run_id:
            return  # a nested run's own completion — not ours
        if event.budget_exhausted:
            queue.put_nowait(BudgetExhaustedEvent())
        queue.put_nowait(None)
