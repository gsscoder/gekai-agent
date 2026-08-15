from __future__ import annotations

import copy
import dataclasses
import uuid
from pathlib import Path
from typing import Any

from ..llm.events import DelegationCompleted, DelegationStarted, EventBus
from ..llm.tools import Tool, tool
from ..llm.types import TextBlock
from ..permissions import PermissionCallback
from ..settings import Permissions
from ..subagents import SUBAGENTS


async def run_subagent(
    agent: str,
    task: str,
    *,
    mission: str = "",
    model: str,
    api_key: str | None,
    api_base: str | None,
    extra_params: dict[str, Any],
    working_dir: Path,
    permissions: Permissions,
    permission_callback: PermissionCallback | None,
    bus: EventBus | None,
    hidden_grant_callback: Any | None,
    tools_override: frozenset[str] | None = None,
    parent_tools: frozenset[str] | None = None,
) -> str:
    """Run `agent` (any roster name, invocable or post-planning-only) on `task`
    as a cold, fire-and-forget nested run — the interpreter's `dispatch` for a
    subagent step (plan 27 improvement 2), and the `delegate` tool's own
    primitive (plan-delegate-reintroduction Phase 2). The nested run's
    start/outcome are traced on `bus` (root's own session), the run itself is
    isolated.

    `parent_tools` (tighten-only capability intersection): when set, the
    caller's own effective tool-name set — intersected into `tools_override`
    before the child is built, so a delegating unit can never hand its child
    more capability than it holds itself, regardless of `agent`'s own
    declared `tools`/`tool_policy`. The nested agent is always built with
    `can_delegate=False`: a subagent reached via delegation never itself
    receives the `delegate` tool, so cycles are structurally impossible.
    """
    roster = {s.name: s for s in SUBAGENTS}
    resolved = roster.get(agent)
    if resolved is None:
        return f"[error] unknown agent '{agent}'; available: {', '.join(sorted(roster))}"

    effective_override = tools_override
    if parent_tools is not None:
        effective_override = parent_tools if effective_override is None else (effective_override & parent_tools)

    # Deferred to avoid circular import (tools -> harness -> tools)
    from ..harness.core import _build_agent, _enrich_system_base

    system_base = _enrich_system_base(resolved.build_system_base(), working_dir)
    nested = _build_agent(
        model, api_key, api_base, extra_params, working_dir,
        permissions, permission_callback, system_base, bus,
        subagent=resolved,
        hidden_grant_callback=hidden_grant_callback,
        tools_override=effective_override,
        can_delegate=False,
    )
    nested_run_id = uuid.uuid4().hex
    if bus is not None:
        bus.emit(DelegationStarted(agent=agent, task=task, mission=mission, run_id=nested_run_id))
    try:
        history = await nested.run(task, run_id=nested_run_id)
    except Exception as exc:
        return f"[error] {agent} failed: {exc}"
    finally:
        if bus is not None:
            bus.emit(DelegationCompleted(agent=agent, run_id=nested_run_id))

    for msg in reversed(history):
        if msg.role == "assistant":
            if isinstance(msg.content, list):
                text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
            else:
                text = msg.content or ""
            if text:
                return text
    return f"[{agent} completed with no text output]"


def make_delegate_tool(
    targets: tuple[str, ...],
    model: str,
    api_key: str | None,
    api_base: str | None,
    extra_params: dict[str, Any],
    working_dir: Path,
    permissions: Permissions,
    permission_callback: PermissionCallback | None,
    bus: EventBus | None,
    hidden_grant_callback: Any | None,
    parent_tools: frozenset[str] | None,
) -> Tool:
    """Thin wrapper over `run_subagent` — a subagent's bounded, declared
    expansion axis (plan-delegate-reintroduction Phase 2/3). Model-visible
    params are exactly `agent` and `task`; everything else travels through
    `Tool.hidden_params` + `.with_bound()`, so a wiring mistake fails loud at
    `ToolRegistry.register` rather than silently leaking a live credential
    into the tool schema. `agent`'s schema `enum` is narrowed to `targets` —
    this unit's own declared `delegates_to` — not the whole roster, so the
    model cannot name an undeclared subagent.
    """
    roster = {s.name: s for s in SUBAGENTS}
    roster_str = "; ".join(f"{name}: {roster[name].description}" for name in targets if name in roster)

    async def delegate(
        agent: str,
        task: str,
        *,
        model: str,
        api_key: str | None,
        api_base: str | None,
        extra_params: dict[str, Any],
        working_dir: Path,
        permissions: Permissions,
        permission_callback: PermissionCallback | None,
        bus: EventBus | None,
        hidden_grant_callback: Any | None,
        parent_tools: frozenset[str] | None,
    ) -> str:
        return await run_subagent(
            agent, task,
            model=model, api_key=api_key, api_base=api_base,
            extra_params=extra_params, working_dir=working_dir,
            permissions=permissions, permission_callback=permission_callback,
            bus=bus, hidden_grant_callback=hidden_grant_callback,
            parent_tools=parent_tools,
        )

    t = tool(
        delegate,
        name="delegate",
        description=(
            f"Delegate a self-contained task to a specialist subagent. "
            f"Available agents — {roster_str}. "
            "Prefer handling the task yourself; delegate only work genuinely "
            "outside your own mandate that a listed specialist covers."
        ),
        required_permission="none",
        is_concurrency_safe=False,
        hidden_params={
            "model", "api_key", "api_base", "extra_params", "working_dir",
            "permissions", "permission_callback", "bus", "hidden_grant_callback",
            "parent_tools",
        },
    )
    t = t.with_bound(
        model=model, api_key=api_key, api_base=api_base, extra_params=extra_params,
        working_dir=working_dir, permissions=permissions, permission_callback=permission_callback,
        bus=bus, hidden_grant_callback=hidden_grant_callback, parent_tools=parent_tools,
    )
    schema = copy.deepcopy(t.input_schema)
    schema["properties"]["agent"]["enum"] = sorted(targets)
    return dataclasses.replace(t, input_schema=schema)


__all__ = ["run_subagent", "make_delegate_tool"]
