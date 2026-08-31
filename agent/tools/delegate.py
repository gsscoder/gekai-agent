from __future__ import annotations

import copy
import dataclasses
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm.events import DelegationCompleted, DelegationStarted, EventBus
from ..llm.tools import Tool, tool
from ..llm.types import TextBlock
from ..permissions import PermissionCallback
from ..session import Session
from ..settings import Permissions
from ..subagents import SUBAGENTS

# Shared sentinel prefix for a crashed/unresolvable delegated run (both
# failure messages below carry it) — the interpreter's `run_task_graph`
# imports this to halt a step whose dispatch crashed, instead of letting the
# error string flow into verify/`{{step_k}}` substitution as if it were a
# legitimate output.
ERROR_PREFIX = "[error] "


@dataclass(slots=True)
class DispatchContext:
    """The model/credential/session-plumbing bundle every dispatch into a
    nested agent needs — repeats verbatim across `run_subagent`,
    `make_delegate_tool`, and its `delegate` closure; carried as one unit
    instead of 9 loose parameters."""
    model: str
    api_key: str | None
    api_base: str | None
    extra_params: dict[str, Any]
    working_dir: Path
    permissions: Permissions
    permission_callback: PermissionCallback | None
    bus: EventBus | None
    hidden_grant_callback: Any | None
    session: Session | None = None
    verbose_telemetry: bool = True


async def run_subagent(
    agent: str,
    task: str,
    *,
    mission: str = "",
    ctx: DispatchContext,
    tools_override: frozenset[str] | None = None,
    parent_tools: frozenset[str] | None = None,
    can_delegate: bool = False,
) -> str:
    """Run `agent` (any roster name, invocable or post-planning-only) on `task`
    as a cold, fire-and-forget nested run — the interpreter's `dispatch` for a
    subagent step (plan 27 improvement 2), and the `delegate` tool's own
    primitive (plan-delegate-reintroduction Phase 2). The nested run's
    start/outcome are traced on `ctx.bus` (root's own session), the run
    itself is isolated.

    `parent_tools` (tighten-only capability intersection): when set, the
    caller's own effective tool-name set — intersected into `tools_override`
    before the child is built, so a delegating unit can never hand its child
    more capability than it holds itself, regardless of `agent`'s own
    declared `tools`/`tool_policy`. `can_delegate` defaults to `False` — a
    subagent reached via delegation never itself receives the `delegate`
    tool, so cycles are structurally impossible. The interpreter's `dispatch`
    (harness/core.py) is the one caller that passes `True`: a graph-spawned
    step is depth 0, not a delegation target, so the depth-1 cap does not
    apply to it — it applies one hop further, to whatever that step itself
    delegates to (`make_delegate_tool`'s own closure never forwards this
    flag, so that next hop is still built with the `False` default).
    """
    roster = {s.name: s for s in SUBAGENTS}
    resolved = roster.get(agent)
    if resolved is None:
        return f"{ERROR_PREFIX}unknown agent '{agent}'; available: {', '.join(sorted(roster))}"

    effective_override = tools_override
    if parent_tools is not None:
        effective_override = parent_tools if effective_override is None else (effective_override & parent_tools)

    # Deferred to avoid circular import (tools -> harness -> tools)
    from ..harness.core import _build_agent, _enrich_system_base
    from ..llm.resolve import ResolvedTier

    system_base = _enrich_system_base(resolved.build_system_base(), ctx.working_dir)
    resolved_tier = ResolvedTier(
        model=ctx.model, api_key=ctx.api_key, api_base=ctx.api_base, extra_params=ctx.extra_params,
    )
    nested = _build_agent(
        resolved_tier, ctx.working_dir,
        ctx.permissions, ctx.permission_callback, system_base, ctx.bus,
        subagent=resolved,
        hidden_grant_callback=ctx.hidden_grant_callback,
        tools_override=effective_override,
        can_delegate=can_delegate,
    )
    if ctx.verbose_telemetry and ctx.session is not None:
        from ..persistence import append_debug
        append_debug(ctx.session, {"content": {"system": nested.system, "agent": agent}})
    nested_run_id = uuid.uuid4().hex
    if ctx.bus is not None:
        ctx.bus.emit(DelegationStarted(agent=agent, task=task, mission=mission, run_id=nested_run_id))
    try:
        history = await nested.run(task, run_id=nested_run_id)
    except Exception as exc:
        return f"{ERROR_PREFIX}{agent} failed: {exc}"
    finally:
        if ctx.bus is not None:
            ctx.bus.emit(DelegationCompleted(agent=agent, run_id=nested_run_id))

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
    ctx: DispatchContext,
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
        ctx: DispatchContext,
        parent_tools: frozenset[str] | None,
    ) -> str:
        return await run_subagent(
            agent, task,
            ctx=ctx,
            parent_tools=parent_tools,
        )

    t = tool(
        delegate,
        name="delegate",
        description=(
            f"Delegate a self-contained task to a specialist subagent. "
            f"Available agents — {roster_str}. "
            "Prefer handling the task yourself for work inside your own mandate; "
            "a read-only lookup specialist in the list above is the exception — hand it "
            "a broad or open-ended search instead of running many grep/read calls yourself."
        ),
        required_permission="none",
        is_concurrency_safe=False,
        hidden_params={"ctx", "parent_tools"},
    )
    t = t.with_bound(ctx=ctx, parent_tools=parent_tools)
    schema = copy.deepcopy(t.input_schema)
    schema["properties"]["agent"]["enum"] = sorted(targets)
    return dataclasses.replace(t, input_schema=schema)


__all__ = ["DispatchContext", "ERROR_PREFIX", "run_subagent", "make_delegate_tool"]
