"""Agent assembly and nested dispatch: everything the harness needs to turn a
resolved tier plus a unit's identity into a runnable `Agent`, and to run one
cold nested agent to completion.

`build_agent` is the single construction point — root, a seed-dispatched
subagent, a graph step, and a `delegate` call all go through it, so tool
selection, the permission overlay, and the iteration ceiling have exactly one
implementation. `run_subagent` and `make_delegate_tool` sit here rather than
in `agent/tools/` because they build nested agents, which is harness work;
`agent/tools/` stays purely model-facing filesystem and shell tools.
"""

from __future__ import annotations

import copy
import dataclasses
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm import Agent
from ..llm.events import DelegationCompleted, DelegationStarted, EventBus
from ..llm.providers.openai import OpenAIAdapter
from ..llm.tools import Tool, tool
from ..llm.types import TextBlock
from ..directive_pump import pump as pump_directives
from ..permissions import PermissionCallback, PermissionGate, Permissions
from ..persistence import append_debug
from ..persona import render_tool_instruction
from ..session import Session
from ..shell import resolve_shell
from ..subagents import SUBAGENTS, Subagent
from ..tiers.resolve import ResolvedTier
from ..tools import HiddenGrantCallback, make_tools

# Shared sentinel prefix for a crashed/unresolvable delegated run (both
# failure messages below carry it) — the interpreter's `run_task_graph`
# imports this to halt a step whose dispatch crashed, instead of letting the
# error string flow into verify/`{{step_k}}` substitution as if it were a
# legitimate output.
ERROR_PREFIX = "[error] "

# Tool-calling rounds allowed per turn. `Agent`'s own default (10) is a
# conservative library default; a turn here routinely spends rounds walking a
# real repo — a single "explain this codebase" turn was observed spending all
# ten on reads alone and never reaching an answer. Raised as harness policy so
# the vendored default stays untouched. This is a ceiling, not a target: turns
# that finish early still stop early, and the closing salvage call
# (`Agent._salvage_kwargs`) still catches whatever does reach the ceiling.
MAX_ITERATIONS = 25


# --- system-base assembly -------------------------------------------------
# Each of these appends one block to a unit's system prompt. All three are
# root-only by construction: every call site reaches them from a
# `subagent is None` branch. A subagent's context is isolated by design, and
# project instructions are precisely the channel that isolation exists to
# close.


def pumped_system_base(system_base: str) -> tuple[str, list[str]]:
    """Appends the escaping domain directives, letting root borrow domain
    expertise without borrowing a specialist's role."""
    text, domains = pump_directives()
    if text:
        system_base = f"{system_base}\n<domain_directives>\n{text}"
    return system_base, domains


def gekai_md_system_base(system_base: str, session: Session) -> str:
    """Appends GEKAI.md under its own tag, so the model can always tell the
    user's rules from Gekai's own `<directives>` block.

    The system base — not a simulated read-and-understand turn in message
    history — is where this lives because message history is what `/compact`
    evicts. A GEKAI.md seeded as a turn would silently stop applying
    somewhere around turn 40 with no signal to anyone; the system base is
    reassembled every turn and never falls out of the window.

    No truncation, no normalization, no reordering: if the file is oversized
    that is a telemetry fact for later, not a silent edit here.
    """
    if session.gekai_md is None:
        return system_base
    return f'{system_base}\n<project_instructions source="GEKAI.md">\n{session.gekai_md.text}'


def enrich_system_base(system_base: str, working_dir: Path) -> str:
    is_empty = not any(p for p in working_dir.iterdir() if p.name != ".gekai")
    return system_base + (
        f"\nworking root directory: {working_dir}"
        f"\nfile tool paths are relative to this root"
        f"\nthe directory name is only a label — do not infer requirements from it or use it to add unrequested features or complexity"
        + ("\nthis directory is empty — do not create a redundant wrapper subdirectory mirroring the project name; package layout (src/, tests/, etc.) is fine" if is_empty else "")
    )


# --- agent construction ---------------------------------------------------


def build_agent(
    resolved: ResolvedTier,
    working_dir: Path,
    permissions: Permissions,
    permission_callback: PermissionCallback | None,
    system_base: str,
    bus: EventBus | None = None,
    subagent: Subagent | None = None,
    hidden_grant_callback: HiddenGrantCallback | None = None,
    tools_override: frozenset[str] | None = None,
    can_delegate: bool = True,
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

    if tools_override is not None:
        selected = [t for t in selected if t.name in tools_override]

    # `delegate` is declared, not ambient: only a subagent (never root —
    # `subagent is None` excludes it by construction) that names targets in
    # its own `delegates_to` gets the tool, and only when this build is
    # itself allowed to delegate (`can_delegate`, the depth-1 cap). Appended
    # straight to `selected` after the `tools_override` filter, never through
    # `make_tools()` — `delegate` is not a filesystem/shell rung and must
    # stay out of `tools/catalog.py`. `parent_tools` captures this build's
    # own selected tool-name set (pre-delegate) so the child's effective
    # grant can only ever be tightened, never widened, past it.
    if subagent is not None and subagent.delegates_to and can_delegate:
        selected = selected + [make_delegate_tool(
            subagent.delegates_to,
            DispatchContext(
                model=resolved.model, api_key=resolved.api_key, api_base=resolved.api_base,
                extra_params=resolved.extra_params, working_dir=working_dir,
                permissions=permissions, permission_callback=permission_callback,
                bus=bus, hidden_grant_callback=hidden_grant_callback,
            ),
            frozenset(t.name for t in selected),
        )]

    system = f"{system_base}\n<tools>\n{render_tool_instruction([t.name for t in selected], shell_kind=resolve_shell().kind)}"

    max_iterations = (
        subagent.max_iterations
        if subagent is not None and subagent.max_iterations is not None
        else MAX_ITERATIONS
    )
    adapter = OpenAIAdapter(api_key=resolved.api_key, base_url=resolved.api_base)
    agent = Agent(
        provider=adapter,
        model=resolved.model,
        system=system,
        event_bus=bus,
        extra_params=resolved.extra_params,
        max_iterations=max_iterations,
    )
    for t in selected:
        agent.tools.register(t)
    agent.tools.set_gate(PermissionGate(
        permissions=effective,
        on_request=permission_callback,
    ))

    # Root never receives `delegate` — it stays a pure work-operator; the
    # harness (sequencer + interpreter) owns all cross-agent control flow for
    # root's own dispatches. Subagent-level delegation, wired above, is a
    # separate, bounded axis layered on top of that invariant, not a reversal
    # of it: declared per-unit (`delegates_to`), depth-capped at 1, and
    # tighten-only on tool capability.
    return agent


# --- nested dispatch ------------------------------------------------------


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
    subagent step, and the `delegate` tool's own primitive. The nested run's
    start/outcome are traced on `ctx.bus` (root's own session), the run itself
    is isolated.

    `parent_tools` (tighten-only capability intersection): when set, the
    caller's own effective tool-name set — intersected into `tools_override`
    before the child is built, so a delegating unit can never hand its child
    more capability than it holds itself, regardless of `agent`'s own declared
    `tools`/`tool_policy`. `can_delegate` defaults to `False` — a subagent
    reached via delegation never itself receives the `delegate` tool, so
    cycles are structurally impossible. The interpreter's `dispatch` is the
    one caller that passes `True`: a graph-spawned step is depth 0, not a
    delegation target, so the depth-1 cap applies one hop further, to whatever
    that step itself delegates to (`make_delegate_tool`'s own closure never
    forwards this flag, so that next hop keeps the `False` default).
    """
    roster = {s.name: s for s in SUBAGENTS}
    resolved = roster.get(agent)
    if resolved is None:
        return f"{ERROR_PREFIX}unknown agent '{agent}'; available: {', '.join(sorted(roster))}"

    effective_override = tools_override
    if parent_tools is not None:
        effective_override = parent_tools if effective_override is None else (effective_override & parent_tools)

    system_base = enrich_system_base(resolved.build_system_base(), ctx.working_dir)
    resolved_tier = ResolvedTier(
        model=ctx.model, api_key=ctx.api_key, api_base=ctx.api_base, extra_params=ctx.extra_params,
    )
    nested = build_agent(
        resolved_tier, ctx.working_dir,
        ctx.permissions, ctx.permission_callback, system_base, ctx.bus,
        subagent=resolved,
        hidden_grant_callback=ctx.hidden_grant_callback,
        tools_override=effective_override,
        can_delegate=can_delegate,
    )
    if ctx.verbose_telemetry and ctx.session is not None:
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
    expansion axis. Model-visible params are exactly `agent` and `task`;
    everything else travels through `Tool.hidden_params` + `.with_bound()`, so
    a wiring mistake fails loud at `ToolRegistry.register` rather than
    silently leaking a live credential into the tool schema. `agent`'s schema
    `enum` is narrowed to `targets` — this unit's own declared `delegates_to`
    — not the whole roster, so the model cannot name an undeclared subagent.
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
        return await run_subagent(agent, task, ctx=ctx, parent_tools=parent_tools)

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


__all__ = [
    "ERROR_PREFIX",
    "MAX_ITERATIONS",
    "DispatchContext",
    "build_agent",
    "enrich_system_base",
    "gekai_md_system_base",
    "make_delegate_tool",
    "pumped_system_base",
    "run_subagent",
]
