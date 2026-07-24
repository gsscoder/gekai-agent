from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..llm.events import DelegationCompleted, DelegationStarted, EventBus
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
) -> str:
    """Run `agent` (any roster name, invocable or post-planning-only) on `task`
    as a cold, fire-and-forget nested run — the interpreter's `dispatch` for a
    subagent step (plan 27 improvement 2). The nested run's start/outcome are
    traced on `bus` (main's own session), the run itself is isolated.
    """
    roster = {s.name: s for s in SUBAGENTS}
    resolved = roster.get(agent)
    if resolved is None:
        return f"[error] unknown agent '{agent}'; available: {', '.join(sorted(roster))}"

    # Deferred to avoid circular import (tools -> harness -> tools)
    from ..harness.core import _build_agent, _enrich_system_base

    system_base = _enrich_system_base(resolved.build_system_base(), working_dir)
    nested = _build_agent(
        model, api_key, api_base, extra_params, working_dir,
        permissions, permission_callback, system_base, bus,
        subagent=resolved,
        hidden_grant_callback=hidden_grant_callback,
        tools_override=tools_override,
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


__all__ = ["run_subagent"]
