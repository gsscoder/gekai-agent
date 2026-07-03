from __future__ import annotations

import copy
import uuid
from dataclasses import replace as _replace
from pathlib import Path
from typing import Any

from ..llm.events import DelegationCompleted, DelegationStarted, EventBus
from ..llm.tools import Tool, tool
from ..llm.types import TextBlock
from ..permissions import PermissionCallback
from ..settings import Permissions
from ..subagents import SUBAGENTS


def make_delegate_tool(
    model: str,
    api_key: str | None,
    api_base: str | None,
    extra_params: dict[str, Any],
    working_dir: Path,
    permissions: Permissions,
    permission_callback: PermissionCallback | None,
    bus: EventBus | None,
    hidden_grant_callback: Any | None,
) -> Tool:
    roster = {s.name: s for s in SUBAGENTS if s.user_invocable}
    roster_str = "; ".join(f"{name}: {s.description}" for name, s in sorted(roster.items()))

    async def delegate(agent: str, task: str) -> str:
        resolved = roster.get(agent)
        if resolved is None:
            return f"[error] unknown agent '{agent}'; available: {roster_str}"

        # Deferred to avoid circular import (tools → harness → tools)
        from ..harness.core import _build_agent, _enrich_system_base

        system_base = _enrich_system_base(resolved.build_system_base(), working_dir)
        nested = _build_agent(
            model, api_key, api_base, extra_params, working_dir,
            permissions, permission_callback, system_base, bus,
            subagent=resolved,
            hidden_grant_callback=hidden_grant_callback,
        )
        nested_run_id = uuid.uuid4().hex
        if bus is not None:
            bus.emit(DelegationStarted(agent=agent, task=task, run_id=nested_run_id))
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

    t = tool(
        delegate,
        name="delegate",
        description=(
            f"Delegate a self-contained task to a specialist subagent. "
            f"Available agents — {roster_str}. "
            "Prefer a specialist whenever the task matches one; handle work yourself only when no specialist covers it. "
            "Call once per specialist unit; never split one artifact across multiple calls; "
            "order by dependency (scaffold → logic → tests)."
        ),
        required_permission="none",
        is_concurrency_safe=False,
    )
    schema = copy.deepcopy(t.input_schema)
    schema["properties"]["agent"]["enum"] = sorted(roster.keys())
    return _replace(t, input_schema=schema)
