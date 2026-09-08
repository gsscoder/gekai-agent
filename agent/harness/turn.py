"""Turn-step bookkeeping (plan 28 Phase 0): the harness's own account of one
`GekaiAgent.process_stream` run, extracted out of `tui/app.py`.

Before this extraction, `GekaiApp._run_step` computed token/tool/outcome
bookkeeping AND mounted TUI widgets in the same inline loop — harness
business logic and rendering were the same code. `run_step` here owns only
the bookkeeping (harness territory); it re-emits every item to an `on_event`
callback so a caller (the TUI, or anything else) can render without also
having to compute turn state. Behavior-preserving: same events consumed in
the same order, same aggregation, same outcome rule.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..agent import DirectiveVerdictCallback, GekaiAgent
from ..events import (
    AgentEvent,
    BudgetExhaustedEvent,
    DoneEvent,
    ForeignFileDetectedEvent,
    InferEndEvent,
    LogEvent,
    MaxIterationsEvent,
)
from ..permissions import PermissionCallback
from ..session import Session
from ..tools import ExternalGrantCallback, HiddenGrantCallback

OnEvent = Callable[[AgentEvent | str], Awaitable[None]]


@dataclass
class TurnResult:
    """Aggregate outcome of one turn — the non-rendering half of what
    `_run_step` used to compute inline while also mounting widgets."""
    outcome: str = "ok"  # "ok" | "max_iterations"
    answer: str = ""
    max_iter_hit: bool = False
    budget_exhausted: bool = False
    query_tool_count: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_chars: int = 0
    files_touched: list[str] = field(default_factory=list)


async def run_step(
    agent: GekaiAgent,
    session: Session,
    raw: str,
    seed: str | None,
    *,
    turn_id: str,
    session_id: str,
    permission_callback: PermissionCallback | None,
    hidden_grant_callback: HiddenGrantCallback | None,
    external_grant_callback: ExternalGrantCallback | None,
    append_user: bool = True,
    on_event: OnEvent | None = None,
    on_directive_verdict: DirectiveVerdictCallback | None = None,
) -> TurnResult:
    """Drive `agent.process_stream` to completion, aggregating token/tool/
    outcome bookkeeping and emitting harness telemetry (`estimate`,
    `task_graph`, `harness`), while re-emitting every item to `on_event` in
    the same order it arrived — a caller renders from `on_event`, never from
    the aggregation here.

    `on_directive_verdict` (plan 35 Phase 3) is the TUI's `#directive-notice`
    callback (`GekaiApp._apply_directive_verdict`), threaded through so a
    `ForeignFileDetectedEvent` below can fire `agent.start_foreign_file_audit`
    with the same callback GEKAI.md's own session-start audit uses — one
    slot, one line, last verdict wins (concept 7), whichever file's audit
    lands last.
    """
    events = agent.events
    result = TurnResult()
    answer_chunks: list[str] = []
    harness_start = time.monotonic()

    async for item in agent.process_stream(
        session, raw,
        permission_callback=permission_callback,
        hidden_grant_callback=hidden_grant_callback,
        external_grant_callback=external_grant_callback,
        turn_id=turn_id,
        append_user=append_user,
        seed=seed,
    ):
        if on_event is not None:
            await on_event(item)

        if isinstance(item, str):
            answer_chunks.append(item)
            continue

        # Telemetry-declaring events go straight to events-*.jsonl (see
        # `AgentEvent.telemetry`). None of them render: `on_event` above
        # already offered every item to the caller, and the TUI's handler
        # recognizes none of these, so they are not TUI-visible.
        if item.telemetry is not None:
            events.emit(
                item.telemetry, session=session_id, turn=turn_id,
                **item.telemetry_payload(),
            )

        if isinstance(item, ForeignFileDetectedEvent):
            # Not telemetry-only: this is the trigger itself —
            # `start_foreign_file_audit` does its own cache-check/call/
            # swallow/telemetry, mirroring `start_directive_audit`.
            agent.start_foreign_file_audit(item.rel_path, item.text, on_directive_verdict)
        elif isinstance(item, LogEvent):
            if item.tool_name:
                result.tool_counts[item.tool_name] = result.tool_counts.get(item.tool_name, 0) + 1
        elif isinstance(item, InferEndEvent):
            result.llm_calls += 1
            result.prompt_tokens += item.prompt_tokens or 0
            result.completion_tokens += item.completion_tokens or 0
        elif isinstance(item, DoneEvent):
            result.thinking_chars = item.thinking_chars
            result.files_touched = item.files_touched
        elif isinstance(item, MaxIterationsEvent):
            result.max_iter_hit = True
        elif isinstance(item, BudgetExhaustedEvent):
            result.budget_exhausted = True

    result.answer = "".join(answer_chunks).rstrip()
    result.outcome = (
        "max_iterations"
        if (result.max_iter_hit and not answer_chunks) or (not answer_chunks and not result.files_touched)
        else "ok"
    )
    result.query_tool_count = sum(result.tool_counts.values())
    events.emit(
        "harness", session=session_id, turn=turn_id, outcome=result.outcome,
        llm_calls=result.llm_calls, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, thinking_chars=result.thinking_chars,
        tools=result.tool_counts, duration_ms=round((time.monotonic() - harness_start) * 1000),
        budget_exhausted=result.budget_exhausted,
    )
    return result


__all__ = ["TurnResult", "run_step", "OnEvent"]
