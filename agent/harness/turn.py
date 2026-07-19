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

from ..agent import GekaiAgent
from ..events import (
    AgentEvent,
    BudgetExhaustedEvent,
    DirectivePumpEvent,
    DoneEvent,
    EstimateEvent,
    InferEndEvent,
    LogEvent,
    MaxIterationsEvent,
    ScaleEvent,
    TaskGraphStartedEvent,
)
from ..permissions import PermissionCallback
from ..pipeline import Route
from ..session import Session
from ..tools import HiddenGrantCallback

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
    trivial: bool,
    permission_callback: PermissionCallback | None,
    hidden_grant_callback: HiddenGrantCallback | None,
    append_user: bool = True,
    on_event: OnEvent | None = None,
) -> TurnResult:
    """Drive `agent.process_stream` to completion, aggregating token/tool/
    outcome bookkeeping and emitting harness telemetry (`estimate`,
    `task_graph`, `harness`), while re-emitting every item to `on_event` in
    the same order it arrived — a caller renders from `on_event`, never from
    the aggregation here.
    """
    events = agent.events
    step_route = Route(trivial=trivial)
    result = TurnResult()
    answer_chunks: list[str] = []
    harness_start = time.monotonic()

    async for item in agent.process_stream(
        session, raw, step_route,
        permission_callback=permission_callback,
        hidden_grant_callback=hidden_grant_callback,
        turn_id=turn_id,
        append_user=append_user,
        seed=seed,
    ):
        if on_event is not None:
            await on_event(item)

        if isinstance(item, str):
            answer_chunks.append(item)
        elif isinstance(item, EstimateEvent):
            events.emit(
                "estimate", session=session_id, turn=turn_id,
                decision=item.decision, specialists=item.specialists,
                duration_ms=item.duration_ms,
            )
        elif isinstance(item, TaskGraphStartedEvent):
            events.emit(
                "task_graph", session=session_id, turn=turn_id,
                step_count=item.step_count, agents=item.agents,
                verify_placements=item.verify_placements,
            )
        elif isinstance(item, ScaleEvent):
            # Telemetry only (plan 28 Phase 2) — events-*.jsonl via
            # `EventLogger.emit`, same sink as `estimate`/`task_graph` above.
            # `_on_event` above already re-emits it to the TUI's `on_event`
            # like every other item, but the TUI's own handler doesn't
            # recognize `ScaleEvent` (same as `EstimateEvent`/
            # `TaskGraphStartedEvent`), so it renders nothing — not
            # TUI-visible, per the event's docstring.
            events.emit(
                "scale", session=session_id, turn=turn_id,
                component=item.component, default_tier=item.default_tier,
                chosen_tier=item.chosen_tier, reason=item.reason,
            )
        elif isinstance(item, DirectivePumpEvent):
            # Telemetry only (plan 28 Phase 3, hard problem 3) — same
            # not-TUI-visible treatment as ScaleEvent above.
            events.emit(
                "directive_pump", session=session_id, turn=turn_id,
                domains=item.domains,
            )
        elif isinstance(item, LogEvent):
            if item.tool_name:
                result.query_tool_count += 1
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
    events.emit(
        "harness", session=session_id, turn=turn_id, outcome=result.outcome,
        llm_calls=result.llm_calls, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, thinking_chars=result.thinking_chars,
        tools=result.tool_counts, duration_ms=round((time.monotonic() - harness_start) * 1000),
        budget_exhausted=result.budget_exhausted,
    )
    return result


__all__ = ["TurnResult", "run_step", "OnEvent"]
