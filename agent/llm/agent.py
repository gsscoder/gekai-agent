"""Agent run loop."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import CostCeilingExceeded, MaxIterationsExceeded
from .events import (
    AgentStarted,
    AgentStopped,
    Event,
    EventBus,
    ModelRequestSent,
    ModelResponseReceived,
    RetryAttemptEvent,
    StopReason,
    TextChunkReceived,
    ThinkingChunkReceived,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TurnStarted,
    UsageUpdated,
)
from .providers.base import ProviderAdapter
from .result import AgentResult, AgentResultEvent
from .retry import RetryAttempt, RetryPolicy, retry_call
from .tools import ToolRegistry
from .types import (
    CompletionResponse,
    Cost,
    Message,
    Pricing,
    StreamDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    TokenCount,
    ToolResultBlock,
    ToolUseBlock,
    UsageTally,
)


@dataclass
class Agent:
    provider: ProviderAdapter
    model: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    system: str | None = None
    max_iterations: int = 10
    tool_timeout: float | None = 30.0
    max_tokens: int = 4096
    retry_policy: RetryPolicy | None = None
    pricing: Pricing = field(
        default_factory=lambda: Pricing(input_per_mtok=1.00, output_per_mtok=2.00)
    )
    usage: UsageTally = field(default_factory=UsageTally)
    event_bus: EventBus | None = None
    cost_ceiling: float | None = None
    wall_clock_timeout: float | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    _run_id: str = field(default="", init=False, repr=False, compare=False)

    async def run(self, prompt: str | list[Message], *, run_id: str | None = None) -> list[Message]:
        self._run_id = run_id or uuid.uuid4().hex
        messages = self._normalize_prompt(prompt)
        self._emit(AgentStarted(prompt=prompt, model=self.model))
        try:
            await self._run_under_timeout(messages)
        except asyncio.TimeoutError:
            self._emit(AgentStopped(stop_reason="timeout", turns=self.usage.turns))
            raise
        except asyncio.CancelledError:
            self._emit(AgentStopped(stop_reason="cancelled", turns=self.usage.turns))
            raise
        return messages

    async def _run_under_timeout(self, messages: list[Message]) -> None:
        if self.wall_clock_timeout is None:
            await self._run_loop(messages)
        else:
            await asyncio.wait_for(self._run_loop(messages), self.wall_clock_timeout)

    async def _run_loop(self, messages: list[Message]) -> None:
        async def _complete(*, with_tools: bool = True) -> CompletionResponse:
            self.usage.record_call()
            final: CompletionResponse | None = None
            async for ev in self.provider.stream(**self._provider_kwargs(messages, with_tools=with_tools)):
                if isinstance(ev, ThinkingDelta):
                    self._emit(ThinkingChunkReceived(text=ev.text))
                elif isinstance(ev, TextDelta):
                    self._emit(TextChunkReceived(text=ev.text))
                elif isinstance(ev, StreamDone):
                    final = ev.response
            if final is None:
                raise RuntimeError(
                    f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                )
            return final

        try:
            for turn in range(1, self.max_iterations + 1):
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                response = await retry_call(self._instrumented_policy(turn), _complete)
                self._emit(ModelResponseReceived(turn=turn, response=response))
                if not await self._apply_response(response, messages, turn=turn):
                    if not self._assistant_text(messages):
                        nudge = await retry_call(
                            self._instrumented_policy(turn), lambda: _complete(with_tools=False)
                        )
                        await self._apply_response(nudge, messages, turn=turn)
                    self._emit(AgentStopped(stop_reason="complete", turns=self.usage.turns))
                    return

            budget_exhausted = False
            if not self._assistant_text(messages):
                budget_exhausted = True
                turn = self.max_iterations + 1
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                response = await retry_call(
                    self._instrumented_policy(turn), lambda: _complete(with_tools=False)
                )
                self._emit(ModelResponseReceived(turn=turn, response=response))
                await self._apply_response(response, messages, turn=turn)

            if self._assistant_text(messages):
                self._emit(
                    AgentStopped(
                        stop_reason="complete",
                        turns=self.usage.turns,
                        budget_exhausted=budget_exhausted,
                    )
                )
                return
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise
        except BaseException as exc:
            self._emit_stopped_for(exc)
            raise

        self._emit(
            AgentStopped(stop_reason="max_iterations", turns=self.usage.turns, budget_exhausted=True)
        )
        raise MaxIterationsExceeded(
            f"Agent exceeded max_iterations={self.max_iterations} without terminating"
        )

    async def run_stream(
        self, prompt: str | list[Message], *, run_id: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        self._run_id = run_id or uuid.uuid4().hex
        messages = self._normalize_prompt(prompt)
        self._emit(AgentStarted(prompt=prompt, model=self.model))

        try:
            for turn in range(1, self.max_iterations + 1):
                final_response: CompletionResponse | None = None
                self.usage.record_call()
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                async for event in self.provider.stream(**self._provider_kwargs(messages)):
                    if isinstance(event, StreamDone):
                        final_response = event.response
                    yield event

                if final_response is None:
                    raise RuntimeError(
                        f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                    )
                self._emit(ModelResponseReceived(turn=turn, response=final_response))

                if not await self._apply_response(final_response, messages, turn=turn):
                    if not self._assistant_text(messages):
                        final_response = None
                        self.usage.record_call()
                        async for event in self.provider.stream(
                            **self._provider_kwargs(messages, with_tools=False)
                        ):
                            if isinstance(event, StreamDone):
                                final_response = event.response
                            yield event
                        if final_response is None:
                            raise RuntimeError(
                                f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                            )
                        self._emit(ModelResponseReceived(turn=turn, response=final_response))
                        await self._apply_response(final_response, messages, turn=turn)
                    self._emit(AgentStopped(stop_reason="complete", turns=self.usage.turns))
                    return

            budget_exhausted = False
            if not self._assistant_text(messages):
                budget_exhausted = True
                turn = self.max_iterations + 1
                final_response = None
                self.usage.record_call()
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                async for event in self.provider.stream(
                    **self._provider_kwargs(messages, with_tools=False)
                ):
                    if isinstance(event, StreamDone):
                        final_response = event.response
                    yield event

                if final_response is None:
                    raise RuntimeError(
                        f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                    )
                self._emit(ModelResponseReceived(turn=turn, response=final_response))
                await self._apply_response(final_response, messages, turn=turn)

            if self._assistant_text(messages):
                self._emit(
                    AgentStopped(
                        stop_reason="complete",
                        turns=self.usage.turns,
                        budget_exhausted=budget_exhausted,
                    )
                )
                return
        except asyncio.CancelledError:
            self._emit(AgentStopped(stop_reason="cancelled", turns=self.usage.turns))
            raise
        except BaseException as exc:
            self._emit_stopped_for(exc)
            raise

        self._emit(
            AgentStopped(stop_reason="max_iterations", turns=self.usage.turns, budget_exhausted=True)
        )
        raise MaxIterationsExceeded(
            f"Agent exceeded max_iterations={self.max_iterations} without terminating"
        )

    def run_sync(self, prompt: str | list[Message]) -> list[Message]:
        return asyncio.run(self.run(prompt))

    async def run_with_result(
        self, prompt: str | list[Message], *, run_id: str | None = None
    ) -> AgentResult:
        self._run_id = run_id or uuid.uuid4().hex
        messages = self._normalize_prompt(prompt)
        self._emit(AgentStarted(prompt=prompt, model=self.model))
        stop_reason: StopReason = "complete"
        error: Exception | None = None
        try:
            await self._run_under_timeout(messages)
        except MaxIterationsExceeded:
            stop_reason = "max_iterations"
        except CostCeilingExceeded as exc:
            stop_reason = "cost_ceiling"
            error = exc
        except asyncio.TimeoutError:
            stop_reason = "timeout"
            self._emit(AgentStopped(stop_reason="timeout", turns=self.usage.turns))
        except asyncio.CancelledError:
            self._emit(AgentStopped(stop_reason="cancelled", turns=self.usage.turns))
            raise
        except Exception as exc:  # noqa: BLE001
            stop_reason = "error"
            error = exc
        return self._build_result(messages, stop_reason=stop_reason, error=error)

    async def run_stream_with_result(
        self, prompt: str | list[Message], *, run_id: str | None = None
    ) -> AsyncIterator[StreamEvent | AgentResultEvent]:
        self._run_id = run_id or uuid.uuid4().hex
        messages: list[Message] = self._normalize_prompt(prompt)
        stop_reason: StopReason = "complete"
        error: Exception | None = None
        budget_exhausted = False
        self._emit(AgentStarted(prompt=prompt, model=self.model))

        try:
            completed = False
            for turn in range(1, self.max_iterations + 1):
                final_response: CompletionResponse | None = None
                self.usage.record_call()
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                async for event in self.provider.stream(**self._provider_kwargs(messages)):
                    if isinstance(event, StreamDone):
                        final_response = event.response
                    yield event

                if final_response is None:
                    raise RuntimeError(
                        f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                    )
                self._emit(ModelResponseReceived(turn=turn, response=final_response))

                if not await self._apply_response(final_response, messages, turn=turn):
                    completed = True
                    break

            if not completed and not self._assistant_text(messages):
                budget_exhausted = True
                turn = self.max_iterations + 1
                final_response = None
                self.usage.record_call()
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                async for event in self.provider.stream(
                    **self._provider_kwargs(messages, with_tools=False)
                ):
                    if isinstance(event, StreamDone):
                        final_response = event.response
                    yield event

                if final_response is None:
                    raise RuntimeError(
                        f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                    )
                self._emit(ModelResponseReceived(turn=turn, response=final_response))
                await self._apply_response(final_response, messages, turn=turn)

            if not completed and not self._assistant_text(messages):
                stop_reason = "max_iterations"
        except MaxIterationsExceeded:
            stop_reason = "max_iterations"
            budget_exhausted = True
        except CostCeilingExceeded as exc:
            stop_reason = "cost_ceiling"
            error = exc
        except asyncio.CancelledError:
            self._emit(AgentStopped(stop_reason="cancelled", turns=self.usage.turns))
            raise
        except Exception as exc:  # noqa: BLE001
            stop_reason = "error"
            error = exc

        self._emit(
            AgentStopped(
                stop_reason=stop_reason,
                turns=self.usage.turns,
                budget_exhausted=budget_exhausted,
                error=error,
            )
        )
        yield AgentResultEvent(
            result=self._build_result(messages, stop_reason=stop_reason, error=error)
        )

    def with_provider(
        self,
        provider: ProviderAdapter,
        *,
        tools: ToolRegistry | None = None,
        event_bus: EventBus | None = None,
    ) -> Agent:
        clone = replace(self, provider=provider, usage=UsageTally())
        if tools is not None:
            clone.tools = tools
        if event_bus is not None:
            clone.event_bus = event_bus
        return clone

    def fresh(self) -> Agent:
        return replace(self, usage=UsageTally())

    async def count_tokens(self, prompt: str | list[Message]) -> TokenCount:
        messages = self._normalize_prompt(prompt)
        return await self.provider.count_tokens(
            model=self.model,
            messages=messages,
            system=self.system,
            tools=self.tools.definitions() or None,
        )

    def cost(self) -> Cost:
        return self.usage.cost(self.pricing)

    # --- private helpers ---

    @staticmethod
    def _normalize_prompt(prompt: str | list[Message]) -> list[Message]:
        if isinstance(prompt, str):
            return [Message(role="user", content=prompt)]
        return list(prompt)

    def _provider_kwargs(self, messages: list[Message], *, with_tools: bool = True) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "system": self.system,
            "tools": (self.tools.definitions() or None) if with_tools else None,
            "max_tokens": self.max_tokens,
            **self.extra_params,
        }

    async def _apply_response(
        self,
        response: CompletionResponse,
        messages: list[Message],
        *,
        turn: int,
    ) -> bool:
        self.usage.add(response.usage)
        self._emit(UsageUpdated(turn=turn, usage=self.usage.snapshot(), delta=response.usage))
        self._check_cost_ceiling()
        messages.append(Message(role="assistant", content=list(response.content)))
        tool_uses = response.tool_uses()
        if not tool_uses:
            return False
        results = await self._run_tools(tool_uses, turn=turn)
        messages.append(Message(role="user", content=list(results)))
        return True

    async def _run_tools(
        self, tool_uses: list[ToolUseBlock], *, turn: int
    ) -> list[ToolResultBlock]:
        if self.event_bus is None:
            return await self.tools.run(tool_uses, timeout=self.tool_timeout)

        def _on_start(call: ToolUseBlock) -> None:
            self._emit(ToolExecutionStarted(turn=turn, call=call))

        def _on_complete(call: ToolUseBlock, result: ToolResultBlock, duration_s: float) -> None:
            self._emit(
                ToolExecutionCompleted(turn=turn, call=call, result=result, duration_s=duration_s)
            )

        return await self.tools.run(
            tool_uses,
            timeout=self.tool_timeout,
            on_start=_on_start,
            on_complete=_on_complete,
        )

    def _check_cost_ceiling(self) -> None:
        if self.cost_ceiling is None:
            return
        spent = self.cost().total
        if spent > self.cost_ceiling:
            raise CostCeilingExceeded(spent=spent, ceiling=self.cost_ceiling)

    def _emit(self, event: Event) -> None:
        if self.event_bus is not None:
            self.event_bus.emit(replace(event, run_id=self._run_id))

    def _emit_stopped_for(self, exc: BaseException) -> None:
        if isinstance(exc, MaxIterationsExceeded):
            self._emit(
                AgentStopped(
                    stop_reason="max_iterations", turns=self.usage.turns, budget_exhausted=True
                )
            )
        elif isinstance(exc, CostCeilingExceeded):
            self._emit(AgentStopped(stop_reason="cost_ceiling", turns=self.usage.turns, error=exc))
        elif isinstance(exc, Exception):
            self._emit(AgentStopped(stop_reason="error", turns=self.usage.turns, error=exc))

    @staticmethod
    def _assistant_text(messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "assistant":
                if isinstance(msg.content, str):
                    return msg.content
                return "".join(b.text for b in msg.content if isinstance(b, TextBlock))
        return ""

    def _build_result(
        self,
        messages: list[Message],
        *,
        stop_reason: StopReason,
        error: Exception | None,
    ) -> AgentResult:
        text = self._assistant_text(messages)
        return AgentResult(
            messages=messages,
            text=text,
            stop_reason=stop_reason,
            turns=self.usage.turns,
            usage=self.usage.snapshot(),
            cost=self.cost(),
            error=error,
            run_id=self._run_id,
        )

    def _instrumented_policy(self, turn: int) -> RetryPolicy | None:
        policy = self.retry_policy
        if policy is None:
            return None
        user_cb = policy.on_retry

        def _count_and_forward(attempt: RetryAttempt) -> None:
            self.usage.record_retry()
            self._emit(
                RetryAttemptEvent(
                    turn=turn,
                    attempt=attempt.attempt,
                    delay=attempt.delay,
                    exc_type=type(attempt.exc).__name__,
                    exc_message=str(attempt.exc),
                )
            )
            if user_cb is not None:
                user_cb(attempt)

        return replace(policy, on_retry=_count_and_forward)


__all__ = ["Agent", "MaxIterationsExceeded"]
