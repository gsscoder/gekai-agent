from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable

from ..directive_pump import detect_languages
from ..llm.errors import MaxIterationsExceeded
from ..llm.events import Event as LlmEvent, EventBus
from ..llm.model_caps import resolve_thinking_params
from ..llm.types import Message, TextBlock, ThinkingBlock
from ..events import (
    AgentEvent,
    DirectivePumpEvent,
    DiffEvent,
    DoneEvent,
    EstimateEvent,
    MaxIterationsEvent,
    ResponderEvent,
    ScaleEvent,
    SubAgentStartEvent,
    TaskGraphHaltedEvent,
    TaskGraphStartedEvent,
    ToolScopeEvent,
    VerifyEvent,
)
from ..permissions import PermissionCallback
from ..persistence import append_debug
from ..persona import ROOT_SYSTEM_PROMPT
from ..pipeline.estimate import Estimator
from ..pipeline.plan import Task, TaskGraph
from ..pipeline.sequencer import Sequencer
from ..pipeline.verifier import Verdict, Verifier
from ..session import Session
from ..text_format import clean_output
from ..tiers.catalog import TierName, TierPolicy
from ..tiers.resolve import ResolvedTier
from ..subagents import SUBAGENTS, Subagent
from ..tools import ExternalGrantCallback, HiddenGrantCallback
from .bridge import (
    bridge_llm_event,
    format_diff_summary,
    maybe_flag_foreign_instruction_file,
    truncate_diffs_block,
)
from . import dispatch
from .dispatch import (
    DispatchContext,
    enrich_system_base,
    gekai_md_system_base,
    pumped_system_base,
    run_subagent,
)
from .interpreter import TaskGraphHalted, StepResult, run_task_graph
from .scaling import WorkSignal, scale, _sequencer_signal
from .tool_scope import scope as tool_scope

_log = logging.getLogger(__name__)

_ROOT_COLOR = "#4169E1"
_RECENCY_N = 2


def _recency_turns(messages: list[dict], n: int) -> list[Message]:
    turns = [m for m in messages[:-1] if m["role"] in ("user", "assistant")]
    return [Message(role=m["role"], content=m["content"]) for m in turns[-(n * 2):]]


class Harness:
    def __init__(
        self,
        resolve: Callable[[TierName, str], ResolvedTier],
        sequencer_policy: TierPolicy,
        root_dispatch_policy: TierPolicy,
        subagent_dispatch_policy: TierPolicy,
        verifier_policy: TierPolicy,
        estimator: ResolvedTier | None = None,
        verbose_telemetry: bool = True,
    ) -> None:
        # Assignment-time tier scaling (plan 28 Phase 2): the 4 scaled
        # touchpoints (sequencer, root-dispatch, subagent-dispatch, verifier)
        # no longer get one frozen `ResolvedTier` baked in at construction —
        # they get this resolver closure (over the current catalog+bindings)
        # plus each touchpoint's declared `TierPolicy`, so a dispatch site can
        # pick a tier per call (`scale(policy, signal)`) and resolve it fresh
        # (`resolve(tier, touchpoint_name)` — the touchpoint name travels with
        # the tier because `scale()` throws the touchpoint's identity away,
        # and its declared operating point must survive tier movement).
        # `estimator` stays optional and still frozen: it is not one of the
        # scaled components — tests that don't wire one get the old flat
        # behavior (pre-plan-27 "trivial"). Root's synthesis
        # (`_respond`) runs at `root_dispatch_policy` like any other
        # root-dispatch call (plan 32 Phase 3: root absorbs the Responder,
        # which had its own separate frozen tier).
        self._resolve = resolve
        self._sequencer_policy = sequencer_policy
        self._root_dispatch_policy = root_dispatch_policy
        self._subagent_dispatch_policy = subagent_dispatch_policy
        self._verifier_policy = verifier_policy
        self._verbose_telemetry = verbose_telemetry
        self._estimator = Estimator(estimator) if estimator is not None else None
        # `Sequencer` is no longer built once here: the sequencer's tier is
        # chosen per `_stream_graph()` call (pre-plan signal), so it is
        # constructed fresh there, against a freshly resolved tier.

    async def stream(
        self,
        session: Session,
        user_input: str,
        permission_callback: PermissionCallback | None = None,
        hidden_grant_callback: HiddenGrantCallback | None = None,
        external_grant_callback: ExternalGrantCallback | None = None,
        seed: str | None = None,
    ) -> AsyncIterator[AgentEvent | str]:
        bus = EventBus()

        subagent: Subagent | None = None
        estimate_duration_ms = 0
        if seed is not None:
            # Explicit slash-alias dispatch (e.g. `/refactor ...`): a single
            # task action bound to this session, not a mutation for the
            # sequencer to plan. Bypasses the sequencer/task graph entirely
            # and resolves straight to a cold-ish subagent run below — this
            # is the only way `subagent` here ever gets bound (delegate.py's
            # `run_subagent` builds its own `Agent` and never calls this).
            estimate_decision = "dispatch"
            subagent = next((s for s in SUBAGENTS if s.name == seed), None)
            if subagent is None:
                raise ValueError(f"unknown seed agent {seed!r}")
        elif self._estimator is not None:
            t0 = time.monotonic()
            estimate = await self._estimator.estimate(user_input, history=session.messages)
            estimate_duration_ms = round((time.monotonic() - t0) * 1000)
            estimate_decision = estimate.scope
        else:
            estimate_decision = "solo"  # no estimator wired — safest default, no planning
        yield EstimateEvent(decision=estimate_decision, specialists=[], duration_ms=estimate_duration_ms)

        if estimate_decision == "mutate":
            async for item in self._stream_graph(
                session, user_input, permission_callback, hidden_grant_callback,
                external_grant_callback, bus,
            ):
                yield item
            return

        async for item in self._stream_solo(
            session, user_input, permission_callback, subagent,
            hidden_grant_callback, external_grant_callback, bus, chat_rung=(estimate_decision == "chat"),
        ):
            yield item

    async def _stream_solo(
        self,
        session: Session,
        user_input: str,
        permission_callback: PermissionCallback | None,
        subagent: Subagent | None,
        hidden_grant_callback: HiddenGrantCallback | None,
        external_grant_callback: ExternalGrantCallback | None,
        bus: EventBus,
        *,
        chat_rung: bool,
        emit_start_event: bool = True,
    ) -> AsyncIterator[AgentEvent | str]:
        """No-graph turn: trivial single-agent (root), a seed-dispatched
        subagent run bound to this session (warm context, no domain
        pump/GEKAI.md — decision 2/3 of the seed-dispatch fix — those two
        channels carry session/project state and would break the subagent's
        isolation from root's conversation; the language axis (plan 36
        Phase 3, below) is exempt from that invariant, since it is
        deterministic, mission-free craft resolved fresh from this call's own
        `user_input` — never a leak of session state), or the
        Sequencer-failure fallback (`_stream_graph`'s `except ValueError`) —
        the same solo run any of those cases would have taken had the turn
        never been routed into a task graph. Runs at root-dispatch's
        configured default, always — there is no per-node signal here (no
        verify/retry concept exists in this path), so it never modulates
        (plan 28 Phase 2 guardrail).

        `emit_start_event=False` is only for the Sequencer-failure fallback:
        `_stream_graph` already yielded a `SubAgentStartEvent` for
        name="sequencer" before falling back here, and a turn must carry
        exactly one such event, not two.
        """
        pumped_domains: list[str] = []
        detected_languages: list[str] = []
        if subagent is None:
            system_base = ROOT_SYSTEM_PROMPT
            if not chat_rung:
                system_base, pumped_domains = pumped_system_base(system_base)
            system_base = gekai_md_system_base(system_base, session)
        else:
            # plan 36 Phase 3: a seed-dispatched subagent is bound to
            # `user_input` at this same moment root's own domain pump above
            # resolves — mirrors that call, on the language axis instead.
            if subagent.language_aware:
                detected_languages = detect_languages(user_input, session.working_dir)
            system_base = subagent.build_system_base(languages=detected_languages)
        system_base = enrich_system_base(system_base, session.working_dir)
        if pumped_domains:
            yield DirectivePumpEvent(domains=pumped_domains)
        if detected_languages:
            yield DirectivePumpEvent(languages=detected_languages)

        root_dispatch_resolved = self._resolve(self._root_dispatch_policy.default, "root-dispatch")
        if chat_rung:
            # The chat rung strips reasoning params on chit-chat. A bare `{}`
            # here is "unspecified", which providers like DeepSeek default to
            # reasoning ON — so this must resolve to root's model's explicit-
            # disable payload, not an empty dict.
            effective_extra_params = resolve_thinking_params(root_dispatch_resolved.model, enabled=False)
        else:
            effective_extra_params = root_dispatch_resolved.extra_params
        if subagent is not None:
            # Seed-dispatch's tool ceiling clamp (decision 4): mirrors the
            # graph path's `tool_scope(matched.tool_policy, step_scope)`,
            # here with `step_scope=None` always (no sequencer step exists),
            # so this only ever applies the subagent's own ceiling — the
            # scope reason is always "default" (nothing to narrow further),
            # so no `ToolScopeEvent` telemetry, unlike the graph path.
            tools_override: frozenset[str] | None
            tools_override, _scope_reason = tool_scope(subagent.tool_policy, None)
        else:
            tools_override = None
        agent = dispatch.build_agent(
            dataclasses.replace(root_dispatch_resolved, extra_params=effective_extra_params),
            session.working_dir, session.permissions, permission_callback, system_base, bus,
            subagent=subagent,
            hidden_grant_callback=hidden_grant_callback,
            external_grant_callback=external_grant_callback,
            tools_override=tools_override,
        )
        if self._verbose_telemetry:
            append_debug(session, {"content": {"system": agent.system, "extra_params": effective_extra_params}})

        # `None` is the sentinel `bridge_llm_event` pushes when this run's own
        # `AgentStopped` arrives, ending the consumption loop below.
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        files_touched: list[str] = []
        root_run_id = uuid.uuid4().hex

        def _on_event(event: LlmEvent) -> None:
            bridge_llm_event(
                event, queue, session, files_touched, self._verbose_telemetry, root_run_id,
                emit_text_chunks=subagent is None,
            )
            # Root-only (decision 11): `stream()` also runs a cold subagent
            # outside a task graph (`subagent` set, no estimator/graph
            # involved) — a specialist's incidental `read_file` must never
            # raise this, only a root dispatch the user is actually driving.
            if subagent is None:
                maybe_flag_foreign_instruction_file(event, queue)

        if emit_start_event:
            yield SubAgentStartEvent(
                name=subagent.name if subagent else "root",
                description=subagent.description if subagent else "thinking",
                color=_ROOT_COLOR,
            )

        # `stream()`'s own `subagent` param is only ever bound by the
        # seed-dispatch path above — a genuine graph-spawned subagent run
        # goes through `delegate.py`'s `run_subagent`, which builds its own
        # `Agent` directly and never calls `stream()`. A seed-dispatched
        # subagent is a single task action bound to this session (decision
        # 1), so it gets warm session context like any root turn, never `[]`.
        prior = _recency_turns(session.messages, _RECENCY_N)
        prior.append(Message(role="user", content=user_input))
        unsubscribe = bus.subscribe(_on_event)
        agent_task: asyncio.Task = asyncio.create_task(agent.run(prior, run_id=root_run_id))

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

    async def _stream_graph(
        self,
        session: Session,
        user_input: str,
        permission_callback: PermissionCallback | None,
        hidden_grant_callback: HiddenGrantCallback | None,
        external_grant_callback: ExternalGrantCallback | None,
        bus: EventBus,
    ) -> AsyncIterator[AgentEvent | str]:
        """Case 3/4 mutation path: sequencer produces a validated TaskGraph, the
        fixed interpreter walks it (plan 27 improvements 2-3; renamed plan 28).
        Every step is a cold, fire-and-forget subagent run whose start/outcome
        are traced on `bus` — the same bridge the single-agent path above
        uses, so DiffEvent/LogEvent/Delegation* rendering is shared, not
        reimplemented. Root never runs inside the graph (plan 32 Phase 3); it
        synthesizes the turn's answer afterward, in `_respond`.
        """
        working_dir = session.working_dir
        permissions = session.permissions
        queue: asyncio.Queue = asyncio.Queue()
        files_touched: list[str] = []
        diff_summaries: list[str] = []
        all_diff_events: list[DiffEvent] = []

        # Per-dispatch diff/mutation tracking for the verifier gate (below):
        # populated SYNCHRONOUSLY inside `bridge_llm_event`, called directly
        # by `bus.emit()` — unlike `all_diff_events` above, which only fills
        # up as this generator's own consumption loop drains `queue` at its
        # own pace, racing against `graph_task`. `dispatch()` awaits the
        # entire nested run before reading these, so by construction there is
        # no cross-task race: every mutation from that dispatch has already
        # landed by the time `dispatch()` resumes.
        dispatch_diff_sink: list[DiffEvent] = []
        dispatch_mutation_count: list[int] = [0]

        def _on_event(event: LlmEvent) -> None:
            bridge_llm_event(
                event, queue, session, files_touched, self._verbose_telemetry, run_id=None,
                diff_sink=dispatch_diff_sink, mutation_count=dispatch_mutation_count,
            )

        unsubscribe = bus.subscribe(_on_event)

        # SubAgentStartEvent is always the first event (AgentEvent contract) —
        # the TUI's per-step renderer setup relies on it, same as the
        # single-agent path; nested steps then render as delegation badges
        # underneath it via the same DelegationStart/DoneEvent bridge.
        yield SubAgentStartEvent(name="sequencer", description="sequencing", color=_ROOT_COLOR)

        # Sequencer pre-plan signal (plan 28 Phase 2): a cheap, engineered
        # read of the raw prompt, computed before the task graph exists (no
        # `Task.verify` to read yet) — demotes CORE->SUPP on a clearly-easy
        # request, otherwise stays at the configured default.
        sequencer_tier, sequencer_reason = scale(self._sequencer_policy, _sequencer_signal(user_input))
        resolved_sequencer = self._resolve(sequencer_tier, "sequencer")
        if sequencer_tier != self._sequencer_policy.default:
            yield ScaleEvent(
                component="sequencer",
                default_tier=self._sequencer_policy.default.value,
                chosen_tier=sequencer_tier.value,
                reason=sequencer_reason,
            )
        sequencer = Sequencer(resolved_sequencer)

        try:
            graph = await sequencer.sequence(user_input)
        except ValueError as exc:
            unsubscribe()
            _log.warning("sequencer failed to produce a valid task graph; falling back to solo: %s", exc)
            async for item in self._stream_solo(
                session, user_input, permission_callback, None,
                hidden_grant_callback, external_grant_callback, bus, chat_rung=False, emit_start_event=False,
            ):
                yield item
            return

        yield TaskGraphStartedEvent(
            step_count=len(graph),
            agents=[s.agent for s in graph],
            verify_placements=sum(1 for s in graph if s.verify),
            summary=graph.summary,
            steps=[s.instruction for s in graph],
        )

        def _scale_and_resolve(policy, component, signal):
            tier, reason = scale(policy, signal)
            resolved = self._resolve(tier, component)
            if tier != policy.default:
                queue.put_nowait(ScaleEvent(
                    component=component,
                    default_tier=policy.default.value,
                    chosen_tier=tier.value,
                    reason=reason,
                ))
            return resolved

        async def dispatch(
            agent_name: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal(),
            step_scope: str | None = None,
        ) -> str:
            resolved = _scale_and_resolve(self._subagent_dispatch_policy, "subagent-dispatch", signal)
            matched = next((s for s in SUBAGENTS if s.name == agent_name), None)
            tools_override, scope_reason = tool_scope(matched.tool_policy if matched else None, step_scope)
            if scope_reason != "default":
                queue.put_nowait(ToolScopeEvent(unit=agent_name, chosen_rung=step_scope, reason=scope_reason))
            # plan 36 Phase 3: same `ToolScopeEvent` queuing pattern above,
            # for the language axis — `run_subagent` resolves this same
            # detection again internally for the actual system-base build
            # (it has no `queue` to telemeter through), so this is telemetry
            # only, not the injection itself.
            if matched is not None and matched.language_aware:
                detected_languages = detect_languages(instruction, working_dir)
                if detected_languages:
                    queue.put_nowait(DirectivePumpEvent(languages=detected_languages))
            diffs_before = len(dispatch_diff_sink)
            mutations_before = dispatch_mutation_count[0]
            result = await run_subagent(
                agent_name, instruction,
                mission=mission,
                ctx=DispatchContext(
                    model=resolved.model, api_key=resolved.api_key, api_base=resolved.api_base,
                    extra_params=resolved.extra_params, working_dir=working_dir,
                    permissions=permissions, permission_callback=permission_callback,
                    bus=bus, hidden_grant_callback=hidden_grant_callback,
                    external_grant_callback=external_grant_callback,
                    session=session, verbose_telemetry=self._verbose_telemetry,
                ),
                tools_override=tools_override,
                can_delegate=True,
            )
            last_dispatch_diffs_cell[0] = dispatch_diff_sink[diffs_before:]
            last_dispatch_files_delta[0] = dispatch_mutation_count[0] - mutations_before
            return result

        # ponytail: single-element cell smuggling "diffs emitted by the most
        # recent dispatch() call" from `dispatch` to `verify_agent`. Safe
        # without locking because run_task_graph (interpreter.py) awaits
        # every dispatch() strictly sequentially, one step at a time —
        # never concurrently — so there's no race on this shared slot.
        last_dispatch_diffs_cell: list[list[DiffEvent]] = [[]]
        last_dispatch_files_delta: list[int] = [0]

        async def verify_agent(step: Task, out: str, attempt: int, index: int) -> Verdict:
            diffs = last_dispatch_diffs_cell[0]
            coding_diffs = [d for d in diffs if d.via in ("edit_file", "overwrite")]
            if not coding_diffs:
                if attempt >= 1:
                    # This is the re-check after a repair dispatch — a repair
                    # that produced no coding diff at all is evidence the
                    # repair failed, not evidence there's nothing to check.
                    # Fail closed so the interpreter's halt logic fires,
                    # instead of silently promoting a still-broken step to ok.
                    violations = ["repair dispatch produced no code changes"]
                    queue.put_nowait(VerifyEvent(
                        step_index=index, agent=step.agent, ok=False,
                        violation_count=len(violations), violations=violations,
                        chosen_tier="", duration_ms=0, gate="repair_no_op",
                    ))
                    return Verdict(ok=False, violations=violations)
                # Gate: nothing but new-file writes / no diffs at all — treat as
                # non-coding (fs scaffolding, pure discovery, etc). No LLM call.
                if last_dispatch_files_delta[0] > 0:
                    # The dispatch mutated files even though the gate found no
                    # coding diff to check (e.g. a new-file write, or a
                    # move/delete-only step) — not a no-op, so log it even
                    # though the usual never-emit-on-skip convention applies.
                    queue.put_nowait(VerifyEvent(
                        step_index=index, agent=step.agent, ok=True, violation_count=0,
                        chosen_tier="", duration_ms=0, gate="skipped_with_mutations",
                    ))
                return Verdict(ok=True, violations=[])

            diff_text = "\n\n".join(format_diff_summary(d) for d in coding_diffs)
            signal = WorkSignal(retry=attempt)  # attempt 0 -> SUPP (default), attempt 1 (re-check after repair) -> promoted toward CORE
            tier, reason = scale(self._verifier_policy, signal)
            resolved = self._resolve(tier, "verifier")
            if tier != self._verifier_policy.default:
                queue.put_nowait(ScaleEvent(
                    component="verifier", default_tier=self._verifier_policy.default.value,
                    chosen_tier=tier.value, reason=reason,
                ))
            verifier = Verifier(resolved)
            start = time.monotonic()
            verdict = await verifier.verify(
                user_input=user_input, step_instruction=step.instruction,
                diff_text=diff_text, output=out,
            )
            queue.put_nowait(VerifyEvent(
                step_index=index,
                agent=step.agent, ok=verdict.ok, violation_count=len(verdict.violations),
                violations=verdict.violations,
                chosen_tier=tier.value, duration_ms=round((time.monotonic() - start) * 1000),
            ))
            return verdict

        graph_task: asyncio.Task[list[StepResult]] = asyncio.create_task(
            run_task_graph(graph, dispatch, user_input, verify_agent=verify_agent)
        )

        try:
            while not graph_task.done():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                if isinstance(item, DiffEvent):
                    diff_summaries.append(format_diff_summary(item))
                    all_diff_events.append(item)
                yield item
            while not queue.empty():
                item = queue.get_nowait()
                if isinstance(item, DiffEvent):
                    diff_summaries.append(format_diff_summary(item))
                    all_diff_events.append(item)
                yield item

            try:
                results = await graph_task
            except TaskGraphHalted as halted:
                yield TaskGraphHaltedEvent(step_index=halted.index, agent=halted.step.agent, reason=halted.reason)
                yield DoneEvent(thinking_chars=0, files_touched=files_touched)
                answer, responder_event = await self._respond(
                    user_input, session, graph, halted.results, halted=halted,
                    permission_callback=permission_callback, hidden_grant_callback=hidden_grant_callback,
                    external_grant_callback=external_grant_callback,
                    files_touched=files_touched, diff_summaries=diff_summaries,
                )
                if responder_event is not None:
                    yield responder_event
                yield answer
                return

            yield DoneEvent(thinking_chars=0, files_touched=files_touched)
            answer, responder_event = await self._respond(
                user_input, session, graph, results, halted=None,
                permission_callback=permission_callback, hidden_grant_callback=hidden_grant_callback,
                external_grant_callback=external_grant_callback,
                files_touched=files_touched, diff_summaries=diff_summaries,
            )
            if responder_event is not None:
                yield responder_event
            yield answer
        finally:
            unsubscribe()

    async def _respond(
        self,
        user_input: str,
        session: Session,
        graph: TaskGraph,
        results: list[StepResult],
        *,
        halted: TaskGraphHalted | None,
        permission_callback: PermissionCallback | None,
        hidden_grant_callback: HiddenGrantCallback | None,
        external_grant_callback: ExternalGrantCallback | None,
        files_touched: list[str],
        diff_summaries: list[str],
    ) -> tuple[str, ResponderEvent | None]:
        """Synthesizes the turn's user-facing answer from what actually ran —
        a real root call (plan 32 Phase 3: root absorbs the Responder), run
        at `root-dispatch` with session context (`_recency_turns` + the
        graph's own summary + each step's output), the same tier-scaling/
        resolve pattern the no-graph path in `stream()` uses. Covers both the
        success path and the halted path (root reports the halt too). Fail-
        soft: any exception — empty synthesis, model/network error — falls
        back to the mechanical `_recap` so a turn is never lost to its own
        wrap-up."""
        t0 = time.monotonic()
        try:
            resolved = self._resolve(self._root_dispatch_policy.default, "root-dispatch")
            system_base, _pumped_domains = pumped_system_base(ROOT_SYSTEM_PROMPT)
            system_base = gekai_md_system_base(system_base, session)
            system_base = enrich_system_base(system_base, session.working_dir)
            agent = dispatch.build_agent(
                resolved,
                session.working_dir, session.permissions, permission_callback, system_base,
                hidden_grant_callback=hidden_grant_callback,
                external_grant_callback=external_grant_callback,
            )
            outputs_block = "\n\n".join(
                f"--- step {i + 1} output ---\n{r.output}" for i, r in enumerate(results)
            )
            files_touched_line = f"files actually modified this turn: {', '.join(files_touched) if files_touched else '(none)'}"
            diffs_block = truncate_diffs_block(
                "\n\n".join(diff_summaries) if diff_summaries else "(no diffs captured)"
            )
            if halted is None:
                synthesis_prompt = (
                    f"the task graph you planned has finished. summary: {graph.summary}\n\n"
                    f"{files_touched_line}\n\n"
                    f"<diffs>\n{diffs_block}\n</diffs>\n\n"
                    f"<step_outputs>\n{outputs_block}\n</step_outputs>\n\n"
                    "write the reply the user will see: state what was done and answer any question "
                    "asked, using only the step outputs above — ground every claim about what changed "
                    "in <diffs> above, not just what the step outputs say happened; be terse, no "
                    "preamble, no markdown headers"
                )
            else:
                synthesis_prompt = (
                    f"the task graph you planned halted before finishing. summary: {graph.summary}\n\n"
                    f"{files_touched_line}\n\n"
                    f"<diffs>\n{diffs_block}\n</diffs>\n\n"
                    f"<step_outputs>\n{outputs_block}\n</step_outputs>\n\n"
                    f"step {halted.index + 1} ({halted.step.agent}) HALTED: {halted.reason}\n"
                    + (
                        f"the halted step's own last output, before it was judged as not done, was:\n"
                        f"{halted.last_output}\n\n"
                        if halted.last_output else ""
                    )
                    + "prior steps' work is kept; nothing was rolled back\n\n"
                    "write the reply the user will see: report what was completed and name the step "
                    "that halted and why (drawing on its own last output above, when present), using "
                    "only the information above — ground every claim about what changed in <diffs> "
                    "above, not just what the step outputs say happened; a halted step's listed "
                    "violations are real defects — report them plainly, do not paper over them with an "
                    "optimistic summary; be terse, no preamble, no markdown headers"
                )
            prior = _recency_turns(session.messages, _RECENCY_N)
            prior.append(Message(role="user", content=synthesis_prompt))
            history = await agent.run(prior)
            answer = _last_assistant_text(history)
            if not answer:
                raise ValueError("root synthesis returned empty text")
            if not files_touched:
                answer = f"{answer}\n\nno files were modified this turn"
            return answer, ResponderEvent(duration_ms=round((time.monotonic() - t0) * 1000), fell_back=False)
        except Exception as exc:
            _log.warning("root synthesis failed; falling back to mechanical recap: %s", exc)
            recap = _recap(graph, halted=halted)
            if not files_touched:
                recap = f"{recap}\n\nno files were modified this turn"
            return recap, ResponderEvent(duration_ms=round((time.monotonic() - t0) * 1000), fell_back=True)


def _last_assistant_text(history: list[Message]) -> str:
    for msg in reversed(history):
        if msg.role == "assistant":
            if isinstance(msg.content, list):
                text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
            else:
                text = msg.content or ""
            return clean_output(text)
    return ""


def _recap(graph: TaskGraph, *, halted: TaskGraphHalted | None) -> str:
    """Mechanical (no narrator LLM call) recap of the task graph's outcome —
    this is both the yielded assistant text (so it persists into session
    history for next-turn continuity, plan 27 hard problem 4) and the
    rendered checkpoint artifact (decision 15)."""
    if halted is None:
        return graph.summary
    last_output_line = f"its last output was:\n{halted.last_output}\n" if halted.last_output else ""
    return (
        f"{graph.summary}\n"
        f"step {halted.index + 1} ({halted.step.agent}) HALTED: {halted.reason}\n"
        f"{last_output_line}"
        "prior steps' work is kept; nothing was rolled back."
    )
