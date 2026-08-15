from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable

from agent.llm import Agent
from agent.llm.errors import MaxIterationsExceeded
from agent.llm.events import AgentStopped, DelegationCompleted, DelegationStarted, Event as LlmEvent, EventBus, TextChunkReceived, ThinkingChunkReceived, ToolExecutionCompleted, ToolExecutionStarted, UsageUpdated
from agent.llm.providers.openai import OpenAIAdapter
from agent.llm.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

from pathlib import Path
from typing import TYPE_CHECKING

from ..directive_pump import pump as pump_directives
from ..llm.model_caps import resolve_thinking_params
from ..llm.tiers import TierName, TierPolicy
from ..permissions import PermissionCallback, PermissionGate
from ..persistence import append_debug
from ..persona import ROOT_SYSTEM_PROMPT, render_tool_instruction
from ..pipeline.estimate import Estimator
from ..pipeline.plan import Task, TaskGraph
from ..pipeline.sequencer import Sequencer
from ..harness.interpreter import TaskGraphHalted, StepResult, run_task_graph
from ..harness.scaling import WorkSignal, scale, _sequencer_signal
from ..session import Session
from ..settings import Permissions
from ..text_format import clean_output
from ..diff import build_diff
from ..events import BudgetExhaustedEvent, DelegationDoneEvent, DelegationStartEvent, DirectivePumpEvent, DiffEvent, DoneEvent, EstimateEvent, ForeignFileDetectedEvent, InferEndEvent, LogEvent, MaxIterationsEvent, AgentEvent, ResponderEvent, ScaleEvent, TaskGraphHaltedEvent, TaskGraphStartedEvent, SubAgentStartEvent, TextChunkEvent, ThinkingTokenEvent, ToolScopeEvent
from ..shell import resolve_shell
from ..subagents import SUBAGENTS, Subagent
from ..tools import HiddenGrantCallback, make_tools
from ..tools.delegate import make_delegate_tool, run_subagent
from .tool_scope import scope as tool_scope

if TYPE_CHECKING:
    # deferred: agent.llm.resolve imports agent.harness.touchpoints, which
    # would otherwise cycle back through agent.harness's package __init__
    # (`from .core import Harness`) before this module finishes loading.
    # `from __future__ import annotations` (top of file) makes this
    # type-checking-only import safe for the annotations below.
    from ..llm.resolve import ResolvedTier

_ROOT_COLOR = "#4169E1"
_RECENCY_N = 2


def _pumped_system_base(system_base: str, prompt: str) -> tuple[str, list[str]]:
    """Dynamic-directive pump (plan 28 Phase 3, decision 12): appends the
    escaping directives of this prompt's mechanically-detected domains, if
    any. Never called for a subagent dispatch — pumping is root-only
    (decision 13, the specialist already carries its own directives)."""
    text, domains = pump_directives(prompt)
    if text:
        system_base = f"{system_base}\n<domain_directives>\n{text}"
    return system_base, domains


def _gekai_md_system_base(system_base: str, session: Session) -> str:
    """GEKAI.md injection (plan 35 Concept 1): root's own project-context
    file, appended verbatim beside the domain-directive pump, under its own
    `<project_instructions source="GEKAI.md">` tag so the model can always
    tell the user's rules from Gekai's own `<directives>` block. Root-only
    by construction, not by a flag: both call sites below only reach this
    function from the `subagent is None` branch, mirroring exactly how
    `_pumped_system_base` is root-only (decision 10 / plan 28 decision 13)
    — a subagent's context is isolated by design, and project instructions
    are precisely the channel that isolation exists to close.

    The system base — not a simulated read-and-understand turn in message
    history — is where this lives because message history is what
    `/compact` evicts. A GEKAI.md seeded as a turn would silently stop
    applying somewhere around turn 40 with no signal to anyone; the system
    base is reassembled every turn and never falls out of the window.

    No truncation, no normalization, no reordering (decision 1): if the
    file is oversized that is a telemetry fact for later, not a silent
    edit here. Nothing is appended when there is nothing to inject — same
    "don't emit an empty tag" discipline as `_pumped_system_base`.
    """
    if session.gekai_md is None:
        return system_base
    return f'{system_base}\n<project_instructions source="GEKAI.md">\n{session.gekai_md.text}'


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

    # `delegate` is declared, not ambient (plan-delegate-reintroduction Phase
    # 3): only a subagent (never root — `subagent is None` excludes it by
    # construction, no redundant check needed) that names targets in its own
    # `delegates_to` gets the tool, and only when this build is itself
    # allowed to delegate (`can_delegate`, the depth-1 cap — a subagent
    # reached via delegation is always built with `can_delegate=False`, see
    # `tools/delegate.py`'s `run_subagent`). Appended straight to `selected`
    # after the `tools_override` filter above, never through `make_tools()`
    # — `delegate` is not a filesystem/shell rung and must stay out of
    # `tools/catalog.py`. `parent_tools` captures this build's own selected
    # tool-name set (pre-delegate) so the child's effective grant can only
    # ever be tightened, never widened, past it (tighten-only invariant).
    if subagent is not None and subagent.delegates_to and can_delegate:
        selected = selected + [make_delegate_tool(
            subagent.delegates_to,
            model, api_key, api_base, extra_params, working_dir,
            permissions, permission_callback, bus, hidden_grant_callback,
            frozenset(t.name for t in selected),
        )]

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

    # Root still never receives `delegate` — it stays a pure work-operator;
    # the harness (sequencer + interpreter) owns all cross-agent control flow
    # for root's own dispatches (plan 27 decision 11, superseding plan 25's
    # agents-as-tools). Subagent-level delegation, wired above, is a
    # separate, bounded axis layered on top of that invariant, not a
    # reversal of it: declared per-unit (`delegates_to`), depth-capped at 1,
    # and tighten-only on tool capability.
    return agent


class Harness:
    def __init__(
        self,
        resolve: Callable[[TierName, str], ResolvedTier],
        sequencer_policy: TierPolicy,
        root_dispatch_policy: TierPolicy,
        subagent_dispatch_policy: TierPolicy,
        estimator: ResolvedTier | None = None,
        debug: bool = False,
    ) -> None:
        # Assignment-time tier scaling (plan 28 Phase 2): the 3 scaled
        # touchpoints (sequencer, root-dispatch, subagent-dispatch) no longer
        # get one frozen `ResolvedTier` baked in at construction — they get
        # this resolver closure (over the current catalog+bindings) plus
        # each touchpoint's declared `TierPolicy`, so a dispatch site can
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
        self._debug = debug
        self._estimator = (
            Estimator(
                model=estimator.model,
                api_key=estimator.api_key,
                api_base=estimator.api_base,
                extra_params=estimator.extra_params,
            )
            if estimator is not None
            else None
        )
        # `Sequencer` is no longer built once here: the sequencer's tier is
        # chosen per `_stream_graph()` call (pre-plan signal), so it is
        # constructed fresh there, against a freshly resolved tier.

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

        estimate_decision = "skipped"
        estimate_duration_ms = 0
        if subagent is None:
            if seed is not None:
                # Explicit slash-alias dispatch (e.g. `/refactor ...`): a
                # single task action bound to this session, not a mutation
                # for the sequencer to plan. Bypasses the sequencer/task
                # graph entirely and resolves straight to a cold-ish
                # subagent run below — this is the only way `stream()`'s own
                # `subagent` local ever gets bound (delegate.py's
                # `run_subagent` builds its own `Agent` and never calls
                # `stream()`).
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
                session, user_input, permission_callback, hidden_grant_callback, bus,
            ):
                yield item
            return

        # No-graph turn: trivial single-agent (root), or a seed-dispatched
        # subagent run bound to this session (warm context, no directive
        # pump/GEKAI.md — decision 2/3 of the seed-dispatch fix). Runs at
        # root-dispatch's configured default, always — there is no per-node
        # signal here (no verify/retry concept exists in this path), so it
        # never modulates (plan 28 Phase 2 guardrail).
        pumped_domains: list[str] = []
        if subagent is None:
            system_base, pumped_domains = _pumped_system_base(ROOT_SYSTEM_PROMPT, user_input)
            system_base = _gekai_md_system_base(system_base, session)
        else:
            system_base = subagent.build_system_base()
        system_base = _enrich_system_base(system_base, session.working_dir)
        if pumped_domains:
            yield DirectivePumpEvent(domains=pumped_domains)

        root_dispatch_resolved = self._resolve(self._root_dispatch_policy.default, "root-dispatch")
        chat_rung = estimate_decision == "chat"
        if extra_params is not None:
            effective_extra_params = extra_params
        elif chat_rung:
            # Estimator's chat rung (plan 33): strip reasoning params on
            # chit-chat. Formerly keyed off the now-deleted Gate/Route's
            # `trivial` flag (plan 33 Phase 2 merged that axis into the
            # Estimator; Phase 3 deleted Gate entirely). A bare `{}` here
            # is "unspecified", which providers like DeepSeek default to
            # reasoning ON (plan 34 phase 1's fix, applied here too) — so
            # this must resolve to root's model's explicit-disable payload,
            # not an empty dict.
            effective_extra_params = resolve_thinking_params(root_dispatch_resolved.model, enabled=False)
        else:
            effective_extra_params = root_dispatch_resolved.extra_params
        if chat_rung:
            # plan 34 phase 3: the chat rung carries no tool schemas — a
            # greeting/chit-chat turn never needs them, and dropping the 9
            # tool JSON schemas cuts root's prompt from ~2134 tokens toward
            # a few hundred. The system prompt itself is untouched (decision
            # 5) so a chat turn keeps solo's voice.
            tools_override: frozenset[str] | None = frozenset()
        elif subagent is not None:
            # Seed-dispatch's tool ceiling clamp (decision 4): mirrors the
            # graph path's `tool_scope(matched.tool_policy, step_scope)`,
            # here with `step_scope=None` always (no sequencer step exists),
            # so this only ever applies the subagent's own ceiling — the
            # scope reason is always "default" (nothing to narrow further),
            # so no `ToolScopeEvent` telemetry, unlike the graph path.
            tools_override, _scope_reason = tool_scope(subagent.tool_policy, None)
        else:
            tools_override = None
        agent = _build_agent(
            root_dispatch_resolved.model, root_dispatch_resolved.api_key, root_dispatch_resolved.api_base,
            effective_extra_params,
            session.working_dir, session.permissions, permission_callback, system_base, bus,
            subagent=subagent,
            hidden_grant_callback=hidden_grant_callback,
            tools_override=tools_override,
        )
        if self._debug:
            append_debug(session, {"content": {"system": agent.system, "extra_params": effective_extra_params}})

        queue: asyncio.Queue[
            LogEvent | DiffEvent | InferEndEvent | ThinkingTokenEvent | TextChunkEvent | BudgetExhaustedEvent
            | DelegationStartEvent | DelegationDoneEvent | ForeignFileDetectedEvent | None
        ] = asyncio.Queue()
        files_touched: list[str] = []
        root_run_id = uuid.uuid4().hex

        def _on_event(event: LlmEvent) -> None:
            _bridge_llm_event(
                event, queue, session, files_touched, self._debug, root_run_id,
                emit_text_chunks=subagent is None,
            )
            # Root-only (decision 11): `stream()` also runs a cold subagent
            # outside a task graph (`subagent` set, no estimator/graph
            # involved) — a specialist's incidental `read_file` must never
            # raise this, only a root dispatch the user is actually driving.
            if subagent is None:
                _maybe_flag_foreign_instruction_file(event, queue)

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

        def _on_event(event: LlmEvent) -> None:
            _bridge_llm_event(event, queue, session, files_touched, self._debug, run_id=None)

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
        sequencer = Sequencer(
            model=resolved_sequencer.model, api_key=resolved_sequencer.api_key,
            api_base=resolved_sequencer.api_base, extra_params=resolved_sequencer.extra_params,
        )

        try:
            graph = await sequencer.sequence(user_input)
        except ValueError as exc:
            unsubscribe()
            yield TaskGraphHaltedEvent(step_index=-1, agent="sequencer", reason=str(exc))
            yield DoneEvent(thinking_chars=0, files_touched=[])
            yield f"[sequencer failed to produce a valid task graph: {exc}]"
            return

        yield TaskGraphStartedEvent(
            step_count=len(graph),
            agents=[s.agent for s in graph],
            verify_placements=sum(1 for s in graph if s.verify),
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
            return await run_subagent(
                agent_name, instruction,
                mission=mission,
                model=resolved.model, api_key=resolved.api_key,
                api_base=resolved.api_base,
                extra_params=resolved.extra_params, working_dir=working_dir,
                permissions=permissions, permission_callback=permission_callback,
                bus=bus, hidden_grant_callback=hidden_grant_callback,
                tools_override=tools_override,
            )

        async def verify_agent(step: Task, out: str) -> bool:
            # No dedicated verdict-emitting verify agent exists yet (open point 2,
            # hard problem 3) — "mechanical" or any unnamed check is a placeholder:
            # non-empty output passes. A named post-planning-only agent gets a
            # PASS/FAIL-prefixed review request; anything else is a fail-open pass.
            roster_names = {s.name for s in SUBAGENTS}
            if step.verify in roster_names:
                verdict = await dispatch(
                    step.verify,
                    f"verify this step's output against its instruction, then answer PASS or FAIL "
                    f"on the first line.\n\ninstruction: {step.instruction}\n\noutput:\n{out}",
                )
                return verdict.strip().upper().startswith("PASS")
            return bool(out.strip())

        graph_task: asyncio.Task[list[StepResult]] = asyncio.create_task(
            run_task_graph(graph, dispatch, verify_agent=verify_agent)
        )

        try:
            while not graph_task.done():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                yield item
            while not queue.empty():
                yield queue.get_nowait()

            try:
                results = await graph_task
            except TaskGraphHalted as halted:
                yield TaskGraphHaltedEvent(step_index=halted.index, agent=halted.step.agent, reason=halted.reason)
                yield DoneEvent(thinking_chars=0, files_touched=files_touched)
                answer, responder_event = await self._respond(
                    user_input, session, graph, [], halted=halted,
                    permission_callback=permission_callback, hidden_grant_callback=hidden_grant_callback,
                )
                if responder_event is not None:
                    yield responder_event
                yield answer
                return

            yield DoneEvent(thinking_chars=0, files_touched=files_touched)
            answer, responder_event = await self._respond(
                user_input, session, graph, results, halted=None,
                permission_callback=permission_callback, hidden_grant_callback=hidden_grant_callback,
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
            system_base, _pumped_domains = _pumped_system_base(ROOT_SYSTEM_PROMPT, user_input)
            system_base = _gekai_md_system_base(system_base, session)
            system_base = _enrich_system_base(system_base, session.working_dir)
            agent = _build_agent(
                resolved.model, resolved.api_key, resolved.api_base, resolved.extra_params,
                session.working_dir, session.permissions, permission_callback, system_base,
                hidden_grant_callback=hidden_grant_callback,
            )
            outputs_block = "\n\n".join(
                f"--- step {i + 1} output ---\n{r.output}" for i, r in enumerate(results)
            )
            if halted is None:
                synthesis_prompt = (
                    f"the task graph you planned has finished. summary: {graph.summary}\n\n"
                    f"<step_outputs>\n{outputs_block}\n</step_outputs>\n\n"
                    "write the reply the user will see: state what was done and answer any question "
                    "asked, using only the step outputs above; be terse, no preamble, no markdown headers"
                )
            else:
                synthesis_prompt = (
                    f"the task graph you planned halted before finishing. summary: {graph.summary}\n"
                    f"step {halted.index + 1} ({halted.step.agent}) HALTED: {halted.reason}\n"
                    "prior steps' work is kept; nothing was rolled back.\n\n"
                    "write the reply the user will see: report what was completed and name the step "
                    "that halted and why, using only the information above; be terse, no preamble, "
                    "no markdown headers"
                )
            prior = _recency_turns(session.messages, _RECENCY_N)
            prior.append(Message(role="user", content=synthesis_prompt))
            history = await agent.run(prior)
            answer = _last_assistant_text(history)
            if not answer:
                raise ValueError("root synthesis returned empty text")
            return answer, ResponderEvent(duration_ms=round((time.monotonic() - t0) * 1000), fell_back=False)
        except Exception:
            return _recap(graph, halted=halted), ResponderEvent(duration_ms=round((time.monotonic() - t0) * 1000), fell_back=True)


def _maybe_flag_foreign_instruction_file(event: LlmEvent, queue: asyncio.Queue) -> None:
    """Plan 35 v3's foreign-file trigger, hung off the same
    `ToolExecutionCompleted` moment `_bridge_llm_event` already reacts to
    for `edit_file`/`write_file` — kept as its own function rather than a
    branch inside `_bridge_llm_event` because that function is shared with
    the graph path's cold subagent steps (`_stream_graph`'s own
    `_on_event`), which must never reach here (decision 11); the caller
    only invokes this under its own `subagent is None` guard.

    Markdown-only, no prefilter (v3 deletes the regex prefilter — decision
    5): every root-dispatched `.md` `read_file` result fires a
    `ForeignFileDetectedEvent` unconditionally; the one cheap LLM question
    (`agent.directive_audit.Auditor`) is the only judgment left, downstream.
    Whether the audit is even enabled is checked once, downstream, in
    `GekaiAgent.start_foreign_file_audit` — the same place GEKAI.md's own
    audit already checks it; not duplicated here.
    """
    if not isinstance(event, ToolExecutionCompleted):
        return
    if event.result.is_error or event.call.name != "read_file":
        return
    path = (event.call.input or {}).get("path", "")
    if not path.lower().endswith(".md"):
        return
    queue.put_nowait(ForeignFileDetectedEvent(rel_path=path, text=event.result.content))


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
    return (
        f"{graph.summary}\n"
        f"step {halted.index + 1} ({halted.step.agent}) HALTED: {halted.reason}\n"
        "prior steps' work is kept; nothing was rolled back."
    )


def _bridge_llm_event(
    event: LlmEvent,
    queue: asyncio.Queue,
    session: Session,
    files_touched: list[str],
    debug: bool,
    run_id: str | None,
    *,
    emit_text_chunks: bool = False,
) -> None:
    """Shared bus->queue bridge for both the single-agent path and the
    plan-interpreter path — tool/diff/thinking/delegation rendering is one
    implementation, not duplicated per path.

    `emit_text_chunks` (plan 34 Phase 2): only the no-graph direct-dispatch
    caller (`Harness.stream`) passes `True` here. The graph path
    (`_stream_graph`'s `dispatch()`) leaves it at the default `False` so a
    graph-routed step's streamed answer text — which would otherwise
    interleave with `LogEvent`/`DiffEvent` in a multi-step transcript — never
    reaches the TUI as a `TextChunkEvent`. Display-only either way: the
    persisted answer always comes from the assembled response, never from
    these chunks."""
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
    elif isinstance(event, TextChunkReceived):
        if emit_text_chunks:
            queue.put_nowait(TextChunkEvent(text=event.text))
    elif isinstance(event, DelegationStarted):
        queue.put_nowait(DelegationStartEvent(agent_name=event.agent, task=event.task, mission=event.mission))
    elif isinstance(event, DelegationCompleted):
        queue.put_nowait(DelegationDoneEvent(agent_name=event.agent))
    elif isinstance(event, AgentStopped) and run_id is not None:
        if event.run_id != run_id:
            return  # a nested run's own completion — not ours
        if event.budget_exhausted:
            queue.put_nowait(BudgetExhaustedEvent())
        queue.put_nowait(None)
