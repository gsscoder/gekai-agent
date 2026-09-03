"""Coverage for `Harness.stream`'s plan 27 improvement 4 wiring (renamed
plan 28): chit-chat/trivial stay on the single-agent path; mutate and a
`/agent-x` seed both route through the sequencer + interpreter
(`_stream_graph`) — the "one mutation path" (decision 4).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.harness import dispatch as harness_dispatch
from agent.events import DoneEvent, EstimateEvent, ScaleEvent, SubAgentStartEvent, TaskGraphHaltedEvent, TaskGraphStartedEvent, ToolScopeEvent, VerifyEvent
from agent.harness import core as harness_core
from agent.harness.core import Harness
from agent.harness.touchpoints import resolve_at_tier
from agent.llm import model_caps
from agent.llm.events import ToolExecutionCompleted
from agent.llm.model_caps import MODEL_CAPS, ModelCaps
from agent.tiers.resolve import ResolvedTier, resolve_tier
from agent.tiers.catalog import EFFORT_LADDER, ModelCatalogEntry, TierBinding, TierName, TierPolicy
from agent.llm.types import CompletionResponse, Message, StreamDone, TextBlock, ToolResultBlock, ToolUseBlock
from agent.pipeline.estimate import ScopeEstimate
from agent.pipeline.plan import Task, TaskGraph
from agent.pipeline.sequencer import Sequencer
from agent.pipeline.verifier import Verdict
from agent.session import Session
from agent.permissions import Permissions
from agent.tools.catalog import ALL_TOOLS, RUNGS


def run(coro):
    return asyncio.run(coro)


def _make_session(tmp_path: Path) -> Session:
    return Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))


def _only(events: list, cls: type) -> list:
    return [e for e in events if isinstance(e, cls)]


def _make_harness() -> Harness:
    """Plan 28 Phase 2: `Harness` takes a resolver closure + a `TierPolicy`
    per scaled touchpoint instead of one frozen `ResolvedTier` each. The
    resolver here always returns `root_tier`, so every touchpoint (including
    the sequencer's freshly-built `Sequencer`, now constructed inside
    `_stream_graph()` instead of `__init__`) still resolves to the exact
    same config these tests were written against."""
    return _make_scaling_harness(
        lambda _tier, _touchpoint: ResolvedTier(
            model="test-model", api_key="key", api_base="http://localhost", extra_params={}
        )
    )


def _patch_sequencer_sequence(monkeypatch: pytest.MonkeyPatch, mock: AsyncMock) -> None:
    """`Harness._stream_graph` now builds a fresh `Sequencer` per call (plan 28
    Phase 2 — the sequencer's tier is chosen per call), so a test can no
    longer stash a mock onto a pre-built `harness._sequencer` instance; patch
    `Sequencer.sequence` at the class level instead — every `Sequencer()` built
    during the test picks it up the same way an instance-attribute override
    used to."""
    monkeypatch.setattr(Sequencer, "sequence", mock)


async def _drain(harness: Harness, session: Session, prompt: str, **kwargs) -> list:
    collected = []
    async for item in harness.stream(session, prompt, **kwargs):
        collected.append(item)
    return collected


class _ScriptedAdapter:
    """Zero-arg-instantiable fake provider, mirroring tests/test_harness_core.py's
    pattern, so the single-agent path actually emits AgentStopped on the bus
    (a hand-rolled fake Agent that skips the bus never terminates the queue)."""

    responses: list[CompletionResponse] = []

    def __init__(self, *args, **kwargs) -> None:
        self._responses = list(type(self).responses)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        raise NotImplementedError

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield StreamDone(response=self._responses[len(self.calls) - 1])


# --- plan 28 Phase 2 (assignment-time tier scaling) integration fixtures ---
#
# `_make_harness`'s resolver is flat (returns the same `ResolvedTier`
# regardless of which `TierName` it's asked for) — fine for tests that don't
# care about scaling, but it can't distinguish a promoted/demoted dispatch
# from a no-op. The fixtures below add a resolver that *can* distinguish
# tiers, plus capture helpers, so these tests can prove scaling end-to-end
# through the real Harness/_stream_graph/dispatch wiring rather than only the
# already-unit-tested pure `scale()`/`node_signal()`/`_sequencer_signal()`
# functions (tests/test_scaling.py).


def _tier_distinguishing_resolve(tier: TierName, touchpoint_name: str) -> ResolvedTier:
    """Fake resolver returning a distinguishable `.model` per tier (SUPP vs
    CORE), so a test can assert which tier a dispatch actually resolved and
    used."""
    return ResolvedTier(model=f"{tier.value}-model", api_key="key", api_base=None, extra_params={})


def _make_scaling_harness(resolve: Callable[[TierName, str], ResolvedTier]) -> Harness:
    """Like `_make_harness`, but takes a caller-supplied `resolve` instead of
    a flat one, so scaling tests can tell which tier a dispatch used."""
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    estimator_tier = ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={})
    with patch("agent.harness.dispatch.OpenAIAdapter"):
        harness = Harness(
            resolve=resolve,
            sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
            root_dispatch_policy=policy,
            subagent_dispatch_policy=policy,
            verifier_policy=policy,
            estimator=estimator_tier,
        )
    return harness


class _RecordingAdapter:
    """Captures the `model` kwarg passed to `stream()` per instance
    (agent/llm/agent.py's `_provider_kwargs` threads `self.model` into every
    provider call) — lets a test tell which resolved tier's config a given
    `root` dispatch actually used. Mirrors `_ScriptedAdapter` above /
    test_harness_core.py's `_CapturingEventBus` class-level-list capture
    pattern; always answers with fixed, immediate text (no tool calls), since
    these tests care about *which model* a dispatch used, not its content.
    """

    instances: ClassVar[list["_RecordingAdapter"]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[dict] = []
        type(self).instances.append(self)

    async def complete(self, **kwargs):
        raise NotImplementedError

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield StreamDone(
            response=CompletionResponse(content=[TextBlock(text="step done")], stop_reason="end_turn")
        )


def _capturing_sequencer_init(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replaces `Sequencer.__init__` with one that records its kwargs instead
    of building a client, so a test can assert which
    resolved tier's config the sequencer's freshly-built `Sequencer` (built
    per `_stream_graph()` call, plan 28 Phase 2) actually received."""
    calls: list[dict] = []

    def _init(self, tier) -> None:
        calls.append({
            "model": tier.model, "api_key": tier.api_key,
            "api_base": tier.api_base, "extra_params": tier.extra_params,
        })

    monkeypatch.setattr(Sequencer, "__init__", _init)
    return calls


def test_trivial_estimate_skips_sequencer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="solo"))
    plan_mock = AsyncMock(side_effect=AssertionError("sequencer must not run on trivial"))
    _patch_sequencer_sequence(monkeypatch, plan_mock)
    _ScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "sum 10 numbers"))

    assert not any(isinstance(e, TaskGraphStartedEvent) for e in collected)
    plan_mock.assert_not_awaited()


def test_mutate_estimate_routes_through_sequencer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="build a library with tests",
        steps=[
            Task(agent="code-expert", instruction="write the library", mission="write the library"),
            Task(agent="test-expert", instruction="write tests", mission="write tests"),
        ],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))

    async def fake_run_subagent(agent, task, **kwargs):
        return f"{agent} done"

    monkeypatch.setattr(harness_core, "run_subagent", fake_run_subagent)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "build a library with tests"))

    started = _only(collected, TaskGraphStartedEvent)
    assert len(started) == 1
    assert started[0].step_count == 2
    assert started[0].agents == ["code-expert", "test-expert"]
    assert any(isinstance(e, DoneEvent) for e in collected)
    assert collected[-1] == "build a library with tests\n\nno files were modified this turn"


def test_seed_dispatches_directly_without_sequencer_or_estimator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: a `/alias` dispatch (`seed=...`) is a single task action bound to
    # the session -- it must bypass the sequencer/task graph entirely, never
    # touching the estimator or `Sequencer.sequence` (bug A's fix: routing a
    # non-auto-assignable seed like `code-refactorer`/`test-fixer` through
    # the sequencer's auto-assignable-only roster either mis-fires to a
    # different agent or raises inside `parse_task_graph`).
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(side_effect=AssertionError("estimator must not run when seeded"))
    plan_mock = AsyncMock(side_effect=AssertionError("sequencer must not run when seeded"))
    _patch_sequencer_sequence(monkeypatch, plan_mock)
    _ScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "add coverage", seed="test-expert"))

    estimate_events = _only(collected, EstimateEvent)
    assert len(estimate_events) == 1
    assert estimate_events[0].decision == "dispatch"
    assert not any(isinstance(e, TaskGraphStartedEvent) for e in collected)
    assert any(isinstance(e, DoneEvent) for e in collected)
    assert collected[-1] == "done"


def test_seed_dispatch_of_non_auto_assignable_code_refactorer_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: the literal regression from the live session bug report (bug A) --
    # `code-refactorer` is `auto_assignable=False` (delegate-only), so the
    # sequencer's own roster would never offer it as a legal choice; the
    # seed-dispatch path must never ask the sequencer at
    # all and must succeed cleanly.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(side_effect=AssertionError("estimator must not run when seeded"))
    plan_mock = AsyncMock(side_effect=AssertionError("sequencer must not run when seeded"))
    _patch_sequencer_sequence(monkeypatch, plan_mock)
    _ScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text="refactored")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "clean this up", seed="code-refactorer"))

    assert not any(isinstance(e, TaskGraphStartedEvent) for e in collected)
    assert any(isinstance(e, DoneEvent) for e in collected)
    assert collected[-1] == "refactored"


def test_seed_dispatch_of_non_auto_assignable_test_fixer_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: second confirmation of the fix class -- `test-fixer` is also
    # `auto_assignable=False`.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(side_effect=AssertionError("estimator must not run when seeded"))
    plan_mock = AsyncMock(side_effect=AssertionError("sequencer must not run when seeded"))
    _patch_sequencer_sequence(monkeypatch, plan_mock)
    _ScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text="fixed")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "fix the failing test", seed="test-fixer"))

    assert not any(isinstance(e, TaskGraphStartedEvent) for e in collected)
    assert any(isinstance(e, DoneEvent) for e in collected)
    assert collected[-1] == "fixed"


def test_seed_dispatch_of_non_auto_assignable_complexity_remover_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # third confirmation of the fix class -- `complexity-remover` is also
    # `auto_assignable=False` (never sequencer-routable, dispatched only via
    # its `/simplify` seed).
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(side_effect=AssertionError("estimator must not run when seeded"))
    plan_mock = AsyncMock(side_effect=AssertionError("sequencer must not run when seeded"))
    _patch_sequencer_sequence(monkeypatch, plan_mock)
    _ScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text="simplified")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "prune the dead code", seed="complexity-remover"))

    assert not any(isinstance(e, TaskGraphStartedEvent) for e in collected)
    assert any(isinstance(e, DoneEvent) for e in collected)
    assert collected[-1] == "simplified"


def test_graph_halt_yields_halted_event_and_recap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(summary="build something", steps=[Task(agent="code-expert", instruction="write it", mission="write it")])
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))

    async def failing_run_subagent(agent, task, **kwargs):
        return ""  # empty dispatch output -> TaskGraphHalted

    monkeypatch.setattr(harness_core, "run_subagent", failing_run_subagent)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "build something"))

    halted = _only(collected, TaskGraphHaltedEvent)
    assert len(halted) == 1
    assert halted[0].step_index == 0
    assert "prior steps" in collected[-1] or "HALTED" in collected[-1]


def test_sequencer_value_error_falls_back_to_solo_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: a Sequencer failure (e.g. the model returned an unparseable/
    # invalid task graph) must never dump its raw ValueError text into the
    # turn's answer -- the turn instead completes via the solo fallback
    # (`_stream_solo`), reusing the sequencer's own `SubAgentStartEvent`
    # rather than emitting a second one.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    plan_mock = AsyncMock(side_effect=ValueError("sequencer returned no JSON object: 'garbage'"))
    _patch_sequencer_sequence(monkeypatch, plan_mock)
    _ScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text="solo answer")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "get context on the banking pane cause I'm about to ask for a change"))

    assert not any(isinstance(e, TaskGraphHaltedEvent) for e in collected)
    assert any(isinstance(e, DoneEvent) for e in collected)
    assert collected[-1] == "solo answer"
    assert not any(isinstance(e, str) and "sequencer returned no JSON" in e for e in collected)
    # exactly one SubAgentStartEvent for the whole turn -- the sequencer's own,
    # reused by the solo fallback (`emit_start_event=False`), never a second one
    assert len(_only(collected, SubAgentStartEvent)) == 1


# --- plan 28 Phase 2 (assignment-time tier scaling) end-to-end coverage ---
#
# These prove the scaling wiring through the real Harness/_stream_graph/
# dispatch path — not just the pure `scale()`/`node_signal()`/
# `_sequencer_signal()` functions already unit-tested in isolation in
# tests/test_scaling.py.


def test_mechanical_verify_promotes_only_its_own_step_not_the_next(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ/EDGE: scale() is stateless per-dispatch (scaling.py module
    # docstring, decision 6) — a step 1 mechanical-verify promotion to CORE
    # must not leak into step 2's dispatch, which has no verify and must
    # resolve back at root-dispatch's configured default (SUPP). Also proves
    # a ScaleEvent is yielded for the adjusted step only (core.py's
    # `dispatch()`: `if tier != policy.default: queue.put_nowait(ScaleEvent(...))`).
    harness = _make_scaling_harness(_tier_distinguishing_resolve)
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="two specialist steps",
        steps=[
            Task(agent="code-expert", instruction="step one", mission="step one", verify="mechanical"),
            Task(agent="code-expert", instruction="step two", mission="step two", verify=None),
        ],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))

    _RecordingAdapter.instances = []
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _RecordingAdapter)

    session = _make_session(tmp_path)
    # Contains "and"/"both" -> `_sequencer_signal` stays neutral (keeps the
    # sequencer at its own CORE default) so the only ScaleEvent in this run
    # is the subagent-dispatch one under test, not a sequencer one too.
    prompt = "please handle both step one and step two carefully"
    collected = run(_drain(harness, session, prompt))

    # 2 step dispatches + 1 trailing root-dispatch synthesis call (`_respond`,
    # plan 32 Phase 3) -- the synthesis call resolves at root-dispatch's
    # default (SUPP), same as instance 1 below, and is not under test here.
    assert len(_RecordingAdapter.instances) == 3
    assert _RecordingAdapter.instances[0].calls[0]["model"] == "core-model", (
        "step 1 (verify='mechanical') should dispatch at the promoted CORE tier"
    )
    assert _RecordingAdapter.instances[1].calls[0]["model"] == "supp-model", (
        "step 2 (verify=None) should dispatch back at the default SUPP tier "
        "-- no memory of step 1's promotion"
    )

    scale_events = _only(collected, ScaleEvent)
    assert scale_events == [
        ScaleEvent(component="subagent-dispatch", default_tier="supp", chosen_tier="core", reason="reasoning-shaped")
    ]


def test_seed_dispatch_never_constructs_a_sequencer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # REQ: second confirmation of the sequencer-bypass fix -- a seeded
    # dispatch never even constructs a `Sequencer` (not just never awaits
    # `.sequence()`), and resolves its own dispatch at root-dispatch's
    # configured operating point like any other no-graph turn, with no
    # sequencer `ScaleEvent` (there is no sequencer signal to compute).
    harness = _make_scaling_harness(_tier_distinguishing_resolve)
    harness._estimator.estimate = AsyncMock(side_effect=AssertionError("estimator must not run when seeded"))
    sequencer_calls = _capturing_sequencer_init(monkeypatch)
    plan_mock = AsyncMock(side_effect=AssertionError("sequencer must not run when seeded"))
    _patch_sequencer_sequence(monkeypatch, plan_mock)

    _RecordingAdapter.instances = []
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _RecordingAdapter)

    session = _make_session(tmp_path)
    prompt = "rename this variable"

    collected = run(_drain(harness, session, prompt, seed="code-expert"))

    assert sequencer_calls == [], "Sequencer should never be constructed for a seeded dispatch"
    assert not any(isinstance(e, ScaleEvent) for e in collected)
    assert len(_RecordingAdapter.instances) == 1
    assert _RecordingAdapter.instances[0].calls[0]["model"] == "supp-model", (
        "seed dispatch resolves at root-dispatch's default (SUPP) -- no scaling signal applies"
    )


def test_sequencer_stays_at_core_on_multi_clause_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # REQ: item 5 (inverse) -- a longer, multi-clause request (contains
    # "and") keeps `_sequencer_signal` neutral, so the sequencer stays at its
    # own configured default (CORE): the `Sequencer` is built from the
    # CORE-resolved config, and no sequencer ScaleEvent is yielded.
    harness = _make_scaling_harness(_tier_distinguishing_resolve)
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    sequencer_calls = _capturing_sequencer_init(monkeypatch)
    graph = TaskGraph(
        summary="login + tests",
        steps=[Task(agent="code-expert", instruction="do it", mission="do it")],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))

    _RecordingAdapter.instances = []
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _RecordingAdapter)

    session = _make_session(tmp_path)
    prompt = "add a login endpoint and write tests for it"

    collected = run(_drain(harness, session, prompt))

    assert sequencer_calls[0]["model"] == "core-model", "Sequencer should stay on the CORE-resolved config"
    sequencer_events = [e for e in _only(collected, ScaleEvent) if e.component == "sequencer"]
    assert sequencer_events == []


def test_each_touchpoint_resolves_at_its_own_operating_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: the touchpoint identity survives the trip through `scale()` — the
    # resolver closure is handed a tier *and* a touchpoint name, so the
    # sequencer runs CORE's model thinking-off while a subagent dispatch
    # promoted onto that same CORE tier still gets the binding's thinking-on
    # operating point. Uses the real `resolve_tier`, not a fake resolver.
    catalog = {
        "supp-model": ModelCatalogEntry(name="supp-model", base_url=None, efforts=EFFORT_LADDER, thinking=False),
        "core-model": ModelCatalogEntry(name="core-model", base_url=None, efforts=EFFORT_LADDER, thinking=True),
    }
    bindings = {
        TierName.SUPP: TierBinding(model="supp-model", default_effort="low"),
        TierName.CORE: TierBinding(model="core-model", default_effort="max", thinking=True),
    }
    monkeypatch.setitem(MODEL_CAPS, "supp-model", ModelCaps(thinking_style="deepseek"))
    monkeypatch.setitem(MODEL_CAPS, "core-model", ModelCaps(thinking=True, thinking_style="deepseek"))
    # _EFFORT_TO_PARAMS is now keyed by model id, not thinking_style — these
    # fixture ids aren't real DeepSeek ids, so give "core-model" its own
    # fold-down entry to keep this test's model-agnostic intent.
    monkeypatch.setitem(model_caps._EFFORT_TO_PARAMS, "core-model", {"max": {"reasoning_effort": "max"}})
    monkeypatch.setattr("agent.tiers.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.tiers.resolve.credentials.get_api_key", lambda name: "key")
    resolved_at: list[tuple[str, str, dict]] = []

    def _resolve(tier: TierName, touchpoint_name: str) -> ResolvedTier:
        resolved = resolve_at_tier(tier, touchpoint_name, catalog, bindings)
        resolved_at.append((touchpoint_name, resolved.model, resolved.extra_params))
        return resolved

    harness = _make_scaling_harness(_resolve)
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    sequencer_calls = _capturing_sequencer_init(monkeypatch)
    graph = TaskGraph(
        summary="login + tests",
        steps=[Task(agent="code-expert", instruction="do it", mission="do it", verify="mechanical")],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))

    _RecordingAdapter.instances = []
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _RecordingAdapter)

    session = _make_session(tmp_path)
    run(_drain(harness, session, "add a login endpoint and write tests for it"))

    thinking_off = {"extra_body": {"thinking": {"type": "disabled"}}}
    thinking_on = {"reasoning_effort": "max", "extra_body": {"thinking": {"type": "enabled"}}}
    assert sequencer_calls[0]["model"] == "core-model"
    assert sequencer_calls[0]["extra_params"] == thinking_off, "sequencer declares thinking=False"
    # verify="mechanical" promotes the step's dispatch SUPP->CORE: same tier,
    # same model as the sequencer above, but no declared operating point of
    # its own, so it keeps CORE's binding.
    assert ("subagent-dispatch", "core-model", thinking_on) in resolved_at
    assert ("root-dispatch", "supp-model", thinking_off) in resolved_at  # SUPP binding is thinking-off


def test_single_agent_path_never_scales_or_emits_scale_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: item 6 -- the non-graph, trivial-estimate single-agent path
    # (`Harness.stream`'s no-graph branch) always resolves
    # `root_dispatch_policy.default` and structurally never calls `scale()`:
    # proven here by a fake `resolve` that *would* produce a distinguishable
    # config at another tier -- the dispatch still uses the SUPP (default)
    # model, and no `ScaleEvent` is ever yielded, even though this test's
    # `_make_scaling_harness` fixture is the exact one that does trigger
    # ScaleEvents on the graph path above.
    harness = _make_scaling_harness(_tier_distinguishing_resolve)
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="solo"))
    plan_mock = AsyncMock(side_effect=AssertionError("sequencer must not run on trivial"))
    _patch_sequencer_sequence(monkeypatch, plan_mock)

    _RecordingAdapter.instances = []
    monkeypatch.setattr(harness_dispatch, "OpenAIAdapter", _RecordingAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "sum 10 numbers"))

    assert not any(isinstance(e, TaskGraphStartedEvent) for e in collected)
    plan_mock.assert_not_awaited()
    assert not any(isinstance(e, ScaleEvent) for e in collected)
    assert len(_RecordingAdapter.instances) == 1
    assert _RecordingAdapter.instances[0].calls[0]["model"] == "supp-model"


# --- plan 31 Phase 3 (assignment-time tool scoping) end-to-end coverage ---
#
# These prove the wiring through the real Harness/_stream_graph/dispatch
# path: a graph step's `scope` narrows the dispatched unit's registered
# tools (harness/tool_scope.py's `scope()`), not just the pure function
# already unit-tested in isolation in tests/test_tool_scope.py.

_EXCLUDED_FROM_READ = ("edit_file", "write_file", "move_file", "copy_file", "delete_file", "make_dir")
_FULL_RUNG = frozenset(RUNGS[-1])


def _fake_history_agent() -> MagicMock:
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(
        return_value=[Message(role="assistant", content=[TextBlock(text="done")])]
    )
    # `.system` must be a real string, not the default MagicMock attribute:
    # verbose telemetry is on by default, so `run_subagent` always logs it to
    # debug.jsonl via `json.dumps`, which a MagicMock isn't serializable to.
    fake_agent.system = "sys"
    return fake_agent


def _spy_build_agent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _fake(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return _fake_history_agent()

    monkeypatch.setattr(harness_dispatch, "build_agent", _fake)
    return calls


def test_graph_step_scope_read_excludes_write_tools_for_full_ceiling_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # omni-worker declares a full ceiling (`ToolPolicy(ceiling=len(RUNGS) - 1)`,
    # same as root's old ROOT_TOOL_POLICY) -- root itself is never a step
    # agent post-plan-32-Phase-3, so a full-ceiling subagent is the
    # equivalent case for scope narrowing all the way down to "read".
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="read only",
        steps=[Task(agent="omni-worker", instruction="read stuff", mission="read stuff", scope="read")],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    calls = _spy_build_agent(monkeypatch)
    monkeypatch.setattr(harness_dispatch, "enrich_system_base", lambda base, working_dir: base)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "read stuff"))

    # calls[0] is the step's dispatch; calls[1] is the trailing root-dispatch
    # synthesis call (`_respond`, plan 32 Phase 3) -- not under test here.
    assert len(calls) == 2
    override = calls[0]["tools_override"]
    for name in _EXCLUDED_FROM_READ:
        assert name not in override
    assert override == frozenset(RUNGS[0])

    scope_events = _only(collected, ToolScopeEvent)
    assert scope_events == [ToolScopeEvent(unit="omni-worker", chosen_rung="read", reason="narrowed to 'read'")]


def test_graph_step_dispatch_forwards_can_delegate_true_to_build_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # a graph-spawned step is depth 0, not itself a delegation target -- the
    # interpreter's `dispatch` (harness/core.py) must pass `can_delegate=True`
    # through `run_subagent` so a unit declaring `delegates_to` (e.g.
    # omni-worker -> ws-explorer) actually receives its `delegate` tool.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="audit",
        steps=[Task(agent="omni-worker", instruction="audit stuff", mission="audit stuff", scope="read")],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    calls = _spy_build_agent(monkeypatch)
    monkeypatch.setattr(harness_dispatch, "enrich_system_base", lambda base, working_dir: base)

    session = _make_session(tmp_path)
    run(_drain(harness, session, "audit stuff"))

    # calls[0] is the step's dispatch; calls[1] is the trailing root-dispatch
    # synthesis call (`_respond`, plan 32 Phase 3) -- not under test here.
    assert calls[0]["can_delegate"] is True


def test_graph_step_scope_fs_gets_full_tools_for_full_ceiling_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="full access",
        steps=[Task(agent="omni-worker", instruction="do anything", mission="do anything", scope="fs")],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    calls = _spy_build_agent(monkeypatch)
    monkeypatch.setattr(harness_dispatch, "enrich_system_base", lambda base, working_dir: base)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "do anything"))

    # calls[0] is the step's dispatch; the verify gate (core.py's
    # `verify_agent` closure) only calls the LLM verifier when the step's
    # dispatch actually produced an `edit_file` diff -- `_spy_build_agent`'s
    # fake never emits one, so this step auto-passes with no repair; calls[1]
    # is the trailing root-dispatch synthesis call (`_respond`, plan 32
    # Phase 3) -- not under test here.
    assert len(calls) == 2
    assert calls[0]["tools_override"] == _FULL_RUNG == frozenset(ALL_TOOLS)
    assert not any(isinstance(e, ToolScopeEvent) for e in collected)


def test_graph_step_scope_none_gets_full_tools_for_full_ceiling_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="no signal",
        steps=[Task(agent="omni-worker", instruction="do it", mission="do it", scope=None)],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    calls = _spy_build_agent(monkeypatch)
    monkeypatch.setattr(harness_dispatch, "enrich_system_base", lambda base, working_dir: base)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "do it"))

    # calls[0] is the step's dispatch; calls[1] is the trailing root-dispatch
    # synthesis call (`_respond`, plan 32 Phase 3) -- not under test here.
    assert len(calls) == 2
    assert calls[0]["tools_override"] == _FULL_RUNG
    assert not any(isinstance(e, ToolScopeEvent) for e in collected)


def test_graph_step_scope_read_excludes_write_tools_for_subagent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="read only",
        steps=[Task(agent="code-expert", instruction="read stuff", mission="read stuff", scope="read")],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    calls = _spy_build_agent(monkeypatch)
    monkeypatch.setattr(harness_dispatch, "enrich_system_base", lambda base, working_dir: base)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "read stuff"))

    # calls[0] is the step's dispatch; calls[1] is the trailing root-dispatch
    # synthesis call (`_respond`, plan 32 Phase 3) -- not under test here.
    assert len(calls) == 2
    override = calls[0]["tools_override"]
    for name in _EXCLUDED_FROM_READ:
        assert name not in override
    assert override == frozenset(RUNGS[0])

    scope_events = _only(collected, ToolScopeEvent)
    assert scope_events == [ToolScopeEvent(unit="code-expert", chosen_rung="read", reason="narrowed to 'read'")]


# --- the coding-step verifier gate (core.py's `verify_agent` closure): an
# LLM verifier call only ever fires when the step's own dispatch actually
# produced an `edit_file` diff -- `step.verify`/`step.scope` no longer gate
# this at all (that gate moved out of the interpreter, plan 36) ---


def _make_touchless_run_subagent(text: str = "done, looks good"):
    """A fake `run_subagent` that returns non-empty text but never emits a
    `ToolExecutionCompleted` on the bus -- i.e. it never touches a file."""

    async def _run_subagent(agent, task, **kwargs):
        return text

    return _run_subagent


def _make_file_writing_run_subagent(path: str = "src/thing.py", text: str = "done, looks good"):
    """A fake `run_subagent` that emits a real `write_file`
    `ToolExecutionCompleted` on the bus before returning, so
    `_bridge_llm_event` records the path into `files_touched` exactly like a
    real tool-calling run would."""

    async def _run_subagent(agent, task, *, ctx, **kwargs):
        ctx.bus.emit(ToolExecutionCompleted(
            turn=0,
            call=ToolUseBlock(id="1", name="write_file", input={"path": path, "content": "x"}),
            result=ToolResultBlock(tool_use_id="1", content="ok"),
            duration_s=0.0,
        ))
        return text

    return _run_subagent


def _make_edit_run_subagent(path: str = "src/thing.py", text: str = "done, looks good"):
    """A fake `run_subagent` that emits a real `edit_file`
    `ToolExecutionCompleted` on the bus -- `_bridge_llm_event` turns it into
    a `DiffEvent(via="edit_file")`, the one thing the verify gate keys on to
    decide a step's dispatch is worth an LLM verifier call. The trailing
    `asyncio.sleep(0)` gives `_stream_graph`'s draining loop a chance to pull
    the DiffEvent off the queue before this returns, mirroring the real
    awaits a genuine tool-calling run always has after its last tool call."""

    async def _run_subagent(agent, task, *, ctx, **kwargs):
        ctx.bus.emit(ToolExecutionCompleted(
            turn=0,
            call=ToolUseBlock(id="1", name="edit_file", input={"path": path, "old_str": "old", "new_str": "new"}),
            result=ToolResultBlock(tool_use_id="1", content="ok"),
            duration_s=0.0,
        ))
        await asyncio.sleep(0)
        return text

    return _run_subagent


def _make_overwrite_run_subagent(path: str = "src/thing.py", text: str = "done, looks good"):
    """A fake `run_subagent` that emits a `write_file` `ToolExecutionCompleted`
    whose result content is `"ok: overwritten"` -- `_bridge_llm_event` turns
    this into a `DiffEvent(via="overwrite")`, which the verify gate now
    treats as coding-worthy same as `edit_file`."""

    async def _run_subagent(agent, task, *, ctx, **kwargs):
        ctx.bus.emit(ToolExecutionCompleted(
            turn=0,
            call=ToolUseBlock(id="1", name="write_file", input={"path": path, "content": "x"}),
            result=ToolResultBlock(tool_use_id="1", content="ok: overwritten"),
            duration_s=0.0,
        ))
        await asyncio.sleep(0)
        return text

    return _run_subagent


def test_write_file_overwrite_diff_triggers_verifier_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: a write_file overwrite (result content "ok: overwritten") mutates
    # existing code just like edit_file -- the gate must include "overwrite",
    # not just "edit_file", when deciding whether to call the LLM verifier.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(summary="overwrite a file", steps=[Task(agent="code-expert", instruction="overwrite the file", mission="overwrite the file")])
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    monkeypatch.setattr(harness_core, "run_subagent", _make_overwrite_run_subagent())
    verify_mock = AsyncMock(return_value=Verdict(ok=True))
    monkeypatch.setattr(harness_core.Verifier, "verify", verify_mock)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "overwrite the file"))

    verify_mock.assert_awaited_once()
    assert not any(isinstance(e, TaskGraphHaltedEvent) for e in collected)


def test_write_file_new_path_emits_skipped_with_mutations_verify_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ (Phase 3 telemetry): a step that mutated files (write_file to a new
    # path) but produced no edit_file/overwrite diff is a gate-skip, not a
    # no-op -- it must emit a VerifyEvent(gate="skipped_with_mutations") so
    # this escape is visible in telemetry, even though the LLM verifier
    # itself never runs.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(summary="write a file", steps=[Task(agent="code-expert", instruction="write the file", mission="write the file")])
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    monkeypatch.setattr(harness_core, "run_subagent", _make_file_writing_run_subagent())
    verify_mock = AsyncMock(return_value=Verdict(ok=True))
    monkeypatch.setattr(harness_core.Verifier, "verify", verify_mock)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "write the file"))

    verify_mock.assert_not_awaited()
    verify_events = _only(collected, VerifyEvent)
    assert len(verify_events) == 1
    assert verify_events[0].gate == "skipped_with_mutations"


def test_touchless_step_emits_no_verify_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ (Phase 3 telemetry, inverse): a genuinely touchless/discovery step
    # (no files mutated) stays on the pre-Phase-3 never-emit-on-no-op
    # convention -- no VerifyEvent at all.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="do something",
        steps=[Task(agent="code-expert", instruction="do it", mission="do it", scope="edit")],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    monkeypatch.setattr(harness_core, "run_subagent", _make_touchless_run_subagent())
    verify_mock = AsyncMock(return_value=Verdict(ok=True))
    monkeypatch.setattr(harness_core.Verifier, "verify", verify_mock)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "do it"))

    verify_mock.assert_not_awaited()
    assert not any(isinstance(e, VerifyEvent) for e in collected)


def test_verify_event_step_index_matches_real_graph_position_not_call_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ (bug-hunter finding, confirmed): VerifyEvent.step_index must be the
    # step's real 0-based position in the graph, not a running count of
    # verify calls made -- a preceding step that emits ZERO verify calls
    # (a genuinely silent gate-skip, no diff and no mutation) must not shift
    # a later step's reported step_index down. Step 0 here is silent; step 1
    # is a real edit, so its VerifyEvent must carry step_index=1, never 0.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="two steps",
        steps=[
            Task(agent="code-expert", instruction="look around", mission="look around"),
            Task(agent="code-expert", instruction="edit the file", mission="edit the file"),
        ],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))

    touchless = _make_touchless_run_subagent()
    editing = _make_edit_run_subagent()
    call_count = [0]

    async def _run_subagent(agent, task, *, ctx, **kwargs):
        # run_task_graph dispatches strictly in order (one `for` loop, no
        # concurrency) -- the first call is always step 0, the second always
        # step 1, regardless of what sibling-step text `_inject_request_summary`
        # wraps into either instruction.
        call_count[0] += 1
        fake = touchless if call_count[0] == 1 else editing
        return await fake(agent, task, ctx=ctx, **kwargs)

    monkeypatch.setattr(harness_core, "run_subagent", _run_subagent)
    verify_mock = AsyncMock(return_value=Verdict(ok=True))
    monkeypatch.setattr(harness_core.Verifier, "verify", verify_mock)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "do two things"))

    verify_mock.assert_awaited_once()
    verify_events = _only(collected, VerifyEvent)
    assert len(verify_events) == 1
    assert verify_events[0].step_index == 1


def test_no_diff_at_all_skips_the_verifier_and_never_halts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: a step whose dispatch touched no files at all -- regardless of
    # `scope` -- has nothing for the verify gate to check: no LLM call, no halt.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="do something",
        steps=[Task(agent="code-expert", instruction="do it", mission="do it", scope="edit")],
    )
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    monkeypatch.setattr(harness_core, "run_subagent", _make_touchless_run_subagent())
    verify_mock = AsyncMock(return_value=Verdict(ok=False, violations=["should never be seen"]))
    monkeypatch.setattr(harness_core.Verifier, "verify", verify_mock)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "do it"))

    verify_mock.assert_not_awaited()
    assert not any(isinstance(e, TaskGraphHaltedEvent) for e in collected)
    assert any(isinstance(e, DoneEvent) for e in collected)


def test_write_file_only_diff_skips_the_verifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: a new-file write is not a "coding" diff for the gate's purposes --
    # only an `edit_file` diff is worth an LLM verifier call.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(summary="write a file", steps=[Task(agent="code-expert", instruction="write the file", mission="write the file")])
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    monkeypatch.setattr(harness_core, "run_subagent", _make_file_writing_run_subagent())
    verify_mock = AsyncMock(return_value=Verdict(ok=False, violations=["should never be seen"]))
    monkeypatch.setattr(harness_core.Verifier, "verify", verify_mock)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "write the file"))

    verify_mock.assert_not_awaited()
    assert not any(isinstance(e, TaskGraphHaltedEvent) for e in collected)
    assert any(isinstance(e, DoneEvent) for e in collected)


def test_edit_file_diff_triggers_verifier_call_and_halts_on_repeated_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: an actual `edit_file` diff is what makes the gate call the LLM
    # verifier -- a failing verdict on both the original and repair attempts
    # halts the graph (the fixed repair/re-verify/halt policy, unchanged),
    # and the halt reason carries the verifier's own violations.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(summary="edit a file", steps=[Task(agent="code-expert", instruction="edit the file", mission="edit the file")])
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    monkeypatch.setattr(harness_core, "run_subagent", _make_edit_run_subagent())
    monkeypatch.setattr(
        harness_core.Verifier, "verify",
        AsyncMock(return_value=Verdict(ok=False, violations=["renamed field mismatch"])),
    )

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "edit the file"))

    halted = _only(collected, TaskGraphHaltedEvent)
    assert len(halted) == 1
    assert halted[0].step_index == 0
    assert "renamed field mismatch" in halted[0].reason


def test_edit_file_diff_triggers_verifier_call_and_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: happy path -- a passing verifier verdict on the real edit diff
    # lets the graph complete normally.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(summary="edit a file", steps=[Task(agent="code-expert", instruction="edit the file", mission="edit the file")])
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    monkeypatch.setattr(harness_core, "run_subagent", _make_edit_run_subagent())
    monkeypatch.setattr(harness_core.Verifier, "verify", AsyncMock(return_value=Verdict(ok=True)))

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "edit the file"))

    assert not any(isinstance(e, TaskGraphHaltedEvent) for e in collected)
    assert any(isinstance(e, DoneEvent) for e in collected)


def _make_batched_edit_run_subagent(path: str = "src/thing.py", text: str = "done, looks good"):
    """A fake `run_subagent` that emits a real batched-`edits` `edit_file`
    `ToolExecutionCompleted` on the bus -- the exact shape a production
    session actually used (see test_edit_file_batched_edits_emit_one_diff_event_per_hunk
    in test_harness_core.py) when the bridge's `edits` handling was missing
    and the verifier gate never fired once in a live session."""

    async def _run_subagent(agent, task, *, ctx, **kwargs):
        ctx.bus.emit(ToolExecutionCompleted(
            turn=0,
            call=ToolUseBlock(id="1", name="edit_file", input={"path": path, "edits": [
                {"old_str": "old", "new_str": "new"},
            ]}),
            result=ToolResultBlock(tool_use_id="1", content="ok"),
            duration_s=0.0,
        ))
        await asyncio.sleep(0)
        return text

    return _run_subagent


def test_batched_edit_file_diff_reaches_the_gate_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ (integration, plan Phase 4): the bridge (`_bridge_llm_event`'s
    # `edits` handling) and the gate (`verify_agent`'s `coding_diffs` check)
    # exercised TOGETHER through the real event bus, not each against a stub
    # of the other -- this is the exact production failure mode (session
    # f30d462c: 7 batched edit_file calls, zero verify events). Revert the
    # bridge's `edits` handling and this test fails, since no DiffEvent would
    # reach `verify_agent` and the gate would silently skip the LLM call.
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(summary="edit a file", steps=[Task(agent="code-expert", instruction="edit the file", mission="edit the file")])
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))
    monkeypatch.setattr(harness_core, "run_subagent", _make_batched_edit_run_subagent())
    verify_mock = AsyncMock(return_value=Verdict(ok=True))
    monkeypatch.setattr(harness_core.Verifier, "verify", verify_mock)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "edit the file"))

    verify_mock.assert_awaited_once()
    assert not any(isinstance(e, TaskGraphHaltedEvent) for e in collected)
