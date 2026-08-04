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

from agent.events import DoneEvent, ScaleEvent, TaskGraphHaltedEvent, TaskGraphStartedEvent, ToolScopeEvent
from agent.harness import core as harness_core
from agent.harness.core import Harness
from agent.llm.providers.base import ProviderAdapter
from agent.llm.resolve import ResolvedTier
from agent.llm.tiers import TierName, TierPolicy
from agent.llm.types import CompletionResponse, Message, StreamDone, TextBlock
from agent.pipeline.estimate import ScopeEstimate
from agent.pipeline.plan import Task, TaskGraph
from agent.pipeline.sequencer import Sequencer
from agent.session import Session
from agent.settings import Permissions
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
        lambda _tier: ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})
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


class _ScriptedAdapter(ProviderAdapter):
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


def _tier_distinguishing_resolve(tier: TierName) -> ResolvedTier:
    """Fake resolver returning a distinguishable `.model` per tier (SUPP vs
    CORE), so a test can assert which tier a dispatch actually resolved and
    used."""
    return ResolvedTier(model=f"{tier.value}-model", api_key="key", api_base=None, extra_params={})


def _make_scaling_harness(resolve: Callable[[TierName], ResolvedTier]) -> Harness:
    """Like `_make_harness`, but takes a caller-supplied `resolve` instead of
    a flat one, so scaling tests can tell which tier a dispatch used."""
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    estimator_tier = ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={})
    with patch("agent.harness.core.OpenAIAdapter"):
        harness = Harness(
            resolve=resolve,
            sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
            root_dispatch_policy=policy,
            subagent_dispatch_policy=policy,
            estimator=estimator_tier,
        )
    return harness


class _RecordingAdapter(ProviderAdapter):
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
    of building a real `AsyncOpenAI` client, so a test can assert which
    resolved tier's config the sequencer's freshly-built `Sequencer` (built
    per `_stream_graph()` call, plan 28 Phase 2) actually received."""
    calls: list[dict] = []

    def _init(self, model, api_key=None, api_base=None, extra_params=None) -> None:
        calls.append({"model": model, "api_key": api_key, "api_base": api_base, "extra_params": extra_params})

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
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

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
    assert collected[-1] == "build a library with tests"


def test_seed_routes_through_sequencer_even_without_mutate_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    harness = _make_harness()
    harness._estimator.estimate = AsyncMock(side_effect=AssertionError("estimator must not run when seeded"))
    graph = TaskGraph(summary="add coverage", steps=[Task(agent="test-expert", instruction="add coverage", mission="add coverage")])
    plan_mock = AsyncMock(return_value=graph)
    _patch_sequencer_sequence(monkeypatch, plan_mock)

    async def fake_run_subagent(agent, task, **kwargs):
        return "done"

    monkeypatch.setattr(harness_core, "run_subagent", fake_run_subagent)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "add coverage", seed="test-expert"))

    plan_mock.assert_awaited_once()
    _, kwargs = plan_mock.await_args
    assert kwargs.get("seed") == "test-expert"
    assert any(isinstance(e, TaskGraphStartedEvent) for e in collected)


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
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _RecordingAdapter)

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


def test_sequencer_demotes_on_short_simple_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # REQ: item 5 -- a short, mechanically-simple request demotes the
    # sequencer off its configured CORE default to SUPP (`_sequencer_signal`,
    # scaling.py); proven end-to-end via the actual resolved config the
    # freshly-built `Sequencer` receives, plus the yielded `ScaleEvent`.
    harness = _make_scaling_harness(_tier_distinguishing_resolve)
    harness._estimator.estimate = AsyncMock(side_effect=AssertionError("estimator must not run when seeded"))
    sequencer_calls = _capturing_sequencer_init(monkeypatch)
    graph = TaskGraph(summary="renamed", steps=[Task(agent="code-expert", instruction="rename it", mission="rename it")])
    _patch_sequencer_sequence(monkeypatch, AsyncMock(return_value=graph))

    _RecordingAdapter.instances = []
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _RecordingAdapter)

    session = _make_session(tmp_path)
    # 3 words, none in the multi-step keyword list -> guaranteed demote
    # (scaling.py's own test_sequencer_signal_demotes_short_simple_prompt
    # uses this exact prompt).
    prompt = "rename this variable"
    assert len(prompt.split()) <= 15  # sanity: within _SEQUENCER_WORD_LIMIT

    collected = run(_drain(harness, session, prompt, seed="code-expert"))

    assert sequencer_calls[0]["model"] == "supp-model", "Sequencer should be built from the SUPP-resolved config"
    sequencer_events = [e for e in _only(collected, ScaleEvent) if e.component == "sequencer"]
    assert sequencer_events == [
        ScaleEvent(component="sequencer", default_tier="core", chosen_tier="supp", reason="easy-demote")
    ]


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
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _RecordingAdapter)

    session = _make_session(tmp_path)
    prompt = "add a login endpoint and write tests for it"

    collected = run(_drain(harness, session, prompt))

    assert sequencer_calls[0]["model"] == "core-model", "Sequencer should stay on the CORE-resolved config"
    sequencer_events = [e for e in _only(collected, ScaleEvent) if e.component == "sequencer"]
    assert sequencer_events == []


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
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _RecordingAdapter)

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
    return fake_agent


def _spy_build_agent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _fake(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return _fake_history_agent()

    monkeypatch.setattr(harness_core, "_build_agent", _fake)
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
    monkeypatch.setattr(harness_core, "_enrich_system_base", lambda base, working_dir: base)

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
    monkeypatch.setattr(harness_core, "_enrich_system_base", lambda base, working_dir: base)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "do anything"))

    # calls[0] is the step's dispatch; calls[1] is the trailing root-dispatch
    # synthesis call (`_respond`, plan 32 Phase 3) -- not under test here.
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
    monkeypatch.setattr(harness_core, "_enrich_system_base", lambda base, working_dir: base)

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
    monkeypatch.setattr(harness_core, "_enrich_system_base", lambda base, working_dir: base)

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
