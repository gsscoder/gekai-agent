"""Coverage for `GekaiAgent.__init__`'s plan 28 Phase 1b wiring: every
touchpoint's model/creds/effort/thinking now comes from `resolve_touchpoint()`
against the global tier catalog+bindings, not `GEKAI_CORE_*`/`GEKAI_SUPPORT_*`
env vars.

Isolation follows tests/test_resolve.py's pattern (monkeypatch
`agent.llm.resolve.credentials.has_api_key`/`get_api_key` directly — no real
keyring) plus monkeypatching `agent.agent.load_model_catalog`/
`load_tier_bindings` (the names imported into `agent.agent`'s namespace) —
no real `~/.gekai/settings.json` touched. `agent.logging.Path.home` is also
patched so `EventLogger`'s always-on log file lands in `tmp_path`, not the
real home directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent import agent as agent_module
from agent import logging as agent_logging
from agent.agent import GekaiAgent
from agent.llm.tiers import ModelCatalogEntry, TierBinding, TierName
from agent.pipeline import Route
from agent.settings import Permissions

CATALOG = {
    "fast-model": ModelCatalogEntry(name="fast-model", base_url="https://fast.example.com", efforts=("low", "medium"), thinking=False),
    "supp-model": ModelCatalogEntry(name="supp-model", base_url="https://supp.example.com", efforts=("low", "medium"), thinking=False),
    "core-model": ModelCatalogEntry(name="core-model", base_url="https://core.example.com", efforts=("high", "xhigh"), thinking=True),
}

BINDINGS = {
    TierName.FAST: TierBinding(model="fast-model", default_effort="low"),
    TierName.SUPP: TierBinding(model="supp-model", default_effort="low"),
    TierName.CORE: TierBinding(model="core-model", default_effort="high", thinking=True),
}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_logging.Path, "home", classmethod(lambda cls: tmp_path))


def _stub_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.llm.resolve.credentials.get_api_key", lambda name: f"key-for-{name}")


def _make_agent(tmp_path: Path) -> GekaiAgent:
    return GekaiAgent(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))


def test_construction_succeeds_when_tiers_unconfigured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Construction must NOT raise: this constructor runs before the TUI (and
    # so before `/tiers`) exists — raising here would permanently lock an
    # unconfigured install out of the only place that can fix it.
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: {})
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: {})

    agent = _make_agent(tmp_path)
    assert agent._gate is None
    assert agent._main is None
    assert agent.model == "unconfigured"


def test_gate_and_process_stream_raise_lazily_when_tiers_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: {})
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: {})

    agent = _make_agent(tmp_path)

    async def _drive_gate() -> None:
        await agent.gate("hello")

    async def _drive_process_stream() -> None:
        async for _ in agent.process_stream(agent.start_session(), "hello", Route(trivial=True)):
            pass

    with pytest.raises(RuntimeError, match="/tiers"):
        asyncio.run(_drive_gate())

    with pytest.raises(RuntimeError, match="/tiers"):
        asyncio.run(_drive_process_stream())


def test_construction_succeeds_and_wires_each_touchpoint_to_its_resolved_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: dict(CATALOG))
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: dict(BINDINGS))
    _stub_credentials(monkeypatch)

    agent = _make_agent(tmp_path)
    assert agent._gate is not None
    assert agent._main is not None

    # sequencer (CORE) is the "default model" stand-in surfaced on the agent
    assert agent.model == "core-model"
    assert agent._api_key == "key-for-core-model"
    assert agent._api_base == "https://core.example.com"

    # gate (FAST)
    assert agent._gate._model == "fast-model"

    # estimator (FAST)
    assert agent._main._estimator is not None
    assert agent._main._estimator._model == "fast-model"

    # sequencer (CORE) -> Planner
    assert agent._main._planner._model == "core-model"

    # main-dispatch (SUPP)
    assert agent._main._main_dispatch.model == "supp-model"
    assert agent._main._main_dispatch.api_key == "key-for-supp-model"

    # subagent-dispatch (SUPP)
    assert agent._main._subagent_dispatch.model == "supp-model"
    assert agent._main._subagent_dispatch.api_key == "key-for-supp-model"


def test_gate_self_heals_after_tiers_configured_mid_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a running GekaiAgent only resolved touchpoints once, at
    # construction. `/tiers` runs inside that same already-constructed agent
    # and only writes to disk — without a retry, every call after a
    # mid-session `/tiers` fix kept raising the pre-`/tiers` error forever
    # (confirmed via a real session log: 4 identical "'fast' is not
    # configured" errors after the user had already saved all 3 bindings).
    current_bindings: dict[TierName, TierBinding] = {}
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: dict(CATALOG))
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: dict(current_bindings))
    _stub_credentials(monkeypatch)
    # Gate.gate() would otherwise make a real network call — stub it so this
    # test only exercises GekaiAgent's own retry/self-heal logic.
    from agent.pipeline.gate import Gate, Route as GateRoute

    async def _stub_gate(self, *a, **kw) -> GateRoute:
        return GateRoute()

    monkeypatch.setattr(Gate, "gate", _stub_gate)

    agent = _make_agent(tmp_path)
    assert agent._gate is None

    async def _drive_gate() -> None:
        await agent.gate("hello")

    with pytest.raises(RuntimeError, match="/tiers"):
        asyncio.run(_drive_gate())

    # Simulate `/tiers` saving all three bindings mid-session (disk changes
    # under the already-running agent, nothing re-constructs it).
    current_bindings.update(BINDINGS)

    asyncio.run(_drive_gate())  # must NOT raise now
    assert agent._gate is not None
    assert agent._gate._model == "fast-model"


def test_construction_succeeds_when_partially_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    partial = {TierName.FAST: BINDINGS[TierName.FAST]}
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: dict(CATALOG))
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: partial)
    _stub_credentials(monkeypatch)

    agent = _make_agent(tmp_path)
    assert agent._gate is None
    assert agent._main is None

    async def _drive_gate() -> None:
        await agent.gate("hello")

    with pytest.raises(RuntimeError, match="/tiers"):
        asyncio.run(_drive_gate())
