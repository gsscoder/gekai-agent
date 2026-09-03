"""Coverage for `GekaiAgent.__init__`'s plan 28 Phase 1b wiring: every
touchpoint's model/creds/effort/thinking now comes from `resolve_touchpoint()`
against the global tier catalog+bindings, not `GEKAI_CORE_*`/`GEKAI_SUPPORT_*`
env vars.

Isolation follows tests/test_resolve.py's pattern (monkeypatch
`agent.tiers.resolve.credentials.has_api_key`/`get_api_key` directly — no real
keyring — keyed by `provider:model`, so a stub returns `key-for-<credential_key>`) plus monkeypatching `agent.agent.load_model_catalog`/
`load_tier_bindings` (the names imported into `agent.agent`'s namespace) —
no real `~/.gekai/settings.json` touched. `agent.telemetry.Path.home` is also
patched so `EventLogger`'s always-on log file lands in `tmp_path`, not the
real home directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent import agent as agent_module
from agent import telemetry as agent_telemetry
from agent.agent import GekaiAgent
from agent.tiers.catalog import TierBinding, TierName
from agent.permissions import Permissions
from tests.conftest import TIER_BINDINGS, TIER_CATALOG


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_telemetry.Path, "home", classmethod(lambda cls: tmp_path))


def _stub_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tiers.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.tiers.resolve.credentials.get_api_key", lambda name: f"key-for-{name}")


def _make_agent(tmp_path: Path) -> GekaiAgent:
    return GekaiAgent(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))


def test_construction_succeeds_when_tiers_unconfigured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Construction must NOT raise: this constructor runs before the TUI (and
    # so before `/models`/`/tier`) exists — raising here would permanently lock an
    # unconfigured install out of the only place that can fix it.
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: {})
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: {})

    agent = _make_agent(tmp_path)
    assert agent._root is None
    assert agent.model == "unconfigured"


def test_process_stream_raises_lazily_when_tiers_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: {})
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: {})

    agent = _make_agent(tmp_path)

    async def _drive_process_stream() -> None:
        async for _ in agent.process_stream(agent.start_session(), "hello"):
            pass

    with pytest.raises(RuntimeError, match="/tier"):
        asyncio.run(_drive_process_stream())


def test_construction_succeeds_and_wires_each_touchpoint_to_its_resolved_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: dict(TIER_CATALOG))
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: dict(TIER_BINDINGS))
    _stub_credentials(monkeypatch)

    agent = _make_agent(tmp_path)
    assert agent._root is not None

    # sequencer (CORE) is the "default model" stand-in surfaced on the agent
    assert agent.model == "core-model"
    assert agent._api_key == "key-for-openai:core-model"
    assert agent._api_base == "https://core.example.com"

    # estimator (FAST)
    assert agent._root._estimator is not None
    assert agent._root._estimator._tier.model == "fast-model"

    # sequencer/root-dispatch/subagent-dispatch (plan 28 Phase 2) are no
    # longer frozen `ResolvedTier`s on the Harness — they're a resolver
    # closure plus each touchpoint's `TierPolicy`; the policy default is
    # what a no-signal (unscaled) dispatch resolves to, matching Phase 1b's
    # frozen behavior exactly.
    assert agent._root._sequencer_policy.default is TierName.CORE
    assert agent._root._resolve(agent._root._sequencer_policy.default, "sequencer").model == "core-model"

    assert agent._root._root_dispatch_policy.default is TierName.SUPP
    root_dispatch_resolved = agent._root._resolve(agent._root._root_dispatch_policy.default, "root-dispatch")
    assert root_dispatch_resolved.model == "supp-model"
    assert root_dispatch_resolved.api_key == "key-for-openai:supp-model"

    assert agent._root._subagent_dispatch_policy.default is TierName.SUPP
    subagent_dispatch_resolved = agent._root._resolve(
        agent._root._subagent_dispatch_policy.default, "subagent-dispatch"
    )
    assert subagent_dispatch_resolved.model == "supp-model"
    assert subagent_dispatch_resolved.api_key == "key-for-openai:supp-model"


def test_process_stream_self_heals_after_tiers_configured_mid_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a running GekaiAgent only resolved touchpoints once, at
    # construction. `/tier` runs inside that same already-constructed agent
    # and only writes to disk — without a retry, every call after a
    # mid-session `/tier` fix kept raising the pre-command error forever
    # (confirmed via a real session log: 4 identical "'fast' is not
    # configured" errors after the user had already saved all 3 bindings).
    current_bindings: dict[TierName, TierBinding] = {}
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: dict(TIER_CATALOG))
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: dict(current_bindings))
    _stub_credentials(monkeypatch)
    # `Harness.stream()` would otherwise make a real network call — stub it
    # so this test only exercises GekaiAgent's own retry/self-heal logic.
    from agent.harness import Harness

    async def _stub_stream(self, *a, **kw):
        return
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(Harness, "stream", _stub_stream)

    agent = _make_agent(tmp_path)
    assert agent._root is None

    async def _drive_process_stream() -> None:
        async for _ in agent.process_stream(agent.start_session(), "hello"):
            pass

    with pytest.raises(RuntimeError, match="/tier"):
        asyncio.run(_drive_process_stream())

    # Simulate `/tier` saving all three bindings mid-session (disk changes
    # under the already-running agent, nothing re-constructs it).
    current_bindings.update(TIER_BINDINGS)

    asyncio.run(_drive_process_stream())  # must NOT raise now
    assert agent._root is not None


def test_construction_succeeds_when_partially_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    partial = {TierName.FAST: TIER_BINDINGS[TierName.FAST]}
    monkeypatch.setattr(agent_module, "load_model_catalog", lambda: dict(TIER_CATALOG))
    monkeypatch.setattr(agent_module, "load_tier_bindings", lambda: partial)
    _stub_credentials(monkeypatch)

    agent = _make_agent(tmp_path)
    assert agent._root is None

    async def _drive_process_stream() -> None:
        async for _ in agent.process_stream(agent.start_session(), "hello"):
            pass

    with pytest.raises(RuntimeError, match="/tier"):
        asyncio.run(_drive_process_stream())
