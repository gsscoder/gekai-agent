"""Coverage for `/tier` (agent/commands/tier.py): the headless half of the
command — listing. Interactive assignment needs live UI and is only
reachable through `agent/tui/app.py::_run_tier_wizard` (see
tests/test_tier_wizard.py); calling `TierCommand.execute` with a tier arg
directly (as this file does for the error-path check) is otherwise
unreachable from the real app, which intercepts `/tier <TIER>` before
dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import settings
from agent.commands.tier import TierCommand
from agent.llm.tiers import ModelCatalogEntry, TierBinding, TierName

pytestmark = pytest.mark.asyncio

FLASH = ModelCatalogEntry(
    name="flash", base_url="https://a.example.com", efforts=("low", "medium"), thinking=False,
)
PRO = ModelCatalogEntry(
    name="pro", base_url="https://b.example.com", efforts=("high", "xhigh"), thinking=True,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(settings, "DEFAULT_MODEL_CATALOG", (FLASH, PRO))


@pytest.fixture(autouse=True)
def _keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)


async def _run(*args: str):
    return await TierCommand().execute(list(args))


async def test_no_args_lists_every_tier_including_unconfigured_ones() -> None:
    settings.save_tier_binding(TierName.FAST, TierBinding(model="flash", default_effort="low", thinking=False))

    result = await _run()

    assert result.error is False
    assert result.reconfigure is False
    lines = result.output.splitlines()
    assert len(lines) == 3
    assert lines[0].split() == ["FAST", "flash", "low", "no", "✓", "ready"]
    assert lines[1].split() == ["SUPP", "—", "—", "—", "not", "configured"]
    assert lines[2].split() == ["CORE", "—", "—", "—", "not", "configured"]


async def test_listing_reports_a_bound_model_whose_key_went_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings.save_tier_binding(TierName.FAST, TierBinding(model="flash", default_effort="low", thinking=False))
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: False)

    result = await _run()

    assert result.output.splitlines()[0].split()[-2:] == ["no", "key"]


async def test_calling_with_a_tier_arg_directly_errors_since_assignment_is_ui_only() -> None:
    # Confirms the headless command never tries to assign anything itself —
    # the real app never reaches this path (see the module docstring).
    result = await _run("CORE")

    assert result.error is True
    assert "interactive" in result.output
    assert settings.load_tier_bindings() == {}
