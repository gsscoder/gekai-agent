"""Coverage for the `/tiers` grid's pure logic helpers in `agent/tui/app.py`:
`_mask_key` (key-display masking), `_tiers_display_key`/`_tiers_key_present`
(precedence between an in-progress edit and the real keyring credential),
`_pending_row_status` (status-cell formatter for an in-progress edit — not
`agent/llm/resolve.py::tier_status()`, which answers the wrong question for
not-yet-committed edits), `_cycle_choice` (model/effort wrap-around cycling),
and `_tier_edit_complete` (the `[ok]` completeness gate).

There is no Pilot-based UI test in this repo (see the old
tests/test_tiers_flow.py, now removed since it only tested the dead
`_format_tier_binding` summary-line helper this feature replaced); the
interactive key-event cascade (`_handle_tiers_enter`, `action_navigate_*`,
`action_confirm_or_submit`, `action_cancel_stream`) requires a live `App` to
drive and is verified by trace-through only, mirroring how
tests/test_tiers_panel.py covers `TiersPanel` itself without mounting.
"""

from __future__ import annotations

import pytest

from agent.llm.tiers import ModelCatalogEntry, TierName, TierSuitability
from agent.tui.app import (
    _cycle_choice,
    _mask_key,
    _pending_row_status,
    _tier_credential_key,
    _tier_edit_complete,
    _tiers_display_key,
    _tiers_key_present,
)

_FLASH = ModelCatalogEntry(
    name="deepseek-v4-flash",
    base_url=None,
    efforts=("low", "medium", "high"),
    thinking=False,
    suitability=TierSuitability(fast="ok", supp="ok", core="warning"),
)
_PRO = ModelCatalogEntry(
    name="deepseek-v4-pro",
    base_url=None,
    efforts=("low", "medium", "high"),
    thinking=True,
    suitability=TierSuitability(fast="deprecated", supp="warning", core="ok"),
)
_CATALOG = {_FLASH.name: _FLASH, _PRO.name: _PRO}


# ---- _mask_key -------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("sk-1234567890abcdef", "sk" + "*" * 14 + "def"),
        ("sk123", "*****"),
        ("ab", "**"),
        ("sk-123", "sk*123"),
    ],
)
def test_mask_key(raw: str, expected: str) -> None:
    assert _mask_key(raw) == expected


# ---- _tiers_display_key / _tiers_key_present --------------------------


def test_display_key_no_key_when_nothing_staged_or_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: False)
    assert _tiers_display_key("m", {}) == "no key"
    assert _tiers_key_present("m", {}) is False


def test_display_key_masks_real_stored_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.tui.app.credentials.get_api_key", lambda name: "sk-1234567890abcdef")
    assert _tiers_display_key("m", {}) == "sk" + "*" * 14 + "def"
    assert _tiers_key_present("m", {}) is True


def test_display_key_staged_edit_wins_over_real_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.tui.app.credentials.get_api_key", lambda name: "sk-old-real-key-value")
    assert _tiers_display_key("m", {"m": "sk-new-staged-key"}) == _mask_key("sk-new-staged-key")
    assert _tiers_key_present("m", {"m": "sk-new-staged-key"}) is True


def test_display_key_explicit_clear_wins_over_real_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    assert _tiers_display_key("m", {"m": ""}) == "no key"
    assert _tiers_key_present("m", {"m": ""}) is False


# ---- _pending_row_status ----------------------------------------------


def test_row_status_not_configured_when_model_unset() -> None:
    assert _pending_row_status(TierName.FAST, None, "low", False, _CATALOG, {}) == "not configured"
    assert _pending_row_status(TierName.FAST, "deepseek-v4-flash", None, False, _CATALOG, {}) == "not configured"


def test_row_status_stale_when_model_missing_from_catalog() -> None:
    status = _pending_row_status(TierName.FAST, "ghost-model", "low", False, _CATALOG, {})
    assert status == "stale — model missing from catalog"


@pytest.mark.parametrize(
    "has_key, tier, model, thinking, keyed, expected",
    [
        (False, TierName.FAST, "deepseek-v4-flash", False, False, "no key"),
        (False, TierName.FAST, "deepseek-v4-flash", False, True, "✓ ready"),
        (True, TierName.CORE, "deepseek-v4-flash", False, False, "warning"),
        (True, TierName.FAST, "deepseek-v4-pro", False, False, "deprecated"),
        (True, TierName.CORE, "deepseek-v4-pro", True, False, "✓ ready"),
        (True, TierName.FAST, "deepseek-v4-flash", False, "clear", "no key"),
    ],
)
def test_pending_row_status(
    monkeypatch: pytest.MonkeyPatch,
    has_key: bool,
    tier: TierName,
    model: str,
    thinking: bool,
    keyed: bool | str,
    expected: str,
) -> None:
    # Regression covered by the last case: a real keyring credential exists,
    # but the key cell was edited to empty and confirmed — the row must show
    # "no key", not "✓ ready", until committed.
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: has_key)
    cred_key = _tier_credential_key(tier, model, "low", thinking)
    if keyed == "clear":
        key_input = {cred_key: ""}
    elif keyed:
        key_input = {cred_key: "sk-123"}
    else:
        key_input = {}
    status = _pending_row_status(tier, model, "low", thinking, _CATALOG, key_input)
    assert status == expected


# ---- _cycle_choice -------------------------------------------------------


@pytest.mark.parametrize(
    "options, current, expected",
    [
        (["a", "b", "c"], None, "a"),
        (["a", "b", "c"], "not-in-list", "a"),
        (["a", "b", "c"], "a", "b"),
        (["a", "b", "c"], "b", "c"),
        (["a", "b", "c"], "c", "a"),
        (("low", "medium", "high"), "medium", "high"),
    ],
)
def test_cycle_choice(options: object, current: object, expected: str) -> None:
    assert _cycle_choice(options, current) == expected


# ---- _tier_edit_complete --------------------------------------------------


@pytest.mark.parametrize(
    "model, key_present, expected",
    [
        (None, False, False),
        (None, True, False),
        ("deepseek-v4-flash", False, False),
        ("deepseek-v4-flash", True, True),
    ],
)
def test_tier_edit_complete(model: str | None, key_present: bool, expected: bool) -> None:
    assert _tier_edit_complete(model, key_present=key_present) is expected
