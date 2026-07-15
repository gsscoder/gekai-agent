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


def test_mask_key_keeps_head_and_tail_with_scaled_middle() -> None:
    assert _mask_key("sk-1234567890abcdef") == "sk" + "*" * 14 + "def"


def test_mask_key_masks_in_full_when_five_chars_or_fewer() -> None:
    assert _mask_key("sk123") == "*****"
    assert _mask_key("ab") == "**"


def test_mask_key_boundary_at_six_chars() -> None:
    assert _mask_key("sk-123") == "sk*123"


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
    assert _pending_row_status(TierName.FAST, None, False, _CATALOG, {}) == "not configured"


def test_row_status_stale_when_model_missing_from_catalog() -> None:
    status = _pending_row_status(TierName.FAST, "ghost-model", False, _CATALOG, {})
    assert status == "stale — model missing from catalog"


def test_row_status_no_key_when_credential_absent_and_not_staged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: False)
    status = _pending_row_status(TierName.FAST, "deepseek-v4-flash", False, _CATALOG, {})
    assert status == "no key"


def test_row_status_ready_when_key_is_only_staged_not_yet_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: False)
    status = _pending_row_status(
        TierName.FAST, "deepseek-v4-flash", False, _CATALOG, {"deepseek-v4-flash": "sk-123"}
    )
    assert status == "✓ ready"


def test_row_status_surfaces_warning_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    status = _pending_row_status(TierName.CORE, "deepseek-v4-flash", False, _CATALOG, {})
    assert status == "warning"


def test_row_status_surfaces_deprecated_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    status = _pending_row_status(TierName.FAST, "deepseek-v4-pro", False, _CATALOG, {})
    assert status == "deprecated"


def test_row_status_ready_on_ok_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    status = _pending_row_status(TierName.CORE, "deepseek-v4-pro", True, _CATALOG, {})
    assert status == "✓ ready"


def test_row_status_no_key_when_stored_credential_is_explicitly_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: a real keyring credential exists, but the key cell was
    # edited to empty and confirmed — the row must show "no key", not
    # "✓ ready", until committed.
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    status = _pending_row_status(TierName.FAST, "deepseek-v4-flash", False, _CATALOG, {"deepseek-v4-flash": ""})
    assert status == "no key"


# ---- _cycle_choice -------------------------------------------------------


def test_cycle_choice_starts_at_first_option_when_unset() -> None:
    assert _cycle_choice(["a", "b", "c"], None) == "a"


def test_cycle_choice_starts_at_first_option_when_stale() -> None:
    assert _cycle_choice(["a", "b", "c"], "not-in-list") == "a"


def test_cycle_choice_advances_to_next_option() -> None:
    assert _cycle_choice(["a", "b", "c"], "a") == "b"
    assert _cycle_choice(["a", "b", "c"], "b") == "c"


def test_cycle_choice_wraps_around_from_last_option() -> None:
    assert _cycle_choice(["a", "b", "c"], "c") == "a"


def test_cycle_choice_works_with_tuple_options() -> None:
    assert _cycle_choice(("low", "medium", "high"), "medium") == "high"


# ---- _tier_edit_complete --------------------------------------------------


def test_tier_edit_incomplete_when_no_model_selected() -> None:
    assert _tier_edit_complete(None, key_present=False) is False
    assert _tier_edit_complete(None, key_present=True) is False


def test_tier_edit_incomplete_when_model_set_but_no_key_anywhere() -> None:
    assert _tier_edit_complete("deepseek-v4-flash", key_present=False) is False


def test_tier_edit_complete_when_key_present() -> None:
    assert _tier_edit_complete("deepseek-v4-flash", key_present=True) is True
