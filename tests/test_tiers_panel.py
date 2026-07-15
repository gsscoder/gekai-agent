"""Coverage for `TiersPanel`'s cursor-movement state machine.

`TiersPanel` is a plain Textual `Widget` (see agent/tui/widgets.py); this repo
has no Pilot-based UI test precedent (see tests/test_tiers_flow.py), and
`compose()` only runs once a widget is mounted under a live `App`. Every
mutator (`show`, `move_up`, `move_down`, `move_left`, `move_right`) calls
`_refresh_display()` after changing state, which does
`query_one("#tiers-entries", Static)` and raises `NoMatches` on an unmounted
instance. `selected_cell` and `hide()` don't touch the DOM and can be tested
directly; everything else here uses `_panel()` below, which stubs out
`_refresh_display` on the instance to bypass the mount requirement while
still exercising the real cursor-movement/`show()` logic.
"""

from __future__ import annotations

from agent.tui.widgets import TierRowView, TiersPanel

_ROWS = [
    TierRowView("FAST", "deepseek-v4-flash", "medium", "n/a", "****", "✓ ready"),
    TierRowView("SUPP", "deepseek-v4-flash", "high", "n/a", "****", "⚠ no key"),
    TierRowView("CORE", "deepseek-v4-pro", "high", "yes", "****", "✓ ready"),
]


def _panel() -> TiersPanel:
    panel = TiersPanel()
    panel._refresh_display = lambda: None  # bypass Static-child mount requirement
    return panel


def test_cursor_starts_at_model_column_of_first_row() -> None:
    panel = TiersPanel()
    assert panel.selected_cell == (0, "model")


def test_left_right_clamps_within_tier_row_columns() -> None:
    panel = _panel()
    for _ in range(5):
        panel.move_right()
    assert panel.selected_cell == (0, "key")  # clamped, no wraparound

    for _ in range(5):
        panel.move_left()
    assert panel.selected_cell == (0, "model")  # clamped back to start


def test_up_down_clamps_across_all_four_rows() -> None:
    panel = _panel()
    panel.move_up()
    assert panel.selected_cell == (0, "model")  # already at top, stays

    for _ in range(5):
        panel.move_down()
    assert panel.selected_cell[0] == 3  # clamped at the commit row

    for _ in range(5):
        panel.move_up()
    assert panel.selected_cell[0] == 0  # clamped back at the top


def test_left_right_on_commit_row_only_toggles_ok_cancel() -> None:
    panel = _panel()
    panel.move_down()
    panel.move_down()
    panel.move_down()
    assert panel.selected_cell == (3, "ok")

    panel.move_right()
    assert panel.selected_cell == (3, "cancel")
    panel.move_right()
    assert panel.selected_cell == (3, "cancel")  # clamped, no wraparound

    panel.move_left()
    assert panel.selected_cell == (3, "ok")
    panel.move_left()
    assert panel.selected_cell == (3, "ok")  # clamped, no wraparound


def test_column_resets_when_crossing_between_tier_and_commit_rows() -> None:
    panel = _panel()
    for _ in range(3):
        panel.move_right()
    assert panel.selected_cell == (0, "key")

    panel.move_down()
    panel.move_down()
    panel.move_down()
    assert panel.selected_cell == (3, "ok")  # "key" is invalid on the commit row

    panel.move_right()
    assert panel.selected_cell == (3, "cancel")

    panel.move_up()
    assert panel.selected_cell == (2, "model")  # "cancel" is invalid on a tier row


def test_hide_does_not_require_mount() -> None:
    panel = TiersPanel()
    panel.hide()
    assert panel.display is False


def test_show_resets_cursor_only_on_first_call() -> None:
    panel = _panel()

    panel.move_right()
    panel.move_right()
    assert panel.selected_cell == (0, "thinking")

    panel.show(_ROWS)
    assert panel.selected_cell == (0, "model")  # first show() resets

    panel.move_down()
    panel.move_right()
    assert panel.selected_cell == (1, "effort")

    panel.show(_ROWS)
    assert panel.selected_cell == (1, "effort")  # later show() preserves position
