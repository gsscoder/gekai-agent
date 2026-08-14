"""Coverage for `ModelsPanel`'s cursor-movement state machine.

`ModelsPanel` is a plain Textual `Widget` (see agent/tui/widgets.py); this
repo has no Pilot-based UI test precedent, and `compose()` only runs once a
widget is mounted under a live `App`. Every mutator (`show`, `move_up`,
`move_down`, `move_left`, `move_right`) calls `_refresh_display()` after
changing state, which does `query_one("#models-entries", Static)` and raises
`NoMatches` on an unmounted instance. `selected_cell` and `hide()` don't
touch the DOM and can be tested directly; everything else here uses
`_panel()` below, which stubs out `_refresh_display` on the instance to
bypass the mount requirement while still exercising the real
cursor-movement/`show()` logic.
"""

from __future__ import annotations

from agent.tui.widgets import ModelRowView, ModelsPanel

_ROWS = [
    ModelRowView("openai", "deepseek-v4-flash", "****", "✓ keyed"),
    ModelRowView("openai", "deepseek-v4-pro", "no key", "no key"),
    ModelRowView("openai", "qwen3.8-max", "****", "✓ keyed"),
]
_COMMIT_ROW = len(_ROWS)


def _panel(rows: list[ModelRowView] | None = None) -> ModelsPanel:
    panel = ModelsPanel()
    panel._refresh_display = lambda: None  # bypass Static-child mount requirement
    panel.show(_ROWS if rows is None else rows)
    return panel


def test_cursor_starts_at_key_column_of_first_row() -> None:
    panel = ModelsPanel()
    assert panel.selected_cell == (0, "key")


def test_left_right_do_nothing_on_a_model_row() -> None:
    # "key" is the only selectable column on a model row — provider and
    # model are catalog facts, not choices.
    panel = _panel()
    for _ in range(5):
        panel.move_right()
    assert panel.selected_cell == (0, "key")

    for _ in range(5):
        panel.move_left()
    assert panel.selected_cell == (0, "key")


def test_up_down_clamps_across_model_rows_and_the_commit_row() -> None:
    panel = _panel()
    panel.move_up()
    assert panel.selected_cell == (0, "key")  # already at top, stays

    for _ in range(8):
        panel.move_down()
    assert panel.selected_cell[0] == _COMMIT_ROW  # clamped at the commit row

    for _ in range(8):
        panel.move_up()
    assert panel.selected_cell[0] == 0  # clamped back at the top


def test_left_right_on_commit_row_only_toggles_ok_cancel() -> None:
    panel = _panel()
    for _ in range(_COMMIT_ROW):
        panel.move_down()
    assert panel.selected_cell == (_COMMIT_ROW, "ok")

    panel.move_right()
    assert panel.selected_cell == (_COMMIT_ROW, "cancel")
    panel.move_right()
    assert panel.selected_cell == (_COMMIT_ROW, "cancel")  # clamped, no wraparound

    panel.move_left()
    assert panel.selected_cell == (_COMMIT_ROW, "ok")
    panel.move_left()
    assert panel.selected_cell == (_COMMIT_ROW, "ok")  # clamped, no wraparound


def test_column_resets_when_crossing_between_model_and_commit_rows() -> None:
    panel = _panel()
    for _ in range(_COMMIT_ROW):
        panel.move_down()
    panel.move_right()
    assert panel.selected_cell == (_COMMIT_ROW, "cancel")

    panel.move_up()
    assert panel.selected_cell == (_COMMIT_ROW - 1, "key")  # "cancel" is invalid on a model row


def test_commit_row_index_tracks_the_catalog_size() -> None:
    # The commit row is not a fixed index — a one-model catalog puts it at 1.
    panel = _panel([_ROWS[0]])
    panel.move_down()
    assert panel.selected_cell == (1, "ok")
    panel.move_down()
    assert panel.selected_cell == (1, "ok")  # clamped


def test_show_with_a_shorter_catalog_clamps_a_past_the_end_cursor() -> None:
    panel = _panel()
    for _ in range(_COMMIT_ROW):
        panel.move_down()
    assert panel.selected_cell == (_COMMIT_ROW, "ok")

    panel.show([_ROWS[0]])
    assert panel.selected_cell == (1, "ok")  # the new commit row, not row 3


def test_hide_does_not_require_mount() -> None:
    panel = ModelsPanel()
    panel.hide()
    assert panel.display is False


def test_show_resets_cursor_only_on_first_call() -> None:
    panel = ModelsPanel()
    panel._refresh_display = lambda: None

    panel.move_down()
    panel.move_down()
    assert panel.selected_cell == (0, "key")  # no rows yet: commit row is 0, so move_down clamps

    panel.show(_ROWS)
    assert panel.selected_cell == (0, "key")  # first show() resets

    panel.move_down()
    assert panel.selected_cell == (1, "key")

    panel.show(_ROWS)
    assert panel.selected_cell == (1, "key")  # later show() preserves position
