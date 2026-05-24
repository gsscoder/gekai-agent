from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Static

from agent.settings import PERMISSION_CHOICES


class PermissionScreen(ModalScreen[str | None]):
    DEFAULT_CSS = """
    PermissionScreen {
        align: center middle;
    }

    #perm-panel {
        width: 70;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    #perm-title {
        text-style: bold;
        height: 1;
        padding-bottom: 1;
    }

    .perm-choice {
        height: 1;
    }
    """

    BINDINGS = [
        ("up", "up", "Up"),
        ("down", "down", "Down"),
        ("enter", "select", "Select"),
        ("escape", "cancel", "Cancel"),
    ]

    can_focus = True
    selected: reactive[int] = reactive(0, init=False)

    def compose(self) -> ComposeResult:
        with Vertical(id="perm-panel"):
            yield Static("Gekai needs access to this workspace", id="perm-title")
            for i, (_, label) in enumerate(PERMISSION_CHOICES):
                yield Static(self._label_text(i, label), id=f"choice-{i}", classes="perm-choice")

    def on_mount(self) -> None:
        self.focus()

    def watch_selected(self) -> None:
        if not self.is_mounted:
            return
        for i, (_, label) in enumerate(PERMISSION_CHOICES):
            self.query_one(f"#choice-{i}", Static).update(self._label_text(i, label))

    def _label_text(self, i: int, label: str) -> Text:
        if i == self.selected:
            return Text(f"❯ {label}", style="bold cyan")
        return Text(f"  {label}", style="bright_black")

    def action_up(self) -> None:
        self.selected = (self.selected - 1) % len(PERMISSION_CHOICES)

    def action_down(self) -> None:
        self.selected = (self.selected + 1) % len(PERMISSION_CHOICES)

    def action_select(self) -> None:
        self.dismiss(PERMISSION_CHOICES[self.selected][0])

    def action_cancel(self) -> None:
        self.dismiss(None)
