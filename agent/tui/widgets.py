from __future__ import annotations

import time
from enum import Enum

from rich.markup import escape as markup_escape
from textual import events
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Markdown, Static


class MessageKind(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    OPERATION = "operation"
    SYSTEM = "system"
    BANNER = "banner"
    HEADER = "header"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    REJECTED = "rejected"
    COMMAND_RESULT = "command_result"


class ChoiceBar(Static):
    DEFAULT_CSS = """
    ChoiceBar {
        height: auto;
        background: ansi_default;
        color: orange;
        padding: 0 0 0 2;
        border-top: solid #3a3a3a;
        display: none;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._question: str = ""
        self._options: list[tuple[str, str]] = []
        self._selected: int = 0

    def show(self, question: str, options: list[tuple[str, str]], default_index: int = 0) -> None:
        self._question = question
        self._options = options
        self._selected = default_index
        self._refresh_display()
        self.display = True

    def hide(self) -> None:
        self.display = False
        self._question = ""
        self._options = []
        self._selected = 0
        self.update("")

    def move_left(self) -> None:
        if self._options:
            self._selected = (self._selected - 1) % len(self._options)
            self._refresh_display()

    def move_right(self) -> None:
        if self._options:
            self._selected = (self._selected + 1) % len(self._options)
            self._refresh_display()

    @property
    def selected_key(self) -> str | None:
        if not self._options:
            return None
        return self._options[self._selected][0]

    def _refresh_display(self) -> None:
        lines: list[str] = [markup_escape(self._question)]
        for i, (_, label) in enumerate(self._options):
            if i == self._selected:
                lines.append(f"[bold cyan]❯ {markup_escape(label)}[/bold cyan]")
            else:
                lines.append(f"  {markup_escape(label)}")
        self.update("\n".join(lines))


class MessageWidget(Widget):
    DEFAULT_CSS = """
    MessageWidget { height: auto; background: ansi_default; }
    MessageWidget > Static { background: ansi_default; }
    MessageWidget > Markdown { background: ansi_default; }
    MessageWidget.user { margin-top: 1; }
    MessageWidget.user > Static { background: #3a3a3a; color: white; }
    MessageWidget.assistant { layout: horizontal; margin-top: 1; }
    MessageWidget.assistant > Static { width: 2; height: auto; }
    MessageWidget.assistant > Markdown { width: 1fr; height: auto; padding: 0; }
    MessageWidget.assistant > .assistant-body { width: 1fr; height: auto; }
    MessageWidget.header { layout: horizontal; height: auto; }
    MessageWidget.header > .header-dot { width: 2; height: auto; }
    MessageWidget.header > .header-text { width: 1fr; height: auto; }
    MessageWidget.interrupted { layout: horizontal; margin-top: 1; }
    MessageWidget.interrupted > Static { width: 2; height: auto; }
    MessageWidget.interrupted > .assistant-body { width: 1fr; height: auto; }
    MessageWidget.error { layout: horizontal; margin-top: 1; }
    MessageWidget.error > Static { width: 2; height: auto; }
    MessageWidget.error > .assistant-body { width: 1fr; height: auto; }
    MessageWidget.rejected { margin-top: 1; }
    """

    def __init__(self, kind: MessageKind, text: str, color: str | None = None) -> None:
        self._kind = kind
        self._text = text
        self._color = color
        super().__init__(classes=kind.value)

    def compose(self) -> ComposeResult:
        if self._kind == MessageKind.ASSISTANT:
            dot_color = self._color or "cyan"
            yield Static(f"[{dot_color}]●[/{dot_color}]")
            if self._color:
                yield Static(markup_escape(self._text), classes="assistant-body")
            else:
                yield Markdown(self._text)
        elif self._kind == MessageKind.INTERRUPTED:
            if self._text:
                yield Static("[red]●[/red]")
                yield Static(self._text, classes="assistant-body")
            else:
                yield Static("[#666666]●[/#666666]")
                yield Static(
                    "[white]Interrupted[/white]\n[#666666]⎿ How should Gekai proceed instead?[/#666666]",
                    classes="assistant-body",
                )
        elif self._kind == MessageKind.ERROR:
            yield Static("[red]●[/red]")
            yield Static(
                f"[white]Error[/white]\n[#666666]⎿ {markup_escape(self._text)}[/#666666]",
                classes="assistant-body",
            )
        elif self._kind == MessageKind.REJECTED:
            yield Static(f"[red]●[/red] [white]Rejected[/white] — [#666666]{markup_escape(self._text)}[/#666666]")
        elif self._kind == MessageKind.COMMAND_RESULT:
            yield Static(self._as_markup())
        elif self._kind == MessageKind.HEADER:
            yield Static("[cyan]●[/cyan]", classes="header-dot")
            yield Static(self._text, classes="header-text")
        elif self._kind == MessageKind.USER:
            yield Static(self._as_user_text())
        else:
            yield Static(self._as_markup())

    def on_mount(self) -> None:
        if self._kind == MessageKind.ASSISTANT and self._color:
            self.query_one(".assistant-body", Static).styles.color = self._color

    def _as_user_text(self) -> str:
        return f"❯ [bold]{markup_escape(self._text)}[/bold]"

    @property
    def text(self) -> str:
        return self._text

    def _as_markup(self) -> str:
        match self._kind:
            case MessageKind.OPERATION:
                color = self._color or "grey50"
                return f"[{color}]{self._text}[/{color}]"
            case MessageKind.SYSTEM:
                return f"[dim]{self._text}[/dim]"
            case MessageKind.BANNER:
                return f"[cyan]{self._text}[/cyan]"
            case MessageKind.COMMAND_RESULT:
                if self._text:
                    return f"[#666666]⎿[/#666666] [#ffd700]{markup_escape(self._text)}[/#ffd700]"
                return "[dim]⎿ (no output)[/dim]"

    def update(self, content: str) -> None:
        self._text = content
        if self._kind == MessageKind.ASSISTANT:
            self.query_one(Markdown).update(content)
        elif self._kind == MessageKind.USER:
            self.query_one(Static).update(self._as_user_text())
        elif self._kind == MessageKind.HEADER:
            self.query_one(".header-text", Static).update(self._text)
        else:
            self.query_one(Static).update(self._as_markup())

    def append_text(self, chunk: str) -> None:
        self._text += chunk
        if self._kind == MessageKind.ASSISTANT:
            self.query_one(Markdown).update(self._text)
        elif self._kind == MessageKind.USER:
            self.query_one(Static).update(self._as_user_text())
        elif self._kind == MessageKind.HEADER:
            self.query_one(".header-text", Static).update(self._text)
        else:
            self.query_one(Static).update(self._as_markup())


def _fmt_ago(ts: float) -> str:
    delta = time.time() - ts
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


class HistoryPanel(Widget):
    DEFAULT_CSS = """
    HistoryPanel {
        display: none;
        height: 5;
        background: ansi_default;
        border-top: solid #3a3a3a;
    }
    HistoryPanel #history-entries {
        height: 1fr;
        background: ansi_default;
        padding: 0 0 0 2;
    }
    """

    class RowClicked(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    _MAX_ENTRIES = 5
    _MAX_TEXT_LEN = 60

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict] = []
        self._selected: int = 0

    def compose(self) -> ComposeResult:
        yield Static("", id="history-entries")

    def show(self, entries: list[dict], selected_index: int = 0) -> None:
        self._entries = entries
        self._selected = selected_index
        self._refresh_display()
        self.display = True

    def hide(self) -> None:
        self.display = False
        self._entries = []
        self._selected = 0

    def move_up(self) -> None:
        if self._entries:
            self._selected = max(0, self._selected - 1)
            self._refresh_display()

    def move_down(self) -> None:
        if self._entries:
            self._selected = min(len(self._entries) - 1, self._selected + 1)
            self._refresh_display()

    def select_index(self, i: int) -> None:
        if 0 <= i < len(self._entries):
            self._selected = i
            self._refresh_display()

    @property
    def selected_text(self) -> str | None:
        if not self._entries:
            return None
        if 0 <= self._selected < len(self._entries):
            return self._entries[self._selected].get("text")
        return None

    @property
    def _window_start(self) -> int:
        n = len(self._entries)
        if n <= self._MAX_ENTRIES:
            return 0
        start = self._selected - self._MAX_ENTRIES // 2
        return max(0, min(start, n - self._MAX_ENTRIES))

    def _refresh_display(self) -> None:
        if not self._entries:
            self.query_one("#history-entries", Static).update("[dim]  no history[/dim]")
            return
        start = self._window_start
        visible = self._entries[start : start + self._MAX_ENTRIES]
        lines = []
        for i, entry in enumerate(visible):
            abs_i = start + i
            ts = entry.get("timestamp", 0.0)
            text = entry.get("text", "")
            ago = _fmt_ago(ts)
            truncated = text if len(text) <= self._MAX_TEXT_LEN else text[: self._MAX_TEXT_LEN] + "…"
            escaped = markup_escape(truncated)
            ago_markup = f"[dim]{ago:>6}[/dim]"
            if abs_i == self._selected:
                lines.append(f"[bold cyan]❯ {ago_markup}  {escaped}[/bold cyan]")
            else:
                lines.append(f"[dim]  {ago_markup}  {escaped}[/dim]")
        self.query_one("#history-entries", Static).update("\n".join(lines))

    def on_click(self, event: events.Click) -> None:
        entries_widget = self.query_one("#history-entries", Static)
        rel_y = event.y - entries_widget.region.y
        start = self._window_start
        visible_count = min(self._MAX_ENTRIES, len(self._entries) - start)
        if 0 <= rel_y < visible_count:
            self.post_message(self.RowClicked(index=start + rel_y))
            event.stop()


class FilePanel(Widget):
    DEFAULT_CSS = """
    FilePanel {
        display: none;
        height: 5;
        background: ansi_default;
        border-top: solid #3a3a3a;
    }
    FilePanel #file-entries {
        height: 1fr;
        background: ansi_default;
        padding: 0 0 0 2;
    }
    """

    class RowClicked(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    _MAX_ENTRIES = 5
    _MAX_PATH_LEN = 60

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._all_entries: list[str] = []
        self._filtered: list[str] = []
        self._selected: int = 0

    def compose(self) -> ComposeResult:
        yield Static("", id="file-entries")

    def show(self, paths: list[str], query: str = "") -> None:
        self._all_entries = paths
        self._apply_filter(query)
        self._refresh_display()
        self.display = True

    def hide(self) -> None:
        self.display = False
        self._all_entries = []
        self._filtered = []
        self._selected = 0

    def filter(self, query: str) -> None:
        self._apply_filter(query)
        self._refresh_display()

    def _apply_filter(self, query: str) -> None:
        q = query.lower()
        if q:
            self._filtered = [p for p in self._all_entries if q in p.lower()]
        else:
            self._filtered = list(self._all_entries)
        self._selected = 0

    def move_up(self) -> None:
        if self._filtered:
            self._selected = max(0, self._selected - 1)
            self._refresh_display()

    def move_down(self) -> None:
        if self._filtered:
            self._selected = min(len(self._filtered) - 1, self._selected + 1)
            self._refresh_display()

    def select_index(self, i: int) -> None:
        if 0 <= i < len(self._filtered):
            self._selected = i
            self._refresh_display()

    @property
    def selected_text(self) -> str | None:
        if not self._filtered:
            return None
        if 0 <= self._selected < len(self._filtered):
            return self._filtered[self._selected]
        return None

    @property
    def _window_start(self) -> int:
        n = len(self._filtered)
        if n <= self._MAX_ENTRIES:
            return 0
        start = self._selected - self._MAX_ENTRIES // 2
        return max(0, min(start, n - self._MAX_ENTRIES))

    def _refresh_display(self) -> None:
        if not self._filtered:
            self.query_one("#file-entries", Static).update("[dim]  no matches[/dim]")
            return
        start = self._window_start
        visible = self._filtered[start : start + self._MAX_ENTRIES]
        lines = []
        for i, path in enumerate(visible):
            abs_i = start + i
            truncated = path if len(path) <= self._MAX_PATH_LEN else "…" + path[-(self._MAX_PATH_LEN - 1):]
            escaped = markup_escape(truncated)
            if abs_i == self._selected:
                lines.append(f"[bold cyan]+ {escaped}[/bold cyan]")
            else:
                lines.append(f"[dim]  {escaped}[/dim]")
        self.query_one("#file-entries", Static).update("\n".join(lines))

    def on_click(self, event: events.Click) -> None:
        entries_widget = self.query_one("#file-entries", Static)
        rel_y = event.y - entries_widget.region.y
        start = self._window_start
        visible_count = min(self._MAX_ENTRIES, len(self._filtered) - start)
        if 0 <= rel_y < visible_count:
            self.post_message(self.RowClicked(index=start + rel_y))
            event.stop()
