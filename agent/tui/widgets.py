from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum

from rich.markup import escape as markup_escape
from textual import events
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Markdown, Static

from agent import __version_core__, __version_label__
from agent.diff import DiffLine, render_diff

# A model that ignores the "no separator lines" directive still renders one:
# Textual's Markdown widget treats a lone ---/***/___ line as a CommonMark
# thematic break (an actual <hr> block, with block-level margin above and
# below it — the "extra blank lines" is that margin, not stray whitespace).
# Directive compliance is best-effort; this is the enforcement backstop for
# assistant text specifically, applied right where that text enters the
# Markdown widget.
_THEMATIC_BREAK_RE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.MULTILINE)


def _strip_thematic_breaks(text: str) -> str:
    text = _THEMATIC_BREAK_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text)


class MessageKind(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    OPERATION = "operation"
    SYSTEM = "system"
    BANNER = "banner"
    HEADER = "header"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    WARNING = "warning"
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
                lines.append(f"[bold white]❯ {markup_escape(label)}[/bold white]")
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
    MessageWidget.assistant Markdown MarkdownBlock { margin: 0 0 1 0; }
    MessageWidget.assistant Markdown MarkdownFence > Label { padding-top: 0; padding-bottom: 0; }
    MessageWidget.assistant > .assistant-body { width: 1fr; height: auto; }
    MessageWidget.header { layout: horizontal; height: auto; }
    MessageWidget.header > .header-dot { width: 2; height: auto; }
    MessageWidget.header > .header-dot.nested { width: 4; height: auto; }
    MessageWidget.header > .header-text { width: 1fr; height: auto; }
    MessageWidget.interrupted { layout: horizontal; margin-top: 1; }
    MessageWidget.interrupted > Static { width: 2; height: auto; }
    MessageWidget.interrupted > .assistant-body { width: 1fr; height: auto; }
    MessageWidget.error { layout: horizontal; margin-top: 1; }
    MessageWidget.error > Static { width: 2; height: auto; }
    MessageWidget.error > .assistant-body { width: 1fr; height: auto; }
    MessageWidget.warning { layout: horizontal; margin-top: 1; }
    MessageWidget.warning > Static { width: 2; height: auto; }
    MessageWidget.warning > .assistant-body { width: 1fr; height: auto; }
    """

    def __init__(self, kind: MessageKind, text: str, color: str | None = None, nested: bool = False) -> None:
        self._kind = kind
        self._text = text
        self._color = color
        # HEADER-only: a nested (depth > 0) subagent header widens its dot
        # column to fit a "⎿ " connector ahead of the spinner/final dot — see
        # SubAgentRenderer._animate_dot/.done in agent/tui/app.py. Depth 0
        # (the default) renders exactly as before: no connector, width 2.
        self._nested = nested
        super().__init__(classes=kind.value)

    def compose(self) -> ComposeResult:
        if self._kind == MessageKind.ASSISTANT:
            yield Static("[white]●[/white]")
            if self._color:
                yield Static(markup_escape(self._text), classes="assistant-body")
            else:
                yield Markdown(_strip_thematic_breaks(self._text))
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
        elif self._kind == MessageKind.WARNING:
            yield Static("[yellow]●[/yellow]")
            yield Static(
                f"[yellow]Warning[/yellow]\n[#666666]⎿ {markup_escape(self._text)}[/#666666]",
                classes="assistant-body",
            )
        elif self._kind == MessageKind.COMMAND_RESULT:
            yield Static(self._as_markup())
        elif self._kind == MessageKind.HEADER:
            dot_classes = "header-dot nested" if self._nested else "header-dot"
            yield Static("[#666666]●[/#666666]", classes=dot_classes)
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
                return f"[grey70]{self._text}[/grey70]"
            case MessageKind.COMMAND_RESULT:
                if self._text:
                    return f"[#666666]⎿[/#666666] [#ffd700]{markup_escape(self._text)}[/#ffd700]"
                return "[dim]⎿ (no output)[/dim]"

    def update(self, content: str) -> None:
        self._text = content
        if self._kind == MessageKind.ASSISTANT:
            self.query_one(Markdown).update(_strip_thematic_breaks(content))
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
                lines.append(f"[bold white]❯ {ago_markup}  {escaped}[/bold white]")
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
                lines.append(f"[bold white]+ {escaped}[/bold white]")
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


@dataclass(frozen=True)
class ModelRowView:
    """Fully-rendered strings for one model row — the widget does no
    formatting/derivation of its own, purely displays what it's given."""

    provider: str  # e.g. "openai" — catalog metadata, not editable
    model: str  # e.g. "deepseek-v4-flash" — catalog metadata, not editable
    key: str  # masked value ("ab****xyz") if a key exists, "no key" if not, or the raw in-progress buffer while editing
    status: str  # e.g. "✓ keyed", "no key"


# The provider and model columns are catalog facts, so only the key cell is
# ever selectable on a model row — a single-entry tuple keeps the same
# cursor machinery the commit row uses without special-casing it.
_MODEL_ROW_COLUMNS = ("key",)
_COMMIT_ROW_COLUMNS = ("ok", "cancel")


class ModelsPanel(Widget):
    """Cursor + display only — mirrors ChoiceBar. No model/credential domain
    logic lives here; the caller (agent/tui/app.py) supplies fully-rendered
    ModelRowView instances via `show()` and interprets what pressing Enter on
    `selected_cell` means."""

    DEFAULT_CSS = """
    ModelsPanel {
        display: none;
        height: auto;
        background: ansi_default;
        border-top: solid #3a3a3a;
    }
    ModelsPanel #models-entries {
        height: auto;
        background: ansi_default;
        padding: 0 0 0 2;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._rows: list[ModelRowView] = []
        self._row: int = 0
        self._column: str = "key"
        self._shown_once: bool = False

    def compose(self) -> ComposeResult:
        yield Static("", id="models-entries")

    def show(self, rows: list[ModelRowView]) -> None:
        """One row per catalog model, in catalog order; the commit row is
        appended by the widget itself."""
        self._rows = rows
        if not self._shown_once:
            # Only reset the cursor on the very first show() — subsequent
            # calls (e.g. re-rendering after an edit) preserve position.
            self._row = 0
            self._column = "key"
            self._shown_once = True
        self._clamp_row()
        self._refresh_display()
        self.display = True

    def hide(self) -> None:
        self.display = False

    @property
    def _commit_row(self) -> int:
        return len(self._rows)

    # Row navigation clamps at the top/bottom rather than wrapping like
    # ChoiceBar's move_left/move_right does for its options — wrapping
    # top-to-bottom across the grid via up/down would be disorienting, so
    # this deliberately diverges from ChoiceBar here.
    def move_up(self) -> None:
        if self._row > 0:
            self._row -= 1
            self._clamp_column()
            self._refresh_display()

    def move_down(self) -> None:
        if self._row < self._commit_row:
            self._row += 1
            self._clamp_column()
            self._refresh_display()

    def move_left(self) -> None:
        columns = self._columns_for_row(self._row)
        idx = columns.index(self._column)
        if idx > 0:
            self._column = columns[idx - 1]
            self._refresh_display()

    def move_right(self) -> None:
        columns = self._columns_for_row(self._row)
        idx = columns.index(self._column)
        if idx < len(columns) - 1:
            self._column = columns[idx + 1]
            self._refresh_display()

    def _columns_for_row(self, row: int) -> tuple[str, ...]:
        return _COMMIT_ROW_COLUMNS if row == self._commit_row else _MODEL_ROW_COLUMNS

    def _clamp_row(self) -> None:
        # A shorter catalog than the one the cursor was last positioned
        # against would otherwise leave the cursor past the commit row.
        if self._row > self._commit_row:
            self._row = self._commit_row
        self._clamp_column()

    def _clamp_column(self) -> None:
        # Model rows and the commit row have disjoint column sets, so a
        # column selected on one side is never valid on the other; land on
        # the first column of whichever side move_up/move_down lands on.
        columns = self._columns_for_row(self._row)
        if self._column not in columns:
            self._column = columns[0]

    @property
    def selected_cell(self) -> tuple[int, str]:
        """(row, column) where row indexes the catalog models, or equals the
        model count for the commit row; column is "key" on a model row, or
        "ok"/"cancel" on the commit row. The provider, model, and status
        columns are NOT selectable — cursor navigation skips them."""
        return (self._row, self._column)

    def _refresh_display(self) -> None:
        if not self._rows:
            self.query_one("#models-entries", Static).update("")
            return

        provider_w = max(8, max(len(r.provider) for r in self._rows))
        model_w = max(5, max(len(r.model) for r in self._rows))
        key_w = max(3, max(len(r.key) for r in self._rows))

        def cell(row_i: int, column: str, text: str, width: int) -> str:
            escaped = markup_escape(text).ljust(width)
            if (row_i, column) == (self._row, self._column):
                return f"[bold white]❯ {escaped}[/bold white]"
            return f"  {escaped}"

        header = (
            "provider".ljust(provider_w)
            + "  " + "model".ljust(model_w)
            + "  " + "  key".ljust(2 + key_w)
            + "  status"
        )
        lines = [f"[dim]{header}[/dim]"]

        for i, row in enumerate(self._rows):
            status_escaped = markup_escape(row.status)
            status_markup = status_escaped if row.status.startswith("✓") else f"[dim]{status_escaped}[/dim]"
            line = (
                markup_escape(row.provider).ljust(provider_w)
                + "  " + markup_escape(row.model).ljust(model_w)
                + "  " + cell(i, "key", row.key, key_w)
                + "  " + status_markup
            )
            lines.append(line)

        def commit_cell(column: str, label: str) -> str:
            escaped = markup_escape(label)
            if (self._commit_row, column) == (self._row, self._column):
                return f"[bold #ffd700]{escaped}[/bold #ffd700]"
            return escaped

        lines.append(commit_cell("ok", "[ok]") + "  " + commit_cell("cancel", "[cancel]"))

        self.query_one("#models-entries", Static).update("\n".join(lines))


class DiffWidget(Widget):
    DEFAULT_CSS = """
    DiffWidget {
        height: auto;
        background: ansi_default;
        padding: 0 0 0 5;
    }
    DiffWidget > Static {
        height: auto;
        background: ansi_default;
    }
    """

    def __init__(self, path: str, diff_lines: list[DiffLine], **kwargs: object) -> None:
        self._path = path
        self._diff_lines = diff_lines
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static(render_diff(self._diff_lines))


_WB_W = 56
_WB_TITLE = "Ready to operate · Input prompt or command"
_wbt_pad = (_WB_W - len(_WB_TITLE)) // 2
_WB_TITLE_LINE = " " * _wbt_pad + _WB_TITLE + " " * (_WB_W - len(_WB_TITLE) - _wbt_pad)
_WB_L1, _WB_S1 = "Insert a new line in the input box", "ctrl+J"
_WB_SP1 = " " * (_WB_W - 2 - len(_WB_L1) - len(_WB_S1))
_WB_L2, _WB_S2 = "Recall prompt history", "ctrl+R"
_WB_SP2 = " " * (_WB_W - 2 - len(_WB_L2) - len(_WB_S2))
_WB_L3, _WB_S3 = "Scroll to bottom", "ctrl+B"
_WB_SP3 = " " * (_WB_W - 2 - len(_WB_L3) - len(_WB_S3))
_WB_L4, _WB_S4 = "Exit the application", "ctrl+Q"
_WB_SP4 = " " * (_WB_W - 2 - len(_WB_L4) - len(_WB_S4))
_WELCOME_CONTENT = (
    f"[dim]{_WB_TITLE_LINE}[/dim]\n"
    f"\n"
    f"[#4a4a4a]\\[[/][dim]{_WB_L1}[/dim]{_WB_SP1}[#888888]{_WB_S1}[/#888888][#4a4a4a]][/]\n"
    f"[#4a4a4a]\\[[/][dim]{_WB_L2}[/dim]{_WB_SP2}[#888888]{_WB_S2}[/#888888][#4a4a4a]][/]\n"
    f"[#4a4a4a]\\[[/][dim]{_WB_L3}[/dim]{_WB_SP3}[#888888]{_WB_S3}[/#888888][#4a4a4a]][/]\n"
    f"[#4a4a4a]\\[[/][dim]{_WB_L4}[/dim]{_WB_SP4}[#888888]{_WB_S4}[/#888888][#4a4a4a]][/]"
)


class WelcomeOverlay(Widget):
    can_focus = False

    DEFAULT_CSS = """
    WelcomeOverlay {
        layer: overlay;
        width: 62;
        height: 10;
        background: ansi_default;
        border: round #4a4a4a;
        border-title-align: center;
        padding: 1 2;
    }
    #welcome-box {
        width: 100%;
        height: 100%;
        background: ansi_default;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(_WELCOME_CONTENT, id="welcome-box")

    def on_mount(self) -> None:
        self.border_title = f"[bold white]gekAI[/bold white] [#4a4a4a]| {__version_core__} {__version_label__}[/#4a4a4a]"
