from __future__ import annotations

from enum import Enum

from rich.markup import escape as markup_escape
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Markdown, Static


class MessageKind(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    OPERATION = "operation"
    SYSTEM = "system"
    BANNER = "banner"
    HEADER = "header"


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
