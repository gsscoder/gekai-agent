from __future__ import annotations

from enum import Enum

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
            yield Static("[cyan]●[/cyan]")
            yield Markdown(self._text)
        elif self._kind == MessageKind.HEADER:
            yield Static("[cyan]●[/cyan]", classes="header-dot")
            yield Static(self._text, classes="header-text")
        elif self._kind == MessageKind.USER:
            yield Static(self._as_user_text(), markup=False)
        else:
            yield Static(self._as_markup())

    def _as_user_text(self) -> str:
        return f"❯ {self._text}"

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
