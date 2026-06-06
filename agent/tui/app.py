from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pyfiglet
from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.geometry import Region
from textual.message import Message
from textual.strip import Strip
from textual.widgets import ProgressBar, Static, TextArea
from textual.worker import Worker

from agent import __version_core__, __version_label__
from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.persistence import append_command, append_debug, append_event, append_message, now_utc_str, _normalize_path
from agent.router import Intent, Route, Session
from agent.settings import PERMISSION_CHOICES, load_blast_radius_limit, load_context_limit, load_scope_gate, load_ws_scan_staleness_min, resolve_permissions, save_permissions
from agent.workspace import list_files
from agent.ws_manager.enrichment import _get_git_state
from agent.ui import random_accent_color, random_farewell, random_operative_verb
from agent.subagent import SubAgentEvent, SubAgentStartEvent, LogEvent, DiffEvent, InferEndEvent, DoneEvent, MaxIterationsEvent, StatusUpdateEvent, ThinkingTokenEvent

from .palette import CommandPalette
from .history import PromptHistory
from .widgets import ChoiceBar, DiffWidget, FilePanel, HistoryPanel, MessageKind, MessageWidget

_NS_COLORS: dict[str, str] = {
    "coding": "#FFD700",
    "management": "#00CED1",
}
_DEFAULT_ROUTE_COLOR = "#3a3a3a"
_ACTION_COLOR = "#4169E1"


class ConversationContainer(ScrollableContainer):
    class Scrolled(Message):
        def __init__(self, at_end: bool) -> None:
            super().__init__()
            self.at_end = at_end

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        at_end = new_value >= self.max_scroll_y or self.max_scroll_y <= 0
        self.post_message(self.Scrolled(at_end=at_end))


_SPINNER_FRAMES = ["·", "•", "●", "•"]
_BRAILLE_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _update_scan_state_key(cache_path: Path, key: str, value: str) -> None:
    try:
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    scan_state = existing.get("scan_state", {})
    scan_state[key] = value
    existing["scan_state"] = scan_state
    try:
        cache_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except OSError:
        pass


class SubAgentRenderer:
    """Manages header, L-connector, token accumulation, and Done line for subagent events."""

    def __init__(self, conversation: ScrollableContainer, debug: bool = False) -> None:
        self._conversation = conversation
        self._first_item = True
        self.name: str = ""
        self._total_tokens: int = 0
        self._infer_count: int = 0
        self._start_time: float = time.monotonic()
        self._debug = debug
        self._current_tool: str = ""
        self._current_count: int = 0
        self._current_widget: Static | None = None
        self._current_prefix: str = ""
        self._log_widgets: list[Static] = []
        self._progress_bar: ProgressBar | None = None
        self._header_widget: MessageWidget | None = None
        self._spinner_task: asyncio.Task | None = None
        self._thinking_buf: str = ""

    async def _animate_dot(self) -> None:
        frame = 0
        try:
            while True:
                if self._header_widget is not None:
                    char = _BRAILLE_FRAMES[frame % len(_BRAILLE_FRAMES)]
                    self._header_widget.query_one(".header-dot", Static).update(f"[#666666]{char}[/#666666]")
                frame += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    async def start(self, name: str, description: str, color: str) -> None:
        self.name = name
        await self._conversation.mount(Static("", classes="assistant-spacer"))
        header_markup = "[bold #666666]Thinking...[/bold #666666]"
        widget = MessageWidget(MessageKind.HEADER, header_markup)
        await self._conversation.mount(widget)
        self._header_widget = widget
        self._spinner_task = asyncio.create_task(self._animate_dot())

    async def log(self, message: str, tool_name: str = "") -> None:
        if message.endswith("..."):
            return
        if not self._debug and tool_name:
            if tool_name == self._current_tool and self._current_widget is not None:
                self._current_count += 1
                kind = message.split()[0] if message else tool_name
                self._current_widget.update(f"{self._current_prefix} {kind} ({self._current_count} calls)")
                self._conversation.scroll_end(animate=False)
                return
            self._current_tool = tool_name
            self._current_count = 1
        prefix = "  ⎿" if self._first_item else "   "
        self._first_item = False
        if not self._debug and tool_name:
            kind = message.split()[0] if message else tool_name
            widget = Static(f"{prefix} {kind} (1 call)")
            await self._conversation.mount(widget)
            self._log_widgets.append(widget)
            self._current_widget = widget
            self._current_prefix = prefix
        else:
            widget = Static(f"{prefix} {message}")
            await self._conversation.mount(widget)
            self._log_widgets.append(widget)
            self._current_widget = None
        self._conversation.scroll_end(animate=False)

    async def status_update(self, event: "StatusUpdateEvent") -> None:
        if self._progress_bar is None:
            bar = ProgressBar(total=event.total, show_eta=False, show_percentage=True, classes="subagent-progress")
            await self._conversation.mount(bar)
            self._progress_bar = bar
            self._conversation.scroll_end(animate=False)
        elif event.total is not None:
            self._progress_bar.update(total=event.total, progress=event.progress)

    def thinking_chunk(self, text: str) -> None:
        self._thinking_buf += text
        snippet = _last_sentence(self._thinking_buf)
        if snippet and self._header_widget is not None:
            self._header_widget.query_one(".header-text", Static).update(
                f"[bold #666666]Thinking({snippet})[/bold #666666]"
            )

    def stop_spinner(self) -> None:
        if self._spinner_task is not None:
            self._spinner_task.cancel()
            self._spinner_task = None

    def accumulate_tokens(self, event: "InferEndEvent") -> None:
        if event.prompt_tokens:
            self._total_tokens += event.prompt_tokens
        if event.completion_tokens:
            self._total_tokens += event.completion_tokens
        self._infer_count += 1

    async def done(self, thinking_chars: int = 0) -> None:
        if self._spinner_task is not None:
            self._spinner_task.cancel()
            self._spinner_task = None
        for w in self._log_widgets:
            await w.remove()
        self._log_widgets.clear()
        if self._progress_bar is not None:
            await self._progress_bar.remove()
            self._progress_bar = None
        elapsed = time.monotonic() - self._start_time
        parts: list[str] = []
        if self._total_tokens > 0:
            parts.append(f"{_fmt_tokens(self._total_tokens)} tokens")
        parts.append(_fmt_duration_verbose(elapsed))
        if self._infer_count > 0:
            calls = f"{self._infer_count} call" + ("s" if self._infer_count != 1 else "")
            parts.append(calls)
        summary = " · ".join(parts)
        if self._header_widget is not None:
            self._header_widget.query_one(".header-dot", Static).update("[#666666]●[/#666666]")
            self._header_widget.query_one(".header-text", Static).update(f"[#666666]Thought ({summary})[/#666666]")
            self._header_widget = None
        self._conversation.scroll_end(animate=False)


def _last_sentence(text: str, max_chars: int = 60) -> str:
    last = -1
    for ch in ".!?\n":
        idx = text.rfind(ch)
        if idx > last:
            last = idx
    fragment = text[last + 1:].strip() if last >= 0 else text.strip()
    if not fragment and last >= 0:
        fragment = text[:last + 1].strip()
    if not fragment:
        return ""
    return (fragment[:max_chars].rstrip() + "...") if len(fragment) > max_chars else fragment


def _resolve_at_refs(text: str) -> str:
    return re.sub(r"@(\S+)", lambda m: f"`{m.group(1)}`", text)


def _fmt_duration(elapsed: float) -> str:
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    return f"{elapsed / 60:.1f}m"


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _fmt_duration_verbose(elapsed: float) -> str:
    if elapsed < 1:
        return f"{elapsed * 1000:.0f}ms"
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"


def _fmt_elapsed(elapsed: float) -> str:
    secs = int(elapsed)
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"


def _fmt_status(verb: str, elapsed: float) -> str:
    return f"{verb.capitalize()}... [white]({_fmt_elapsed(elapsed)})[/white]"


_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "claude": 200_000,
    "gemini-1.5": 1_048_576,
    "gemini-2": 1_048_576,
    "deepseek-chat": 128_000,
}


def _context_limit(model: str) -> int:
    lower = model.lower()
    for key, limit in _CONTEXT_LIMITS.items():
        if key in lower:
            return limit
    return 128_000


def _fmt_context_pct(prompt_tokens: int, limit: int) -> str:
    pct = round(prompt_tokens / limit * 100, 1)
    return f"{pct}% context"


def _fmt_status_bar(model: str, working_dir: str, branch: str | None, prompt_tokens: int, limit: int) -> str:
    pct = _fmt_context_pct(prompt_tokens, limit)
    location = f"📁 {working_dir}"
    if branch:
        location += f" [⎇ {branch}]"
    return f"[dim]\\[{model}][/dim] | {location} | [dim]{pct}[/dim]"


def _estimate_session_tokens(session: Session) -> int:
    # Includes transcript + persistent system messages ([artifact], <lang>).
    # Artifacts from prior Query turns are what make this number grow meaningfully.
    return sum(len(str(m.get("content") or "")) for m in session.messages) // 4


def _strip_default_bg(style: Style | None) -> Style | None:
    if style is None or style.bgcolor is None or not style.bgcolor.is_default:
        return style
    return Style(
        color=style.color,
        bold=style.bold,
        dim=style.dim,
        italic=style.italic,
        underline=style.underline,
        blink=style.blink,
        blink2=style.blink2,
        reverse=style.reverse,
        conceal=style.conceal,
        strike=style.strike,
        underline2=style.underline2,
        frame=style.frame,
        encircle=style.encircle,
        overline=style.overline,
        link=style.link,
    )


class PromptTextArea(TextArea):
    """TextArea that renders without explicit default background, preserving terminal transparency."""

    def on_mount(self) -> None:
        # Non-blinking, always-visible block cursor (the old Input look).
        self.cursor_blink = False

    def get_component_rich_style(self, *names: str, partial: bool = False, default: Style | None = None) -> Style:
        # The cursor is painted via theme.cursor_style, which apply_css derives
        # from this component style each render. Under the transparent ansi theme
        # the CSS path yields an empty style, so TextArea falls back to the
        # inverse of the (default) background — a dim gray. Return an explicit
        # bright-white-on-black block instead.
        if "text-area--cursor" in names:
            return Style(bgcolor=Color.from_ansi(15), color=Color.from_ansi(0))
        return super().get_component_rich_style(*names, partial=partial, default=default)

    def render_lines(self, crop: Region) -> list[Strip]:
        strips = super().render_lines(crop)
        return [
            Strip(
                [Segment(text, _strip_default_bg(style), ctrl) for text, style, ctrl in strip],
                strip.cell_length,
            )
            for strip in strips
        ]


class GekaiApp(App[None]):
    CSS = """
    App {
        background: ansi_default;
        color: ansi_default;
    }

    Screen {
        background: ansi_default;
        color: ansi_default;
    }

    ScrollableContainer {
        height: 1fr;
        padding: 0 0 1 0;
        background: ansi_default;
        scrollbar-size: 0 0;
    }

    #footer {
        dock: bottom;
        height: auto;
        padding-bottom: 1;
        background: ansi_default;
    }

    #status-line {
        height: 1;
        background: ansi_default;
        padding: 0;
        display: none;
    }

    #status-spacer {
        height: 1;
        background: ansi_default;
        display: none;
    }

    #hint-area {
        height: 1;
        background: ansi_default;
        color: grey;
        padding: 0 1 0 0;
        text-align: right;
        display: none;
    }

    #input-area {
        height: auto;
        layers: input marker;
        border-top: solid #3a3a3a;
        border-bottom: solid #3a3a3a;
        border-title-align: right;
        border-title-color: #000000;
        border-title-background: ansi_default;
        padding: 0;
        background: ansi_default;
    }

    #prompt-marker {
        layer: marker;
        position: absolute;
        offset: 0 0;
        width: 1;
        height: 1;
        background: ansi_default;
        color: grey;
        padding: 0;
        margin: 0;
    }

    #prompt {
        layer: input;
        width: 100%;
        min-width: 0;
        height: auto;
        max-height: 8;
        padding: 0 0 0 2;
        background: ansi_default;
        background-tint: transparent;
        color: ansi_default;
    }

    #prompt:focus {
        background: ansi_default;
        background-tint: transparent;
    }

    #prompt .text-area--cursor-line {
        background: ansi_default;
    }

    #prompt .text-area--gutter {
        background: ansi_default;
    }

    #prompt .text-area--cursor-gutter {
        background: ansi_default;
    }

    .assistant-spacer {
        height: 1;
        background: ansi_default;
    }

    MessageWidget {
        background: ansi_default;
    }

    .subagent-progress {
        width: 40%;
        height: 1;
        background: ansi_default;
        padding: 0;
        layout: horizontal;
    }

    .subagent-progress Bar {
        width: 1fr;
    }

    .subagent-progress PercentageStatus {
        width: 5;
        color: grey;
    }

    #copy-notice {
        height: 1;
        background: ansi_default;
        color: grey;
        padding: 0 2 0 0;
        text-align: right;
        display: none;
    }

    #context-bar {
        height: 1;
        background: ansi_default;
        color: grey;
        text-align: left;
        padding: 0 0 0 2;
    }

    #version-bar {
        height: 1;
        background: ansi_default;
        text-align: right;
        padding: 0 2 0 0;
    }

    #scroll-hint-wrap {
        height: 1;
        align-horizontal: center;
        background: ansi_default;
        margin-bottom: 1;
        display: none;
    }

    #scroll-hint {
        height: 1;
        background: #555555;
        color: white;
        width: auto;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel_stream", "Cancel"),
        ("ctrl+c", "quit", "Quit"),
        Binding("ctrl+up", "scroll_to_top", "Scroll to top", priority=True),
        Binding("ctrl+down", "scroll_to_end", "Scroll to bottom", priority=True),
        Binding("pageup", "scroll_page_up", "Scroll page up", priority=True),
        Binding("pagedown", "scroll_page_down", "Scroll page down", priority=True),
        Binding("up", "navigate_up", show=False, priority=True),
        Binding("down", "navigate_down", show=False, priority=True),
        Binding("ctrl+r", "toggle_history", "History", priority=True),
        Binding("enter", "confirm_or_submit", "Confirm", priority=True, show=False),
    ]

    # Slash commands that require parameters are inserted into the input with a
    # trailing space when selected from the CommandPalette, allowing the user to
    # enter arguments before the command is submitted.
    _COMMANDS_WITH_ARGS: set[str] = {"config:gate"}

    def __init__(
        self,
        *,
        agent: GekaiAgent,
        registry: CommandRegistry,
        working_dir: Path,
        version: str,
        branch: str | None,
        restored_id: str | None = None,
        restored_messages: list[dict] | None = None,
        restored_timeline: list[dict] | None = None,
        needs_permissions: bool = False,
        **kwargs: object,
    ) -> None:
        self._agent = agent
        self._session: Session | None = None
        self._command_registry = registry
        self._working_dir = working_dir
        self._version = version
        self._branch = branch
        self._restored_id = restored_id
        self._restored_messages = restored_messages
        self._restored_timeline = restored_timeline
        self._needs_permissions = needs_permissions
        self._worker: Worker | None = None
        self._workspace: dict | None = None
        self._assistant_widget: MessageWidget | None = None
        self._status_task: asyncio.Task[None] | None = None
        self._status_stop: asyncio.Event | None = None
        self._status_frame: int = 0
        self._status_verb: str = ""
        self._route_color: str = _DEFAULT_ROUTE_COLOR
        self._status_start: float = 0.0
        self._esc_pending: bool = False
        self._pending_choice: asyncio.Future[str | None] | None = None
        self._context_limit: int = 128_000
        self._history: PromptHistory | None = None
        self._file_paths: list[str] | None = None
        self._file_at_pos: int = -1
        self._worker_cancelled: bool = False
        self._permission_denied_msg: str | None = None
        self._status_paused: bool = False
        super().__init__(**kwargs)
        self.ansi_color = True

    def compose(self) -> ComposeResult:
        yield ConversationContainer(id="conversation")
        with Container(id="footer"):
            yield ChoiceBar(id="choice-bar")
            yield Static("", id="status-line")
            yield Static("", id="status-spacer")
            yield CommandPalette(self._command_registry, id="command-palette")
            yield Static("", id="hint-area")
            with Container(id="scroll-hint-wrap"):
                yield Static("Scroll to bottom  ctrl+↓", id="scroll-hint")
            yield FilePanel(id="file-panel")
            yield HistoryPanel(id="history-panel")
            yield Static("", id="copy-notice")
            with Container(id="input-area"):
                yield PromptTextArea(id="prompt", show_line_numbers=False, compact=True, highlight_cursor_line=False)
                yield Static("❯", id="prompt-marker")
            yield Static("", id="context-bar")
            yield Static(f"[dim]{__version_core__}[/dim] [bold white]{__version_label__}[/bold white]", id="version-bar")

    async def on_mount(self) -> None:
        # Disable terminal mouse tracking so native text selection works.
        # Patch the method so Textual cannot re-enable it after screen transitions.
        # Trade-off: mouse wheel scroll and in-conversation click handlers stop working.
        _driver = getattr(self, "_driver", None)
        if _driver is not None:
            if hasattr(_driver, "_enable_mouse_support"):
                _driver._enable_mouse_support = lambda: None
            try:
                _driver.write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l")
            except Exception:
                pass

        asyncio.create_task(self._poll_clipboard())
        self.run_worker(self._init_session(), exclusive=True)

    async def _init_session(self) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)

        banner_text = pyfiglet.figlet_format("gek-AI", font="small_slant").rstrip()
        await conversation.mount(MessageWidget(MessageKind.BANNER, banner_text))
        cache_path = self._working_dir / ".gekai" / "workspace.json"
        workspace: dict = {}

        if cache_path.exists():
            try:
                workspace = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                workspace = {}

        self._workspace = workspace
        self._session = self._agent.start_session(
            workspace,
            restored_messages=self._restored_messages,
            session_id=self._restored_id,
        )
        self._session.scope_gate = load_scope_gate(self._working_dir)
        self._session.blast_radius_limit = load_blast_radius_limit(self._working_dir)

        history_path = (
            Path.home() / ".gekai" / "workspaces"
            / _normalize_path(self._working_dir) / "history.jsonl"
        )
        self._history = PromptHistory(history_path)

        override = load_context_limit(self._working_dir)
        self._context_limit = override if override is not None else _context_limit(self._agent.model)
        self.query_one("#context-bar", Static).update(
            _fmt_status_bar(self._agent.model, self._working_dir.name, self._branch, _estimate_session_tokens(self._session), self._context_limit)
        )

        if self._restored_timeline:
            for entry in self._restored_timeline:
                kind = entry.get("kind", "turn")
                content = entry.get("content", "")
                if kind == "turn":
                    role = entry.get("role")
                    if role == "user":
                        await conversation.mount(MessageWidget(MessageKind.USER, content))
                    elif role == "assistant":
                        await conversation.mount(MessageWidget(MessageKind.ASSISTANT, content))
                elif kind == "command":
                    await conversation.mount(MessageWidget(MessageKind.USER, content))
                elif kind == "event":
                    source = entry.get("source", "")
                    if source == "command":
                        await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, content))
                    elif source in ("router", "gate"):
                        await conversation.mount(MessageWidget(MessageKind.REJECTED, content))
                    elif source == "error":
                        await conversation.mount(MessageWidget(MessageKind.ERROR, content))
                    elif source == "interrupted":
                        await conversation.mount(MessageWidget(MessageKind.INTERRUPTED, content))
                    elif source in ("farewell", "max_iterations"):
                        await conversation.mount(MessageWidget(MessageKind.ASSISTANT, content))
            self.call_after_refresh(conversation.scroll_end)
        elif self._restored_messages:
            for msg in self._restored_messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    await conversation.mount(MessageWidget(MessageKind.USER, content))
                elif role == "assistant":
                    await conversation.mount(MessageWidget(MessageKind.ASSISTANT, content))
            self.call_after_refresh(conversation.scroll_end)

        self._set_route_label("default")
        self._focus_prompt()
        self.call_after_refresh(self._focus_prompt)

        if self._needs_permissions:
            self._needs_permissions = False
            choice = await self._ask_choice(
                "Gekai needs access to this workspace:",
                PERMISSION_CHOICES,
            )
            if choice is None:
                self.exit()
                return
            perms = resolve_permissions(choice)
            save_permissions(self._working_dir, perms)
            self._agent.permissions = perms
            self._session.permissions = perms

        await self._run_ws_manager(conversation)

    async def _clear_session(self, command_text: str | None = None) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)
        await conversation.remove_children()
        self._session = self._agent.start_session(self._workspace)
        self._session.scope_gate = load_scope_gate(self._working_dir)
        self._session.blast_radius_limit = load_blast_radius_limit(self._working_dir)
        self.query_one("#context-bar", Static).update(
            _fmt_status_bar(self._agent.model, self._working_dir.name, self._branch, _estimate_session_tokens(self._session), self._context_limit)
        )
        self._assistant_widget = None
        banner_text = pyfiglet.figlet_format("gek-AI", font="small_slant").rstrip()
        await conversation.mount(MessageWidget(MessageKind.BANNER, banner_text))
        if command_text is not None:
            append_command(self._session, command_text)
            await conversation.mount(MessageWidget(MessageKind.USER, command_text))
            await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, ""))
        self._focus_prompt()

    @property
    def session_id(self) -> str | None:
        return self._session.id if self._session else None

    @property
    def session_has_interactions(self) -> bool:
        if self._session is None:
            return False
        from agent.persistence import session_file
        path = session_file(self._session)
        return path.exists() and path.stat().st_size > 0

    def _focus_prompt(self) -> None:
        self.query_one("#prompt", TextArea).focus(scroll_visible=False)

    def _prompt_move_to_end(self, prompt: TextArea) -> None:
        prompt.move_cursor(prompt.document.end)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        history_panel = self.query_one("#history-panel", HistoryPanel)
        if history_panel.display:
            return

        palette = self.query_one(CommandPalette)
        value = event.text_area.text
        if value.startswith("/"):
            palette.filter(value[1:])
        else:
            palette.hide()

        file_panel = self.query_one("#file-panel", FilePanel)
        at_pos = value.rfind("@")
        if at_pos != -1:
            query = value[at_pos + 1:]
            if " " not in query and "\n" not in query:
                self._file_at_pos = at_pos
                if self._file_paths is None:
                    self._file_paths = list_files(self._working_dir)
                if not file_panel.display:
                    file_panel.show(self._file_paths, query)
                else:
                    file_panel.filter(query)
            else:
                file_panel.hide()
                self._file_at_pos = -1
        else:
            file_panel.hide()
            self._file_at_pos = -1

        if self._esc_pending:
            self._clear_hint()

    def on_key(self, event: events.Key) -> None:
        choice_bar = self.query_one(ChoiceBar)
        if choice_bar.display and event.key in ("left", "right"):
            if event.key == "left":
                choice_bar.move_left()
            else:
                choice_bar.move_right()
            event.stop()
            return

        if self._esc_pending and event.key != "escape":
            self._clear_hint()

        prompt = self.query_one("#prompt", TextArea)
        if event.key == "ctrl+j" and prompt.has_focus:
            prompt.insert("\n")
            event.stop()
            return
        if prompt.has_focus or not event.is_printable:
            return
        prompt.focus(scroll_visible=False)
        prompt.insert(event.character)
        event.stop()

    async def _permission_callback(self, kind: str, tool_name: str) -> bool:
        if self._worker_cancelled:
            return False
        labels = {"read": "file reading", "write": "file writing", "exec": "command execution"}
        question = f"'{tool_name}' needs {labels.get(kind, kind)} permission — grant?"
        self._status_paused = True
        pause_start = time.monotonic()
        choice = await self._ask_choice(question, [("y", "Yes"), ("n", "No")])
        self._status_paused = False
        if choice == "y":
            self._status_start += time.monotonic() - pause_start
            setattr(self._session.permissions, kind, True)
            save_permissions(self._working_dir, self._session.permissions)
            return True
        label = labels.get(kind, kind)
        self._permission_denied_msg = f"[white]Access denied[/white]\n[#666666]⎿ the operation requires [bold]{label}[/bold] permission[/#666666]"
        if self._worker is not None and not self._worker.is_finished:
            self._worker_cancelled = True
            self._worker.cancel()
        return False

    async def _ask_choice(
        self,
        question: str,
        options: list[tuple[str, str]],
        default_index: int = 0,
    ) -> str | None:
        loop = asyncio.get_event_loop()
        self._pending_choice = loop.create_future()
        self.query_one(ChoiceBar).show(question, options, default_index)
        return await self._pending_choice

    async def _submit_prompt(self) -> None:
        if self._session is None:
            self._focus_prompt()
            return
        if self._worker is not None and not self._worker.is_finished:
            self._focus_prompt()
            return

        prompt = self.query_one("#prompt", TextArea)
        palette = self.query_one(CommandPalette)
        stripped = prompt.text.strip()
        if palette.display:
            cmd = palette.selected_command
            palette.hide()
            if cmd:
                if cmd in self._COMMANDS_WITH_ARGS:
                    val = f"/{cmd} "
                    prompt.text = val
                    self._prompt_move_to_end(prompt)
                    self._focus_prompt()
                    return
                stripped = f"/{cmd}"
        if not stripped:
            self._focus_prompt()
            return
        if self._history is not None:
            self._history.append(stripped)
            self._history.reset()
        history_panel = self.query_one("#history-panel", HistoryPanel)
        if history_panel.display:
            history_panel.hide()
        prompt.clear()
        conversation = self.query_one("#conversation", ScrollableContainer)
        if stripped.startswith("/"):
            await conversation.mount(MessageWidget(MessageKind.USER, stripped))
            conversation.scroll_end(animate=False)
            had_prior = self.session_has_interactions
            if self._session is not None:
                append_command(self._session, stripped)
            result = await self._command_registry.dispatch(stripped)
            if result.scope_gate is not None and self._session is not None:
                self._session.scope_gate = result.scope_gate
            if result.clear_session:
                await self._clear_session(command_text=stripped)
                return
            output = result.output or ""
            await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, output))
            conversation.scroll_end(animate=False)
            if self._session is not None:
                append_event(self._session, output, source="command")
            if result.exit_app:
                farewell = random_farewell()
                await conversation.mount(MessageWidget(MessageKind.ASSISTANT, farewell))
                conversation.scroll_end(animate=False)
                if self._session is not None and not had_prior:
                    from agent.persistence import session_file
                    session_file(self._session).unlink(missing_ok=True)
                await asyncio.sleep(0.8 + len(farewell.split(" ")) * 0.20)
                self.exit()
                return
            self._focus_prompt()
            return
        await conversation.mount(MessageWidget(MessageKind.USER, stripped))
        self._worker = self.run_worker(self._stream(_resolve_at_refs(stripped)), exclusive=True)

    async def _run_ws_manager(self, conversation: ScrollableContainer) -> None:
        from agent import workspace_db as _wsdb
        try:
            conn = await asyncio.to_thread(_wsdb.ensure, self._working_dir)
            conn.close()
        except Exception:
            pass

    # [dead code] workspace staleness rescan — disabled pending redesign
    async def _maybe_rescan_workspace(self, conversation: ScrollableContainer) -> None:
        """Run workspace re-scan activation logic. Updates session and self._workspace if scan fires."""
        cache_path = self._working_dir / ".gekai" / "workspace.json"
        if not cache_path.exists():
            # Onboarding fallback — should not normally happen here
            await self._run_ws_manager(conversation)
            return

        try:
            scan_state = json.loads(cache_path.read_text(encoding="utf-8")).get("scan_state", {})
        except (OSError, ValueError):
            scan_state = {}

        timestamp_str = scan_state.get("timestamp", "")
        staleness_min = load_ws_scan_staleness_min(self._working_dir)
        staleness_sec = staleness_min * 60

        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - ts).total_seconds()
        except (ValueError, TypeError):
            elapsed = float("inf")

        if elapsed <= staleness_sec:
            return  # fresh — skip

        # Stale: check git state
        commit_hash, dirty = await asyncio.to_thread(_get_git_state, self._working_dir)
        cached_hash = scan_state.get("commit_hash")
        cached_uncommitted = scan_state.get("uncommitted", False)

        if commit_hash == cached_hash and dirty == cached_uncommitted:
            # Git unchanged — silently refresh timestamp
            await asyncio.to_thread(_update_scan_state_key, cache_path, "timestamp", now_utc_str())
            return

        # Git changed — throttle check
        asked_str = scan_state.get("asked_timestamp", "")
        if asked_str:
            try:
                asked_ts = datetime.fromisoformat(asked_str.replace("Z", "+00:00"))
                since_asked = (datetime.now(timezone.utc) - asked_ts).total_seconds()
                if since_asked < staleness_sec:
                    return  # throttled
            except (ValueError, TypeError):
                pass

        # Ask user
        await asyncio.to_thread(_update_scan_state_key, cache_path, "asked_timestamp", now_utc_str())
        if await self._ask_choice("Workspace needs rescan — proceed?", [("y", "Yes"), ("n", "No")]) == "y":
            await self._run_ws_manager(conversation)

    def _set_route_label(self, label: str, color: str | None = None) -> None:
        if color is not None:
            self._route_color = color
        # Inline markup: colored label block + a 2-char border-line segment so the
        # right-aligned title sits spaced off the corner. The colored span and the
        # transparent dash tail must be styled per-cell, which a single
        # border_title_background style can't express.
        self.query_one("#input-area", Container).border_title = (
            f"[#000000 on {self._route_color}] {label.lower()} [/][#3a3a3a]─[/]"
        )

    async def _stream(self, user_input: str) -> None:
        start = time.monotonic()
        verb = random_operative_verb()
        color = random_accent_color()
        conversation = self.query_one("#conversation", ScrollableContainer)
        answer_chunks: list[str] = []
        max_iter_hit: bool = False
        ws_renderer: SubAgentRenderer | None = None
        query_tool_count: int = 0

        try:
            await self._start_status_animation(verb[0], color)
            route = await self._agent.route(user_input, history=self._session.messages)
            if route.intent is Intent.REJECTED:
                msg = "User input must be in English"
                await conversation.mount(MessageWidget(MessageKind.REJECTED, msg))
                append_event(self._session, msg, source="router")
                return
            if self._session is not None:
                append_message(self._session, {"role": "user", "content": user_input})
            _ns = route.namespace
            if route.intent is Intent.ACTION:
                _label = f"action/{_ns}" if _ns else "action"
                _color = _NS_COLORS.get(_ns, _ACTION_COLOR)
            else:
                _label = route.intent.name.lower()
                _color = _DEFAULT_ROUTE_COLOR
            self._set_route_label(_label, color=_color)

            entries: list[tuple[str, list[str]]] | None = None
            processed_input = user_input
            original_input: str | None = None
            if route.intent is Intent.ACTION and route.profile is not None:
                entries = await self._agent.locate(self._session.working_dir, user_input)
                if self._agent.debug:
                    append_debug(self._session, {"content": {"locate": [path for path, _ in entries]}})
                rejected, reason = self._agent.check_gate(entries, self._session.blast_radius_limit)
                if rejected and self._session.scope_gate:
                    reason_text = reason or "request exceeds scope"
                    await conversation.mount(MessageWidget(MessageKind.REJECTED, reason_text))
                    append_event(self._session, reason_text, source="gate")
                    return
                if entries:
                    processed_input = await self._agent.rewrite(user_input, entries)
                    original_input = user_input
                    if self._agent.debug:
                        append_debug(self._session, {"content": {"rewritten": processed_input}})

            if self._agent.debug:
                parts = [route.intent.name.lower()]
                if _ns:
                    parts.append(_ns)
                if route.profile:
                    parts.append(route.profile.name)
                debug_text = f"\\[router: {'/'.join(parts)}]"
                await conversation.mount(MessageWidget(MessageKind.OPERATION, debug_text, color="#BA55D3"))

            async for item in self._agent.process_stream(
                self._session, processed_input, route,
                entries=entries,
                original_input=original_input,
                permission_callback=self._permission_callback,
            ):
                if isinstance(item, str):
                    answer_chunks.append(item)
                elif isinstance(item, SubAgentEvent):
                    if isinstance(item, SubAgentStartEvent):
                        self._set_route_label(item.name, color=item.color)
                        ws_renderer = SubAgentRenderer(conversation, debug=self._agent.debug)
                        await ws_renderer.start(item.name, item.description, item.color)
                    elif ws_renderer:
                        if isinstance(item, LogEvent):
                            await ws_renderer.log(item.message, tool_name=item.tool_name)
                            if item.tool_name:
                                query_tool_count += 1
                        elif isinstance(item, DiffEvent):
                            await conversation.mount(DiffWidget(item.path, item.diff_lines))
                            conversation.scroll_end(animate=False)
                        elif isinstance(item, InferEndEvent):
                            ws_renderer.accumulate_tokens(item)
                        elif isinstance(item, ThinkingTokenEvent):
                            ws_renderer.thinking_chunk(item.text)
                        elif isinstance(item, StatusUpdateEvent):
                            await ws_renderer.status_update(item)
                        elif isinstance(item, DoneEvent):
                            await ws_renderer.done(item.thinking_chars)
                        elif isinstance(item, MaxIterationsEvent):
                            max_iter_hit = True

            if self._session is not None:
                self.query_one("#context-bar", Static).update(
                    _fmt_context_pct(_estimate_session_tokens(self._session), self._context_limit)
                )
            if max_iter_hit and not answer_chunks:
                await conversation.mount(MessageWidget(MessageKind.ERROR, "agent hit iteration limit without producing a response"))
            else:
                answer = "".join(answer_chunks).rstrip("\n")
                self._assistant_widget = MessageWidget(MessageKind.ASSISTANT, answer)
                await conversation.mount(self._assistant_widget)
                elapsed = time.monotonic() - start
                await conversation.mount(
                    MessageWidget(
                        MessageKind.OPERATION,
                        f"* {verb[1]} for {_fmt_duration(elapsed)}" + (f" ({query_tool_count} {'tool' if query_tool_count == 1 else 'tools'})" if query_tool_count > 0 else ""),
                        color=color,
                    )
                )
            conversation.scroll_end(animate=False)
        except Exception as error:
            error_msg = str(error)
            await conversation.mount(MessageWidget(MessageKind.ERROR, error_msg))
            append_event(self._session, error_msg, source="error")
            conversation.scroll_end(animate=False)
        finally:
            self._set_route_label("default", color=_DEFAULT_ROUTE_COLOR)
            await self._stop_status_animation()
            if ws_renderer is not None:
                ws_renderer.stop_spinner()
            if self._worker_cancelled:
                self._worker_cancelled = False
                msg = self._permission_denied_msg or ""
                self._permission_denied_msg = None
                await conversation.mount(MessageWidget(MessageKind.INTERRUPTED, msg))
                append_event(self._session, msg, source="interrupted")
                conversation.scroll_end(animate=False)
            self._worker = None
            self._focus_prompt()

    # [dead code] /workspace:rebuild command handler — disabled pending redesign
    async def _rebuild_workspace(self) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)
        try:
            await self._run_ws_manager(conversation)
            self._session = self._agent.start_session(self._workspace or {})
            conversation.scroll_end(animate=False)
        except Exception as error:
            await conversation.mount(MessageWidget(MessageKind.ERROR, str(error)))
            conversation.scroll_end(animate=False)
        finally:
            self._worker = None
            self._focus_prompt()

    async def _animate_status(self, color: str, stop: asyncio.Event) -> None:
        try:
            while not stop.is_set():
                self._tick_status(color)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.15)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._clear_status()

    async def _start_status_animation(self, verb: str, color: str) -> None:
        self._status_verb = verb
        self._status_start = time.monotonic()
        if self._status_task is not None:
            return
        await self._stop_status_animation()
        stop = asyncio.Event()
        self._status_stop = stop
        self._status_task = asyncio.create_task(self._animate_status(color, stop))

    async def _stop_status_animation(self) -> None:
        if self._status_stop is not None:
            self._status_stop.set()
        task = self._status_task
        self._status_task = None
        self._status_stop = None
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _tick_status(self, color: str) -> None:
        if self._status_paused:
            return
        frame = _SPINNER_FRAMES[self._status_frame % len(_SPINNER_FRAMES)]
        self._status_frame += 1
        elapsed = time.monotonic() - self._status_start
        text = _fmt_status(self._status_verb, elapsed)
        self._set_status(f"{frame} {text}", color)

    def _set_status(self, text: str, color: str) -> None:
        status = self.query_one("#status-line", Static)
        status.update(text)
        status.styles.color = color
        if not status.display:
            status.display = True
            self.query_one("#status-spacer", Static).display = True

    def _clear_status(self) -> None:
        status = self.query_one("#status-line", Static)
        status.update("")
        status.display = False
        self.query_one("#status-spacer", Static).display = False

    def _show_hint(self, text: str) -> None:
        hint = self.query_one("#hint-area", Static)
        hint.update(text)
        hint.display = True

    def _clear_hint(self) -> None:
        self._esc_pending = False
        hint = self.query_one("#hint-area", Static)
        hint.update("")
        hint.display = False

    def on_conversation_container_scrolled(self, event: ConversationContainer.Scrolled) -> None:
        is_streaming = self._worker is not None and not self._worker.is_finished
        self.query_one("#scroll-hint-wrap", Container).display = not event.at_end and not is_streaming

    def action_cancel_stream(self) -> None:
        file_panel = self.query_one("#file-panel", FilePanel)
        if file_panel.display:
            file_panel.hide()
            self._file_at_pos = -1
            self._focus_prompt()
            return

        history_panel = self.query_one("#history-panel", HistoryPanel)
        if history_panel.display:
            history_panel.hide()
            self._focus_prompt()
            return

        if self._pending_choice is not None and not self._pending_choice.done():
            self.query_one(ChoiceBar).hide()
            future = self._pending_choice
            self._pending_choice = None
            future.set_result(None)
            return

        palette = self.query_one(CommandPalette)
        if palette.display:
            self.query_one("#prompt", TextArea).clear()
            palette.hide()
            self._clear_hint()
            return
        if self._worker is not None and not self._worker.is_finished:
            self._worker_cancelled = True
            self._worker.cancel()
            self._clear_hint()
            self._clear_status()
            self._focus_prompt()
            return

        prompt = self.query_one("#prompt", TextArea)
        if self._esc_pending:
            prompt.clear()
            self._clear_hint()
        elif prompt.text:
            self._esc_pending = True
            self._show_hint("ESC again to clear input")

    def action_toggle_history(self) -> None:
        panel = self.query_one("#history-panel", HistoryPanel)
        if panel.display:
            panel.hide()
            self._focus_prompt()
        else:
            if self._history is None:
                return
            entries = self._history.load()
            entries_display = list(reversed(entries))  # newest first
            panel.show(entries_display, selected_index=0)

    async def action_confirm_or_submit(self) -> None:
        if self._pending_choice is not None and not self._pending_choice.done():
            self.query_one("#prompt", TextArea).clear()
            choice_bar = self.query_one(ChoiceBar)
            key = choice_bar.selected_key
            choice_bar.hide()
            future = self._pending_choice
            self._pending_choice = None
            future.set_result(key)
            self._focus_prompt()
            return

        file_panel = self.query_one("#file-panel", FilePanel)
        if file_panel.display:
            path = file_panel.selected_text
            file_panel.hide()
            if path is not None:
                prompt = self.query_one("#prompt", TextArea)
                at_pos = self._file_at_pos
                if at_pos != -1 and at_pos < len(prompt.text):
                    new_value = prompt.text[:at_pos] + f"@{path} "
                    prompt.text = new_value
                    self._prompt_move_to_end(prompt)
            self._file_at_pos = -1
            self._focus_prompt()
            return

        panel = self.query_one("#history-panel", HistoryPanel)
        if panel.display:
            text = panel.selected_text
            panel.hide()
            prompt = self.query_one("#prompt", TextArea)
            if text:
                prompt.text = text
                self._prompt_move_to_end(prompt)
            self._focus_prompt()
            return

        await self._submit_prompt()

    def on_file_panel_row_clicked(self, event: FilePanel.RowClicked) -> None:
        file_panel = self.query_one("#file-panel", FilePanel)
        file_panel.select_index(event.index)
        path = file_panel.selected_text
        file_panel.hide()
        if path is not None:
            prompt = self.query_one("#prompt", TextArea)
            at_pos = self._file_at_pos
            if at_pos != -1 and at_pos < len(prompt.text):
                new_value = prompt.text[:at_pos] + f"@{path} "
                prompt.text = new_value
                self._prompt_move_to_end(prompt)
        self._file_at_pos = -1
        self._focus_prompt()

    def on_history_panel_row_clicked(self, event: HistoryPanel.RowClicked) -> None:
        panel = self.query_one("#history-panel", HistoryPanel)
        panel.select_index(event.index)
        text = panel.selected_text
        if text:
            prompt = self.query_one("#prompt", TextArea)
            prompt.text = text
            self._prompt_move_to_end(prompt)
        panel.hide()
        self._focus_prompt()

    @staticmethod
    def _clipboard_sequence() -> int:
        try:
            import ctypes
            return ctypes.windll.user32.GetClipboardSequenceNumber()
        except Exception:
            return 0

    async def _poll_clipboard(self) -> None:
        last = self._clipboard_sequence()
        while True:
            await asyncio.sleep(0.4)
            current = self._clipboard_sequence()
            if current != last:
                last = current
                self._show_copy_notice()

    def _show_copy_notice(self) -> None:
        notice = self.query_one("#copy-notice", Static)
        notice.update("copied to clipboard")
        notice.display = True
        self.set_timer(2.0, self._hide_copy_notice)

    def _hide_copy_notice(self) -> None:
        try:
            self.query_one("#copy-notice", Static).display = False
        except Exception:
            pass

    def action_scroll_to_top(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_home(animate=False)

    def action_scroll_to_end(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_end(animate=False)

    def action_scroll_page_up(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_page_up(animate=False)

    def action_scroll_page_down(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_page_down(animate=False)

    def action_navigate_up(self) -> None:
        if self.query_one("#file-panel", FilePanel).display:
            self.query_one("#file-panel", FilePanel).move_up()
            return
        panel = self.query_one("#history-panel", HistoryPanel)
        if panel.display:
            panel.move_up()
            if (text := panel.selected_text) is not None:
                self.query_one("#prompt", TextArea).text = text
            return
        if self.query_one(CommandPalette).display:
            self.query_one(CommandPalette).move_up()
            return
        if self.query_one(ChoiceBar).display:
            self.query_one(ChoiceBar).move_left()
            return
        prompt = self.query_one("#prompt", TextArea)
        if prompt.has_focus and not prompt.cursor_at_first_line:
            prompt.action_cursor_up()
            return
        self.query_one("#conversation", ConversationContainer).scroll_up(animate=False)

    def action_navigate_down(self) -> None:
        if self.query_one("#file-panel", FilePanel).display:
            self.query_one("#file-panel", FilePanel).move_down()
            return
        panel = self.query_one("#history-panel", HistoryPanel)
        if panel.display:
            panel.move_down()
            if (text := panel.selected_text) is not None:
                self.query_one("#prompt", TextArea).text = text
            return
        if self.query_one(CommandPalette).display:
            self.query_one(CommandPalette).move_down()
            return
        if self.query_one(ChoiceBar).display:
            self.query_one(ChoiceBar).move_right()
            return
        prompt = self.query_one("#prompt", TextArea)
        if prompt.has_focus and not prompt.cursor_at_last_line:
            prompt.action_cursor_down()
            return
        self.query_one("#conversation", ConversationContainer).scroll_down(animate=False)

    async def action_select_command(self, name: str) -> None:
        palette = self.query_one(CommandPalette)
        palette.hide()
        prompt = self.query_one("#prompt", TextArea)
        if name in self._COMMANDS_WITH_ARGS:
            val = f"/{name} "
            prompt.text = val
            self._prompt_move_to_end(prompt)
            self._focus_prompt()
        else:
            prompt.text = f"/{name}"
            await self._submit_prompt()

    @on(events.Click, "#scroll-hint")
    def _scroll_hint_clicked(self, event: events.Click) -> None:
        event.stop()
        self.action_scroll_to_end()

        prompt = self.query_one("#prompt", TextArea)
        if self._esc_pending:
            prompt.clear()
            self._clear_hint()
        elif prompt.text:
            self._esc_pending = True
            self._show_hint("ESC again to clear input")

        self._clear_status()
        self._focus_prompt()
