from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
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
from agent.diff import DiffLine
from agent.persistence import (
    append_command,
    append_debug,
    append_diff,
    append_event,
    append_operation,
    append_subagent_done,
    append_subagent_start,
    load_route_decisions,
    _normalize_path,
)
from agent.pipeline import Route
from agent.session import Session
from agent.subagents import NAMESPACE_COLORS, Subagent
from agent.subagents.worker import ws_manager
from agent.settings import PERMISSION_CHOICES, load_context_limit, resolve_permissions, save_permissions
from agent.workspace import db as workspace_db, list_files, list_dirs
from agent.tui.styles import random_accent_color, random_operative_verb
from agent.events import AgentEvent, SubAgentStartEvent, LogEvent, DiffEvent, InferEndEvent, DoneEvent, MaxIterationsEvent, BudgetExhaustedEvent, StatusUpdateEvent, ThinkingTokenEvent, SubagentResult

from .palette import CommandPalette
from .history import PromptHistory
from .widgets import ChoiceBar, DiffWidget, FilePanel, HistoryPanel, MessageKind, MessageWidget, WelcomeOverlay

_DEFAULT_ROUTE_COLOR = "#3a3a3a"
_PIPELINE_COLOR = "#ffffff"  # pure-white bg marks active pre-harness pipeline step
_DELEGATION_FILES_CAP = 20


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
        self._badge_namespace: str | None = None
        self._badge_color: str = ""
        self._tool_calls: int = 0

    async def _animate_dot(self) -> None:
        frame = 0
        dot_color = self._badge_color if self._badge_namespace is not None else "#666666"
        try:
            while True:
                if self._header_widget is not None:
                    char = _BRAILLE_FRAMES[frame % len(_BRAILLE_FRAMES)]
                    self._header_widget.query_one(".header-dot", Static).update(f"[{dot_color}]{char}[/{dot_color}]")
                frame += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    async def start(self, name: str, *, namespace: str | None = None, ui_label: str = "", bg_color: str = "") -> None:
        self.name = name
        self._badge_namespace = namespace
        self._badge_color = bg_color
        await self._conversation.mount(Static("", classes="assistant-spacer"))
        if namespace is not None:
            header_markup = _subagent_header_markup(name, bg_color, ui_label)
        else:
            header_markup = "[bold #666666]Thinking...[/bold #666666]"
        widget = MessageWidget(MessageKind.HEADER, header_markup)
        await self._conversation.mount(widget)
        self._header_widget = widget
        self._spinner_task = asyncio.create_task(self._animate_dot())

    async def log(self, message: str, tool_name: str = "") -> None:
        if tool_name:
            self._tool_calls += 1
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
        if self._badge_namespace is not None:
            return  # badge header is persistent — never overwrite it with a thinking preview
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

    async def done(self, thinking_chars: int = 0) -> str:
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
        if self._badge_namespace is not None:
            parts: list[str] = []
            if self._tool_calls > 0:
                parts.append(f"{self._tool_calls} tool" + ("s" if self._tool_calls != 1 else ""))
            if self._total_tokens > 0:
                parts.append(f"{_fmt_tokens(self._total_tokens)} tokens")
            parts.append(_fmt_duration_verbose(elapsed))
            summary = " · ".join(parts)
            if self._header_widget is not None:
                self._header_widget.query_one(".header-dot", Static).update(f"[{self._badge_color}]●[/{self._badge_color}]")
                self._header_widget = None
            # badge header persists untouched — mount the Done summary as a
            # permanent connector line beneath it (it is now the sole survivor
            # under the header, so it always anchors the L-connector)
            await self._conversation.mount(Static(f"  ⎿ Done ({summary})"))
            self._conversation.scroll_end(animate=False)
            return summary
        else:
            parts = []
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
            return summary


def _subagent_header_markup(name: str, bg_color: str, ui_label: str) -> str:
    markup = f"[black on {bg_color} bold] {name} [/]"
    if ui_label:
        markup += f"[white]\\[{ui_label}][/white]"
    return markup


def _fallback_ui_label(text: str, max_words: int = 6) -> str:
    """Cheap stand-in for the rewriter's <ui_label> when rewriting is skipped
    (no located entries) or the label comes back empty — strips backtick-quoted
    paths so file names don't leak into the badge, then takes the leading words."""
    stripped = re.sub(r"`[^`]*`", "", text)
    words = stripped.split()
    return " ".join(words[:max_words])


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


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _route_decision(route: Route) -> str:
    if route.plan is not None:
        return f"plan({len(route.plan)})"
    if route.subagent is not None:
        return f"{route.subagent.namespace}/{route.subagent.name}"
    if route.trivial:
        return "trivial"
    if route.explore:
        return "explore"
    return "main"


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
    return f"\\[{model}] | {location} | {pct}"


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


@dataclass
class _StepResult:
    """Outcome of one locate->rewrite->dispatch pipeline run for a single
    step (a plan step, or the equivalent single-token turn)."""
    outcome: str = "ok"  # "ok" | "max_iterations"
    answer: str = ""
    max_iter_hit: bool = False
    query_tool_count: int = 0
    ws_renderer: SubAgentRenderer | None = None


class PromptTextArea(TextArea):
    """TextArea that renders without explicit default background, preserving terminal transparency."""

    def on_mount(self) -> None:
        # Blinking block cursor (the old Input look, but blinking).
        self.cursor_blink = True

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
        layers: base overlay;
        align: center middle;
        background: ansi_default;
        color: ansi_default;
    }

    ScrollableContainer {
        height: 1fr;
        padding: 0 0 1 0;
        background: ansi_default;
        scrollbar-size: 0 0;
        layer: base;
    }

    #footer {
        dock: bottom;
        height: auto;
        padding-bottom: 1;
        background: ansi_default;
        layer: base;
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
        Binding("ctrl+b", "scroll_to_end", "Scroll to bottom", priority=True),
        Binding("pageup", "scroll_page_up", "Scroll page up", priority=True),
        Binding("pagedown", "scroll_page_down", "Scroll page down", priority=True),
        Binding("up", "navigate_up", show=False, priority=True),
        Binding("down", "navigate_down", show=False, priority=True),
        Binding("ctrl+r", "toggle_history", "History", priority=True),
        Binding("enter", "confirm_or_submit", "Confirm", priority=True, show=False),
    ]

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
        self._welcome_dismissed: bool = False
        self._exit_reason: str = "quit"
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
                yield Static("Scroll to bottom (ctrl+B) ↓", id="scroll-hint")
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

        banner_text = pyfiglet.figlet_format("gekAI", font="small_slant").rstrip()
        await conversation.mount(MessageWidget(MessageKind.BANNER, banner_text))
        self._session = self._agent.start_session(
            restored_messages=self._restored_messages,
            session_id=self._restored_id,
        )
        self._agent.events.emit("session.start", session=self._session.id, resumed=self._restored_id is not None)

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
            route_decisions = load_route_decisions(self._restored_id) if self._agent.debug and self._restored_id else {}
            last_subagent_header: MessageWidget | None = None
            for entry in self._restored_timeline:
                kind = entry.get("kind", "turn")
                content = entry.get("content", "")
                if kind == "turn":
                    role = entry.get("role")
                    if role == "user":
                        await conversation.mount(MessageWidget(MessageKind.USER, content))
                        decision = route_decisions.get(entry.get("turn", ""))
                        if decision:
                            await conversation.mount(MessageWidget(MessageKind.OPERATION, f"\\[router: {decision}]", color="#BA55D3"))
                    elif role == "assistant":
                        await conversation.mount(MessageWidget(MessageKind.ASSISTANT, content))
                elif kind == "command":
                    await conversation.mount(MessageWidget(MessageKind.USER, content))
                elif kind == "event":
                    source = entry.get("source", "")
                    if source == "command":
                        await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, content))
                    elif source == "error":
                        await conversation.mount(MessageWidget(MessageKind.ERROR, content))
                    elif source == "interrupted":
                        await conversation.mount(MessageWidget(MessageKind.INTERRUPTED, content))
                    elif source == "max_iterations":
                        await conversation.mount(MessageWidget(MessageKind.ASSISTANT, content))
                elif kind == "diff":
                    diff_lines = [DiffLine(kind=d["k"], text=d["t"]) for d in entry.get("lines", [])]
                    await conversation.mount(DiffWidget(entry["path"], diff_lines))
                elif kind == "subagent_start":
                    header_widget = MessageWidget(MessageKind.HEADER, _subagent_header_markup(entry["name"], entry["bg_color"], entry["ui_label"]))
                    await conversation.mount(header_widget)
                    last_subagent_header = header_widget
                elif kind == "subagent_done":
                    if last_subagent_header is not None:
                        bg_color = entry.get("bg_color", "")
                        last_subagent_header.query_one(".header-dot", Static).update(f"[{bg_color}]●[/{bg_color}]")
                        last_subagent_header = None
                    await conversation.mount(Static(f"  ⎿ Done ({entry.get('summary', '')})"))
                elif kind == "operation":
                    await conversation.mount(MessageWidget(MessageKind.OPERATION, entry.get("content", ""), color=entry.get("color")))
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

        self._set_route_label("waiting")
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

        conn = workspace_db.ensure(self._working_dir)
        await self._start_status_animation("indexing workspace", random_accent_color())
        try:
            index_stats = await ws_manager.run("onboard", self._working_dir, conn)
        finally:
            await self._stop_status_animation()
            self._clear_status()
        conn.close()
        self._agent.events.emit(
            "workspace.index",
            session=self._session.id,
            file_count=index_stats.file_count,
            indexed_count=index_stats.indexed_count,
            skipped_fresh=index_stats.skipped_fresh,
            symbols_extracted=index_stats.symbols_extracted,
            symbol_files=index_stats.symbol_files,
            duration_ms=index_stats.duration_ms,
        )

        if self._restored_id is None:
            await self.mount(WelcomeOverlay(id="welcome-overlay"))

    async def _clear_session(self, command_text: str | None = None) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)
        await conversation.remove_children()
        self._session = self._agent.start_session()
        self._agent.events.emit("session.start", session=self._session.id, resumed=False)
        self.query_one("#context-bar", Static).update(
            _fmt_status_bar(self._agent.model, self._working_dir.name, self._branch, _estimate_session_tokens(self._session), self._context_limit)
        )
        self._assistant_widget = None
        banner_text = pyfiglet.figlet_format("gekAI", font="small_slant").rstrip()
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
    def exit_reason(self) -> str:
        return self._exit_reason

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
        if not self._welcome_dismissed and event.text_area.text:
            self._welcome_dismissed = True
            self.query("#welcome-overlay").remove()

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
                    self._file_paths = sorted(list_dirs(self._working_dir) + list_files(self._working_dir))
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

    async def _hidden_grant_callback(self, rel: str, mode: str) -> bool:
        if self._worker_cancelled:
            return False
        question = f"Grant {mode} access to hidden path '{rel}' (excluded by .gitignore)?"
        self._status_paused = True
        pause_start = time.monotonic()
        choice = await self._ask_choice(question, [("y", "Yes"), ("n", "No")])
        self._status_paused = False
        self._status_start += time.monotonic() - pause_start
        return choice == "y"

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
            cmd_name = stripped.lstrip("/").split(maxsplit=1)[0] if stripped.lstrip("/").split() else ""
            self._agent.events.emit("command", session=self.session_id, name=cmd_name)
            result = await self._command_registry.dispatch(stripped)
            if result.clear_session:
                await self._clear_session(command_text=stripped)
                return
            output = result.output or ""
            await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, output))
            conversation.scroll_end(animate=False)
            if self._session is not None:
                append_event(self._session, output, source="command")
            if result.exit_app:
                self._quit(had_prior=had_prior)
                return
            self._focus_prompt()
            return
        await conversation.mount(MessageWidget(MessageKind.USER, stripped))
        self._worker = self.run_worker(self._stream(_resolve_at_refs(stripped)), exclusive=True)

    def _set_route_label(self, label: str, color: str | None = None) -> None:
        if color is not None:
            self._route_color = color
        # Inline markup: colored label block + a 2-char border-line segment so the
        # right-aligned title sits spaced off the corner. The colored span and the
        # transparent dash tail must be styled per-cell, which a single
        # border_title_background style can't express.
        self.query_one("#input-area", Container).border_title = (
            f"[bold #000000 on {self._route_color}] {label.lower()} [/][#3a3a3a]─[/]"
        )

    async def _run_step(
        self, raw: str, subagent: Subagent | None, *,
        turn_id: str, session_id: str, conversation: ScrollableContainer,
        stage: list[str], trivial: bool = False, explore: bool = False,
        append_user: bool = True, step_index: int | None = None,
    ) -> _StepResult:
        """Run locate->rewrite->dispatch for `raw` against `subagent` (None => main).
        `stage` is a 1-element mutable holder the caller's except-block reads to attribute
        which sub-stage failed; trivial/explore are only ever set by the single-token call
        site (a plan step's target is never trivial/explore)."""
        events = self._agent.events

        if subagent is not None:
            _label, _color = subagent.namespace, NAMESPACE_COLORS[subagent.namespace]
        elif explore:
            _label, _color = "explore", _DEFAULT_ROUTE_COLOR
        else:
            _label, _color = "main", _DEFAULT_ROUTE_COLOR

        if trivial or explore:
            entries = []
            if self._agent.debug:
                append_debug(self._session, {"content": {"route": "explore" if explore else "trivial", "skipped": ["locate", "rewrite"]}})
        else:
            stage[0] = "locate"
            self._set_route_label("locate", color=_PIPELINE_COLOR)
            t0 = time.monotonic()
            entries, hint_paths, locate_timed_out = await self._agent.locate(self._session.working_dir, raw)
            located_paths = {path for path, _ in entries}
            overlap = len(set(hint_paths) & located_paths) / len(located_paths) if located_paths else 0.0
            events.emit(
                "locate", session=session_id, turn=turn_id, step=step_index,
                files=len(entries), hints=len(hint_paths), overlap=round(overlap, 3),
                timed_out=locate_timed_out,
                duration_ms=_ms(time.monotonic() - t0),
            )
            if self._agent.debug:
                append_debug(self._session, {"content": {"locate": [path for path, _ in entries], "hints": hint_paths}})

        processed_input = raw
        original_input: str | None = None
        ui_label = ""

        if entries or subagent is not None:
            stage[0] = "rewrite"
            self._set_route_label("rewrite", color=_PIPELINE_COLOR)
            t0 = time.monotonic()
            processed_input, rewrite_label = await self._agent.rewrite(raw, entries)
            events.emit("rewrite", session=session_id, turn=turn_id, step=step_index, ok=True, duration_ms=_ms(time.monotonic() - t0))
            original_input = raw
            if rewrite_label:
                ui_label = rewrite_label
            if self._agent.debug:
                append_debug(self._session, {"content": {"rewritten": processed_input}})

        if subagent is not None and not ui_label:
            ui_label = _fallback_ui_label(original_input or raw)

        if self._agent.debug:
            if subagent is not None:
                parts = [subagent.namespace, subagent.name]
            elif trivial:
                parts = ["trivial"]
            elif explore:
                parts = ["explore"]
            else:
                parts = ["main"]
            debug_text = f"\\[router: {'/'.join(parts)}]"
            await conversation.mount(MessageWidget(MessageKind.OPERATION, debug_text, color="#BA55D3"))

        self._set_route_label(_label, color=_color)
        stage[0] = "harness"
        step_route = Route(subagent=subagent, trivial=trivial, explore=explore)
        harness_start = time.monotonic()
        answer_chunks: list[str] = []
        max_iter_hit = False
        budget_exhausted_hit = False
        ws_renderer: SubAgentRenderer | None = None
        query_tool_count: int = 0
        tool_counts: dict[str, int] = {}
        llm_calls = 0
        prompt_tokens_total = 0
        completion_tokens_total = 0
        thinking_chars_total = 0
        files_touched_total: list[str] = []

        async for item in self._agent.process_stream(
            self._session, processed_input, step_route,
            entries=entries,
            original_input=original_input,
            permission_callback=self._permission_callback,
            hidden_grant_callback=self._hidden_grant_callback,
            turn_id=turn_id,
            append_user=append_user,
        ):
            if isinstance(item, str):
                answer_chunks.append(item)
            elif isinstance(item, AgentEvent):
                if isinstance(item, SubAgentStartEvent):
                    ws_renderer = SubAgentRenderer(conversation, debug=self._agent.debug)
                    if subagent is not None:
                        # input-box label stays the bare namespace (set above);
                        # do not override it with item.name/item.color here
                        await ws_renderer.start(
                            item.name,
                            namespace=subagent.namespace,
                            ui_label=ui_label,
                            bg_color=NAMESPACE_COLORS[subagent.namespace],
                        )
                        if self._session is not None:
                            append_subagent_start(
                                self._session,
                                namespace=subagent.namespace,
                                name=item.name,
                                bg_color=NAMESPACE_COLORS[subagent.namespace],
                                ui_label=ui_label,
                                turn=turn_id,
                            )
                    else:
                        self._set_route_label(item.name, color=item.color)
                        await ws_renderer.start(item.name)
                elif ws_renderer:
                    if isinstance(item, LogEvent):
                        await ws_renderer.log(item.message, tool_name=item.tool_name)
                        if item.tool_name:
                            query_tool_count += 1
                            tool_counts[item.tool_name] = tool_counts.get(item.tool_name, 0) + 1
                    elif isinstance(item, DiffEvent):
                        await conversation.mount(DiffWidget(item.path, item.diff_lines))
                        if self._session is not None:
                            append_diff(self._session, item.path, item.diff_lines, turn=turn_id)
                        conversation.scroll_end(animate=False)
                    elif isinstance(item, InferEndEvent):
                        ws_renderer.accumulate_tokens(item)
                        llm_calls += 1
                        prompt_tokens_total += item.prompt_tokens or 0
                        completion_tokens_total += item.completion_tokens or 0
                    elif isinstance(item, ThinkingTokenEvent):
                        ws_renderer.thinking_chunk(item.text)
                    elif isinstance(item, StatusUpdateEvent):
                        await ws_renderer.status_update(item)
                    elif isinstance(item, DoneEvent):
                        done_summary = await ws_renderer.done(item.thinking_chars)
                        thinking_chars_total = item.thinking_chars
                        files_touched_total = item.files_touched
                        if subagent is not None and self._session is not None:
                            append_subagent_done(
                                self._session, done_summary,
                                bg_color=NAMESPACE_COLORS[subagent.namespace], turn=turn_id,
                            )
                    elif isinstance(item, MaxIterationsEvent):
                        max_iter_hit = True
                    elif isinstance(item, BudgetExhaustedEvent):
                        budget_exhausted_hit = True

        harness_outcome = "max_iterations" if (max_iter_hit and not answer_chunks) else "ok"
        events.emit(
            "harness", session=session_id, turn=turn_id, step=step_index, outcome=harness_outcome,
            llm_calls=llm_calls, prompt_tokens=prompt_tokens_total,
            completion_tokens=completion_tokens_total, thinking_chars=thinking_chars_total,
            tools=tool_counts, duration_ms=_ms(time.monotonic() - harness_start),
            budget_exhausted=budget_exhausted_hit,
        )
        subagent_result = SubagentResult(
            summary="".join(answer_chunks).rstrip(),
            files_touched=files_touched_total,
            status="failed" if harness_outcome == "max_iterations" else "ok",
            budget_exhausted=budget_exhausted_hit,
        )
        if subagent is not None:
            events.emit(
                "delegation", session=session_id, turn=turn_id, step=step_index,
                host="main", delegate=subagent.name, namespace=subagent.namespace,
                status=subagent_result.status, files=len(subagent_result.files_touched),
                files_touched=subagent_result.files_touched[:_DELEGATION_FILES_CAP],
                summary_len=len(subagent_result.summary),
                budget_exhausted=subagent_result.budget_exhausted,
            )

        return _StepResult(
            outcome=harness_outcome,
            answer=subagent_result.summary,
            max_iter_hit=max_iter_hit,
            query_tool_count=query_tool_count,
            ws_renderer=ws_renderer,
        )

    async def _stream(self, user_input: str) -> None:
        start = time.monotonic()
        events = self._agent.events
        turn_id = events.new_turn()
        session_id = self.session_id
        events.emit("turn.start", session=session_id, turn=turn_id, input_len=len(user_input))
        outcome = "ok"
        stage = ["route"]
        verb = random_operative_verb()
        color = random_accent_color()
        conversation = self.query_one("#conversation", ScrollableContainer)
        ws_renderer: SubAgentRenderer | None = None

        try:
            await self._start_status_animation(verb[0], color)
            self._set_route_label("route", color=_PIPELINE_COLOR)
            t0 = time.monotonic()
            route = await self._agent.route(user_input, history=self._session.messages)
            events.emit("route", session=session_id, turn=turn_id, decision=_route_decision(route), duration_ms=_ms(time.monotonic() - t0))

            if route.plan is not None:
                if self._agent.debug:
                    append_debug(self._session, {"content": {"plan": [
                        {"agent": (step.subagent.name if step.subagent is not None else "main"), "raw": step.raw}
                        for step in route.plan
                    ]}})
                n = len(route.plan)
                completed = 0
                failure_kind: str | None = None
                failure_reason = ""
                for i, step in enumerate(route.plan, start=1):
                    try:
                        step_result = await self._run_step(
                            step.raw, step.subagent,
                            turn_id=turn_id, session_id=session_id, conversation=conversation,
                            stage=stage, append_user=(i == 1), step_index=i,
                        )
                    except Exception as step_error:
                        failure_kind = "error"
                        failure_reason = str(step_error) or type(step_error).__name__
                        events.emit(
                            "error", level="warning", session=session_id, turn=turn_id, step=i,
                            stage=stage[0], error_type=type(step_error).__name__, message=failure_reason,
                        )
                        break
                    ws_renderer = step_result.ws_renderer
                    if step_result.outcome == "max_iterations" and not step_result.answer:
                        failure_kind = "max_iterations"
                        failure_reason = "hit iteration limit without producing a response"
                        break
                    completed = i
                    if step_result.answer:
                        self._assistant_widget = MessageWidget(MessageKind.ASSISTANT, step_result.answer)
                        await conversation.mount(self._assistant_widget)
                        conversation.scroll_end(animate=False)

                if failure_kind is not None:
                    outcome = failure_kind
                    msg = f"step {completed + 1} of {n} failed: {failure_reason}; completed steps 1..{completed}"
                    await conversation.mount(MessageWidget(MessageKind.ERROR, msg))
                    append_event(self._session, msg, source="plan")
                else:
                    outcome = "ok"
                    elapsed = time.monotonic() - start
                    operation_text = f"* {verb[1]} for {_fmt_duration(elapsed)} ({n} steps)"
                    await conversation.mount(MessageWidget(MessageKind.OPERATION, operation_text, color=color))
                    if self._session is not None:
                        append_operation(self._session, operation_text, color, turn=turn_id)
                conversation.scroll_end(animate=False)
                return

            step_result = await self._run_step(
                user_input, route.subagent,
                turn_id=turn_id, session_id=session_id, conversation=conversation,
                stage=stage, trivial=route.trivial, explore=route.explore,
            )
            ws_renderer = step_result.ws_renderer

            if step_result.outcome == "max_iterations":
                outcome = "max_iterations"

            if self._session is not None:
                self.query_one("#context-bar", Static).update(
                    _fmt_status_bar(self._agent.model, self._working_dir.name, self._branch, _estimate_session_tokens(self._session), self._context_limit)
                )
            if step_result.max_iter_hit and not step_result.answer:
                await conversation.mount(MessageWidget(MessageKind.ERROR, "agent hit iteration limit without producing a response"))
            else:
                answer = step_result.answer
                self._assistant_widget = MessageWidget(MessageKind.ASSISTANT, answer)
                await conversation.mount(self._assistant_widget)
                elapsed = time.monotonic() - start
                operation_text = f"* {verb[1]} for {_fmt_duration(elapsed)}" + (f" ({step_result.query_tool_count} {'tool' if step_result.query_tool_count == 1 else 'tools'})" if step_result.query_tool_count > 0 else "")
                await conversation.mount(MessageWidget(MessageKind.OPERATION, operation_text, color=color))
                if self._session is not None:
                    append_operation(self._session, operation_text, color, turn=turn_id)
            conversation.scroll_end(animate=False)
        except Exception as error:
            error_msg = str(error) or type(error).__name__
            await conversation.mount(MessageWidget(MessageKind.ERROR, error_msg))
            append_event(self._session, error_msg, source="error")
            events.emit("error", level="warning", session=session_id, turn=turn_id, stage=stage[0], error_type=type(error).__name__, message=error_msg)
            outcome = "error"
            conversation.scroll_end(animate=False)
        finally:
            self._set_route_label("waiting", color=_DEFAULT_ROUTE_COLOR)
            await self._stop_status_animation()
            if ws_renderer is not None:
                ws_renderer.stop_spinner()
            if self._worker_cancelled:
                self._worker_cancelled = False
                outcome = "interrupted"
                msg = self._permission_denied_msg or ""
                self._permission_denied_msg = None
                await conversation.mount(MessageWidget(MessageKind.INTERRUPTED, msg))
                append_event(self._session, msg, source="interrupted")
                conversation.scroll_end(animate=False)
            events.emit("turn.end", session=session_id, turn=turn_id, outcome=outcome, duration_ms=_ms(time.monotonic() - start))
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

    def _quit(self, *, had_prior: bool | None = None) -> None:
        """Single end-of-app exit path — used by both `/exit` and the
        ctrl+c / ctrl+q quit bindings, so farewell + resume-hint printing
        in main.py (gated on exit_reason == "command") behaves identically."""
        self._exit_reason = "command"
        prior = self.session_has_interactions if had_prior is None else had_prior
        if self._session is not None and not prior:
            from agent.persistence import session_file
            session_file(self._session).unlink(missing_ok=True)
        self.exit()

    def action_quit(self) -> None:
        self._quit()

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
