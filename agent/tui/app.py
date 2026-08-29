from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path

from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.geometry import Region
from textual.message import Message
from textual.strip import Strip
from textual.widgets import ProgressBar, Static, TextArea
from textual.worker import Worker

from agent import compact, credentials
from agent.agent import GekaiAgent
from agent.directive_audit import AuditVerdict
from agent.harness import turn as harness_turn
from agent.commands.registry import CommandRegistry
from agent.diff import DiffLine
from agent.llm.resolve import TierResolutionError, resolve_touchpoint
from agent.llm.tiers import ModelCatalogEntry, TierBinding, TierName
from agent.persistence import (
    append_command,
    append_compact,
    append_diff,
    append_event,
    append_operation,
    _normalize_path,
)
from agent.session import Session
from agent.subagents import NAMESPACE_COLORS, SUBAGENTS, Subagent
from agent.settings import (
    PERMISSION_CHOICES,
    load_context_limit,
    load_model_catalog,
    load_tier_bindings,
    resolve_permissions,
    save_permissions,
    save_tier_binding,
    tiers_configured,
)
from agent.workspace import list_files, list_dirs
from agent.tui.styles import OPERATIVE_COLOR, _OPERATIVE_VERB
from agent.events import AgentEvent, SubAgentStartEvent, LogEvent, DiffEvent, InferEndEvent, DoneEvent, StatusUpdateEvent, TextChunkEvent, ThinkingTokenEvent, DelegationStartEvent, DelegationDoneEvent, TaskGraphHaltedEvent
from agent.tools.catalog import EDIT_TOOLS, FS_TOOLS, READ_TOOLS, SHELL_TOOLS

from .palette import CommandPalette
from .history import PromptHistory
from .widgets import ChoiceBar, DiffWidget, FilePanel, HistoryPanel, MessageKind, MessageWidget, ModelRowView, ModelsPanel, WelcomeOverlay


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
_THINKING_LINE_CAP = 80  # max characters of the rolling thinking-buffer tail shown on the shared line
_THINKING_LINE_SENTENCE_RESET = 5  # completed-step count that triggers a full wipe/restart of the block
_DIAMOND = "◆"
_SQUARE = "■"
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")


def _tool_kind_color(tool_name: str) -> str:
    """green = read, red = write (edit + filesystem mutation), yellow = shell."""
    if tool_name in READ_TOOLS:
        return "green"
    if tool_name in EDIT_TOOLS or tool_name in FS_TOOLS:
        return "red"
    if tool_name in SHELL_TOOLS:
        return "yellow"
    return "#666666"


def _split_thinking_steps(buffer: str) -> tuple[list[str], str]:
    """Split a thinking-token buffer into completed "steps" — each ending in
    `.`/`!`/`?` — and the remaining in-progress tail (not yet a full step)."""
    steps: list[str] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(buffer):
        step = buffer[start:m.end()].strip()
        if step:
            steps.append(step)
        start = m.end()
    return steps, buffer[start:]


class SubAgentRenderer:
    """Manages header, L-connector, token accumulation, and Done line for subagent events."""

    def __init__(self, conversation: ScrollableContainer, depth: int = 0) -> None:
        self._conversation = conversation
        self._depth = depth
        self.name: str = ""
        self._total_tokens: int = 0
        self._infer_count: int = 0
        self._start_time: float = time.monotonic()
        self._log_widgets: list[Static] = []
        self._progress_bar: ProgressBar | None = None
        self._header_widget: MessageWidget | None = None
        self._spinner_task: asyncio.Task | None = None
        self._badge_namespace: str | None = None
        self._badge_color: str = ""
        self._tool_calls: int = 0
        self._last_tool_name: str = ""

    async def _mount(self, widget: Static | MessageWidget | ProgressBar, *, after: Static | MessageWidget | None = None) -> None:
        """Single chokepoint every mount site routes through. Every card
        aligns to column 0 regardless of `self._depth` — nesting depth beyond
        1 is conveyed only by the "⎿" connector (`_animate_dot`/`done`), not
        by left indentation."""
        if after is not None:
            await self._conversation.mount(widget, after=after)
        else:
            await self._conversation.mount(widget)

    async def _animate_dot(self) -> None:
        frame = 0
        dot_color = self._badge_color if self._badge_namespace is not None else "#666666"
        # depth 0 is root's own header, depth 1 is a subagent's first-level
        # activation — neither draws a connector; only depth > 1 (a subagent
        # delegating to another subagent) is genuinely nested under a peer
        # badge line, so only that gets the "⎿" connector.
        prefix = "⎿ " if self._depth > 1 else ""
        try:
            while True:
                if self._header_widget is not None:
                    char = _BRAILLE_FRAMES[frame % len(_BRAILLE_FRAMES)]
                    self._header_widget.query_one(".header-dot", Static).update(f"[{dot_color}]{prefix}{char}[/{dot_color}]")
                frame += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    async def start(self, name: str, *, namespace: str | None = None, ui_label: str = "", bg_color: str = "") -> None:
        self.name = name
        self._badge_namespace = namespace
        self._badge_color = bg_color
        # Root's own plain turn (namespace is None) mounts no header of its
        # own — the shared `_ThinkingLine` above already shows "Thinking..."
        # then "Thought for ..." for it, and a second "Triaging..."/"Thought
        # for X" header here would just duplicate that. The lead-in spacer
        # only exists to set a badge header apart, so it moves inside this
        # branch too — otherwise it strands a stray blank row between the
        # shared thinking line and the answer, on top of the answer's own
        # `margin-top: 1` (MessageWidget.assistant, agent/tui/widgets.py).
        if namespace is not None:
            await self._mount(Static("", classes="assistant-spacer"))
            header_markup = _subagent_header_markup(name, bg_color, ui_label)
            widget = MessageWidget(MessageKind.HEADER, header_markup, nested=self._depth > 1)
            await self._mount(widget)
            self._header_widget = widget
            self._spinner_task = asyncio.create_task(self._animate_dot())

    async def log(self, message: str, tool_name: str = "") -> None:
        if tool_name:
            self._tool_calls += 1
        if message.endswith("..."):
            return
        marker = f"[{_tool_kind_color(tool_name)}]{_SQUARE}[/{_tool_kind_color(tool_name)}]" if tool_name else " "
        if tool_name:
            # kind (the verb, e.g. "Run") vs detail (its argument, e.g. the
            # command/path) — split via Text.append with explicit styles
            # (not markup interpolation) so a detail containing "[" (a
            # plausible path/command fragment) can never be misparsed as a
            # markup tag, same reasoning as the shared thinking-line rendering.
            # Not gated on a debug flag: that gate existed only to skip the
            # old "(N calls)" aggregation, which no longer exists — every
            # call gets its own line now, styled the same way in every mode.
            kind, _, detail = message.partition(" ")
            line = Text.from_markup(f"{marker} ")
            line.append(kind, style="#666666")
            if detail:
                line.append(f" {detail}", style="white")
            if tool_name == self._last_tool_name and self._log_widgets:
                self._log_widgets[-1].update(line)
                self._conversation.scroll_end(animate=False)
                return
            widget = Static(line)
        else:
            widget = Static(f"{marker} {message}")
        await self._mount(widget)
        self._log_widgets.append(widget)
        self._last_tool_name = tool_name
        self._conversation.scroll_end(animate=False)

    async def status_update(self, event: "StatusUpdateEvent") -> None:
        if self._progress_bar is None:
            bar = ProgressBar(total=event.total, show_eta=False, show_percentage=True, classes="subagent-progress")
            await self._mount(bar)
            self._progress_bar = bar
            self._conversation.scroll_end(animate=False)
        elif event.total is not None:
            self._progress_bar.update(total=event.total, progress=event.progress)

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
                dot_prefix = "⎿ " if self._depth > 1 else ""
                self._header_widget.query_one(".header-dot", Static).update(f"[{self._badge_color}]{dot_prefix}●[/{self._badge_color}]")
                self._header_widget = None
            # badge header persists untouched — mount the Done summary as a
            # permanent line beneath it. Depth > 1 (true nested delegation,
            # a subagent delegating to another subagent) still anchors an
            # L-connector to its parent badge; depth 1 (first-level
            # activation, root's own subagent) has no peer badge line to
            # connect to, so no connector.
            done_prefix = "  ⎿ " if self._depth > 1 else "  "
            await self._mount(Static(f"{done_prefix}Done ({summary})"))
            self._conversation.scroll_end(animate=False)
            return summary
        else:
            summary = self.turn_summary_text()
            if self._header_widget is not None:
                self._header_widget.query_one(".header-dot", Static).update(f"[white]{_DIAMOND}[/white]")
                self._header_widget.query_one(".header-text", Static).update(f"[#666666]Thought for {summary}[/#666666]")
                self._header_widget = None
            self._conversation.scroll_end(animate=False)
            return summary

    def turn_summary_text(self) -> str:
        """Root-style `"{elapsed} · {tokens} tokens · {N} call(s)"` summary
        computed from this renderer's tracked `_start_time`/`_total_tokens`/
        `_infer_count` — the same fields regardless of badge status, so this
        is also what the shared per-turn thinking line uses to render its
        final `"Thought for ..."` text even when `ws_renderer` itself has a
        badge (explicit `/alias` seed dispatch)."""
        elapsed = time.monotonic() - self._start_time
        parts = [_fmt_duration_verbose(elapsed)]
        if self._total_tokens > 0:
            parts.append(f"{_fmt_tokens(self._total_tokens)} tokens")
        if self._infer_count > 0:
            calls = f"{self._infer_count} call" + ("s" if self._infer_count != 1 else "")
            parts.append(calls)
        return " · ".join(parts)


class _ThinkingLine:
    """The turn's single shared thinking-preview block. Unlike a
    `SubAgentRenderer`'s own header/badge (one per renderer), there is
    exactly one of these per turn — mounted once at the very top of the
    turn's output before any renderer can mount anything else, and fed by
    every `ThinkingTokenEvent` regardless of which renderer (root, subagent,
    nested delegation) is currently active. Renders as up to
    `_THINKING_LINE_SENTENCE_RESET` stacked lines — each completed sentence
    becomes its own line, the newest (still in-progress) one carries the
    spinner dot — and once a line beyond that count would be needed, the
    whole block wipes and restarts from a bare "Thinking...". Built as a
    plain `Static`, not a `MessageWidget`, so it stays invisible to every
    `isinstance(w, MessageWidget)` filter the rest of the app/tests use to
    count header lines.

    Kept pinned to the top of the visible viewport while the turn is active
    by re-scrolling `conversation` to this widget's own top on every redraw
    (`scroll_to_widget(..., top=True)`) — not a second widget, just active
    scroll management on the one in-flow copy — so it stays in view even as
    the turn's own subagent cards/logs grow beneath it."""

    def __init__(self, conversation: ScrollableContainer) -> None:
        self._conversation = conversation
        self._widget: Static | None = None
        self._buffer: str = ""
        self._frame: int = 0
        self._spinner_task: asyncio.Task | None = None

    async def mount(self) -> None:
        await self._conversation.mount(Static("", classes="assistant-spacer"))
        self._widget = Static(self._render())
        await self._conversation.mount(self._widget)
        self._pin_to_top()
        self._spinner_task = asyncio.create_task(self._animate())

    async def _animate(self) -> None:
        try:
            while True:
                self._frame += 1
                self._redraw()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def update_chunk(self, text: str) -> None:
        self._buffer += text
        steps, _ = _split_thinking_steps(self._buffer)
        if len(steps) >= _THINKING_LINE_SENTENCE_RESET:
            # Full cycle wipe: once a 5th line would be needed, clear
            # everything and restart from "Thinking..." rather than scrolling.
            self._buffer = ""
        self._redraw()

    @staticmethod
    def _cap(text: str) -> str:
        tail = text[-_THINKING_LINE_CAP:]
        ellipsis = "…" if len(text) > _THINKING_LINE_CAP else ""
        return f"{ellipsis}{tail}"

    def _render(self) -> Text:
        # One Text redraw per frame (dot + lines together) — this widget is
        # a plain Static, not a MessageWidget, so unlike SubAgentRenderer's
        # header it has no separate `.header-dot`/`.header-text` to update
        # independently. Completed sentences stack as lines above the
        # current in-progress one; the dot always marks the active (last)
        # line, same dim color throughout.
        dot_char = _BRAILLE_FRAMES[self._frame % len(_BRAILLE_FRAMES)]
        steps, tail = _split_thinking_steps(self._buffer)
        lines = [*steps, tail] if tail else list(steps)
        text = Text(no_wrap=True)
        if not lines:
            text.append(dot_char, style="#666666")
            text.append(" Thinking...", style="#666666")
            return text
        # rich.text.Text, never a markup string, so a literal "[" streamed
        # by the model can never be misparsed as a markup tag — same
        # reasoning as `SubAgentRenderer.log()`.
        for i, line in enumerate(lines):
            if i > 0:
                text.append("\n")
            prefix = dot_char if i == len(lines) - 1 else " "
            text.append(f"{prefix} {self._cap(line)}", style="#666666")
        return text

    def _redraw(self) -> None:
        if self._widget is not None:
            self._widget.update(self._render())
        self._pin_to_top()

    def _pin_to_top(self) -> None:
        # Fired on every 0.1s animation tick, so any scroll-to-bottom another
        # event triggers in between (e.g. `DiffEvent`'s `scroll_end`) is
        # corrected back within one frame.
        if self._widget is not None:
            self._conversation.scroll_to_widget(self._widget, animate=False, top=True)

    def stop_spinner(self) -> None:
        if self._spinner_task is not None:
            self._spinner_task.cancel()
            self._spinner_task = None

    def finish(self, summary_text: str) -> None:
        self.stop_spinner()
        if self._widget is not None:
            line = Text(no_wrap=True)
            line.append(_DIAMOND, style="white")
            line.append(f" {summary_text}", style="#666666")
            self._widget.update(line)


def _subagent_header_markup(name: str, bg_color: str, ui_label: str) -> str:
    markup = f"[black on {bg_color} bold] {name} [/]"
    if ui_label:
        markup += f"[white]\\[{ui_label}][/white]"
    return markup


_REQUEST_SUMMARY_BLOCK = re.compile(r"<request_summary>.*?</request_summary>\s*", re.DOTALL)


def _fallback_ui_label(text: str, max_words: int = 12) -> str:
    """Cheap stand-in for the rewriter's <ui_label> when rewriting is skipped
    (no located entries) or the label comes back empty — strips backtick-quoted
    paths so file names don't leak into the badge, then takes the leading words.
    Also strips a leading <request_summary> framing block (task graph steps are
    prefixed with one before dispatch) so it never leaks into the badge."""
    text = _REQUEST_SUMMARY_BLOCK.sub("", text, count=1)
    stripped = re.sub(r"`[^`]*`", "", text)
    words = stripped.split()
    return " ".join(words[:max_words])


def _resolve_at_refs(text: str) -> str:
    return re.sub(r"@(\S+)", lambda m: f"`{m.group(1)}`", text)


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


_MASK_STARS = 6  # fixed width — a long key must not blow up the row with 1-for-1 stars


def _mask_key(key: str) -> str:
    """Display form of a real key value: first 2 + last 3 characters kept,
    the middle replaced with a fixed-width run of '*' (never 1-for-1 with
    true length — some keys are long enough that would dominate the row).
    Keys of 5 characters or fewer are too short for head/tail to mean
    anything distinct, so they're masked in full at the same fixed width."""
    if len(key) <= 5:
        return "*" * _MASK_STARS
    return key[:2] + "*" * _MASK_STARS + key[-3:]


def _models_display_key(cred_key: str, key_input: dict[str, str]) -> str:
    """Non-editing display for the key cell: an in-progress edit for this
    model (`key_input`, including an explicit clear stored as "") always wins
    over whatever's actually in the keyring."""
    if cred_key in key_input:
        value = key_input[cred_key]
        return _mask_key(value) if value else "no key"
    if credentials.has_api_key(cred_key):
        return _mask_key(credentials.get_api_key(cred_key))
    return "no key"


def _models_key_present(cred_key: str, key_input: dict[str, str]) -> bool:
    """Whether this model currently has a usable key once this edit lands —
    an in-progress edit (including an explicit clear) wins over the real
    stored credential, same precedence as `_models_display_key`."""
    if cred_key in key_input:
        return bool(key_input[cred_key])
    return credentials.has_api_key(cred_key)


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


def _fmt_tokens_k(n: int) -> str:
    return f"{n / 1000:.1f}k"


_PATH_TRUNCATE_BUDGET = 30


def _truncate_path_middle(path: str, budget: int = _PATH_TRUNCATE_BUDGET) -> str:
    if len(path) <= budget:
        return path
    head_len = budget // 2 - 1
    tail_len = budget - head_len - 1
    return f"{path[:head_len]}…{path[-tail_len:]}"


def _fmt_status_left(
    model: str, effort: str | None, session_tokens: int, other_tokens: int, prompt_tokens: int, limit: int,
) -> str:
    model_label = f"{model} ({effort})" if effort else model
    tokens = f"{_fmt_tokens_k(session_tokens)} · {_fmt_tokens_k(other_tokens)} tokens"
    pct = _fmt_context_pct(prompt_tokens, limit)
    return f"\\[{model_label}] | {tokens} | {pct}"


def _fmt_status_right(working_dir: str, branch: str | None) -> str:
    location = f"📁 {_truncate_path_middle(working_dir)}"
    if branch:
        location += f" | ⎇ {branch}"
    return location


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
    """Outcome of one dispatch pipeline run for a single step."""
    outcome: str = "ok"  # "ok" | "max_iterations"
    answer: str = ""
    max_iter_hit: bool = False
    budget_exhausted: bool = False
    review_report: str | None = None
    query_tool_count: int = 0
    ui_label: str = ""
    ws_renderer: SubAgentRenderer | None = None


@dataclass
class _ModelsEdit:
    """Mutable staging area for an in-progress `/models` edit. Mutated
    synchronously by the existing key-event action cascade (`action_navigate_*`,
    `action_confirm_or_submit`, `action_cancel_stream`) — no worker, no
    future-based primitive. Nothing here reaches the keyring until `[ok]`
    (see `GekaiApp._commit_models_edit`)."""
    entries: list[ModelCatalogEntry]  # one grid row each, in catalog order
    key_input: dict[str, str]  # credential_key(provider, model) -> new value staged this session ("" means explicit clear); NOT yet written to keyring
    key_editing: bool = False  # the currently-selected key cell is in free-text edit mode
    key_edit_buffer: str = ""  # raw text typed/pasted so far while key_editing — always starts empty


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
    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("escape", "cancel_stream", "Cancel"),
        ("ctrl+c", "quit", "Quit"),
        Binding("ctrl+up", "scroll_to_top", "Scroll to top", priority=True),
        Binding("ctrl+b", "scroll_to_end", "Scroll to bottom", priority=True),
        Binding("pageup", "scroll_page_up", "Scroll page up", priority=True),
        Binding("pagedown", "scroll_page_down", "Scroll page down", priority=True),
        Binding("up", "navigate_up", show=False, priority=True),
        Binding("down", "navigate_down", show=False, priority=True),
        Binding("left", "navigate_left", show=False, priority=True),
        Binding("right", "navigate_right", show=False, priority=True),
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
        self._streamed_answer: str = ""
        self._status_task: asyncio.Task[None] | None = None
        self._status_stop: asyncio.Event | None = None
        self._status_frame: int = 0
        self._status_verb: str = ""
        self._status_start: float = 0.0
        self._esc_pending: bool = False
        self._pending_choice: asyncio.Future[str | None] | None = None
        self._context_limit: int = 128_000
        self._other_ops_tokens: int = 0
        self._compact_warning_shown: bool = False
        self._compact_failure_count: int = 0
        self._history: PromptHistory | None = None
        self._file_paths: list[str] | None = None
        self._file_at_pos: int = -1
        self._models_edit: _ModelsEdit | None = None
        self._worker_cancelled: bool = False
        self._permission_denied_msg: str | None = None
        self._status_paused: bool = False
        self._welcome_dismissed: bool = False
        self._exit_reason: str = "quit"
        self._invocable_subagents: dict[str, Subagent] = {
            p.alias or p.name: p for p in SUBAGENTS if p.user_invocable
        }
        super().__init__(**kwargs)
        self.ansi_color = True

    def compose(self) -> ComposeResult:
        yield ConversationContainer(id="conversation")
        with Container(id="footer"):
            yield ChoiceBar(id="choice-bar")
            yield Static("", id="status-line")
            yield Static("", id="status-spacer")
            yield CommandPalette(
                self._command_registry,
                subagents=[(p.alias or p.name, p.short_description) for p in SUBAGENTS if p.user_invocable],
                id="command-palette",
            )
            yield Static("", id="hint-area")
            with Container(id="scroll-hint-wrap"):
                yield Static("Scroll to bottom (ctrl+B) ↓", id="scroll-hint")
            yield FilePanel(id="file-panel")
            yield HistoryPanel(id="history-panel")
            yield ModelsPanel(id="models-panel")
            yield Static("", id="copy-notice")
            yield Static("", id="directive-notice")
            yield Static("", id="param-hint")
            with Container(id="input-area"):
                yield PromptTextArea(id="prompt", show_line_numbers=False, compact=True, highlight_cursor_line=False)
                yield Static("❯", id="prompt-marker")
            yield Static("", id="context-bar")

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
        asyncio.create_task(self._poll_prompt_lock())
        self.run_worker(self._init_session(), exclusive=True)

    async def _init_session(self) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)

        self._hide_directive_notice()
        self._session = self._agent.start_session(
            restored_messages=self._restored_messages,
            session_id=self._restored_id,
        )
        self._agent.events.emit("session.start", session=self._session.id, resumed=self._restored_id is not None)
        # ponytail: audit still fires unconditionally (it's the source of truth
        # for whether project instructions are actually in the system base) —
        # only the visible notice is skipped on restore, since GEKAI.md was
        # already shown loaded before the app closed and re-flashing it reads
        # as a reload that never happened.
        on_verdict = None if self._restored_id is not None else self._apply_directive_verdict
        self._agent.start_directive_audit(self._session, on_verdict)

        history_path = (
            Path.home() / ".gekai" / "workspaces"
            / _normalize_path(self._working_dir) / "history.jsonl"
        )
        self._history = PromptHistory(history_path)

        override = load_context_limit(self._working_dir)
        self._context_limit = override if override is not None else _context_limit(self._agent.model)
        self._refresh_status_bar()

        if self._restored_timeline:
            last_subagent_header: MessageWidget | None = None
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
                    await conversation.mount(Static(f"  Done ({entry.get('summary', '')})"))
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

        await self._maybe_warn_tiers_unconfigured(conversation)

        if self._restored_id is None:
            await self.mount(WelcomeOverlay(id="welcome-overlay"))

    def _refresh_status_bar(self) -> None:
        session_tokens = _estimate_session_tokens(self._session) if self._session is not None else 0
        state = compact.context_state(session_tokens, self._context_limit)
        if state == "warn" or state == "auto":
            pct = round(session_tokens / self._context_limit * 100, 1)
            self._show_hint(f"context {pct}% — auto-compact will trigger at 80%")
            self._compact_warning_shown = True
        elif self._compact_warning_shown:
            self._clear_hint()
            self._compact_warning_shown = False
        left = _fmt_status_left(
            self._agent.model, self._agent.effort, session_tokens, self._other_ops_tokens,
            session_tokens, self._context_limit,
        )
        right = _fmt_status_right(str(self._working_dir), self._branch)
        left_text = Text.from_markup(left)
        right_text = Text(right)
        width = self.size.width or 80
        gap = max(width - 4 - left_text.cell_len - right_text.cell_len, 1)
        bar = Text("  ") + left_text + Text(" " * gap) + right_text + Text("  ")
        bar.no_wrap = True
        bar.overflow = "crop"
        self.query_one("#context-bar", Static).update(bar)

    async def _maybe_warn_tiers_unconfigured(self, conversation: ScrollableContainer) -> None:
        """Nudge, not a gate: `GekaiAgent` tolerates unconfigured tiers at
        construction time (so `/models` and `/tier` stay reachable — see
        `agent/agent.py`'s `_tier_error` deferral) but every touchpoint raises
        the moment it's actually used. This just keeps surfacing that the
        tiers haven't been set up yet, both at startup and on every prompt
        submitted while they stay unset."""
        if tiers_configured():
            return
        await conversation.mount(MessageWidget(
            MessageKind.WARNING,
            "model tiers are not configured — run /models to store API keys, then /tier to assign FAST/SUPP/CORE",
        ))
        conversation.scroll_end(animate=False)

    async def _clear_session(self, command_text: str | None = None) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)
        await conversation.remove_children()
        self._hide_directive_notice()
        self._session = self._agent.start_session()
        self._agent.events.emit("session.start", session=self._session.id, resumed=False)
        self._agent.start_directive_audit(self._session, self._apply_directive_verdict)
        self._other_ops_tokens = 0
        self._refresh_status_bar()
        self._assistant_widget = None
        self._streamed_answer = ""
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

    def _params_for(self, slash_name: str) -> str | None:
        """None = unknown command name. "" = no params. Else the param spec string."""
        subagent = self._invocable_subagents.get(slash_name)
        if subagent is not None:
            return subagent.params
        command = self._command_registry.get(slash_name)
        if command is not None:
            return command.params
        return None

    def _render_param_hint(self) -> None:
        prompt_text = self.query_one("#prompt", TextArea).text
        hint = self.query_one("#param-hint", Static)
        if prompt_text.startswith("/"):
            name = prompt_text[1:].split(None, 1)[0] if prompt_text[1:].split() else ""
            params = self._params_for(name)
            if params:
                hint.update(f"/{name} {params}")
                hint.display = True
                return
        hint.update("")
        hint.display = False

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

        self._render_param_hint()

    def on_key(self, event: events.Key) -> None:
        # ChoiceBar's left/right cursor movement used to be special-cased
        # here, but `action_navigate_left`/`action_navigate_right` (new
        # `priority=True` App bindings) now claim "left"/"right" before this
        # handler would ever see them — see those actions' own comment.
        if self._esc_pending and event.key != "escape":
            self._clear_hint()

        # `/models`' key column, while in edit mode (see `_handle_models_enter`):
        # "p" is the *only* way to fill the buffer — it reads the OS
        # clipboard directly (ctypes, same as `_read_clipboard_text`
        # elsewhere — no reliance on "ctrl+v"/the terminal's bracketed-paste
        # support, which doesn't reach this app) and commits straight into
        # `key_input` on success (single press, no separate confirm step to
        # forget before navigating away). Must
        # be checked before the auto-focus-and-type fallback below, which
        # would otherwise route this keystroke into the chat prompt instead.
        edit = self._models_edit
        if edit is not None and edit.key_editing:
            if event.key == "p":
                clipboard_text = self._read_clipboard_text()
                # Multiple lines aren't rejected — just take the first one,
                # since a key is never legitimately multi-line and this is
                # by far the more common paste artifact (trailing newline,
                # a whole file's worth of clipboard, etc.) than an actual
                # mistake worth blocking on.
                first_line = clipboard_text.splitlines()[0] if clipboard_text else ""
                if not first_line:
                    self._show_hint("clipboard is empty — copy a key first")
                    self.set_timer(2.0, self._clear_hint)
                    event.stop()
                    return
                panel = self.query_one(ModelsPanel)
                row, _ = panel.selected_cell
                if row < len(edit.entries):
                    entry = edit.entries[row]
                    edit.key_input[credentials.credential_key(entry.provider, entry.name)] = first_line
                edit.key_editing = False
                self._render_models_panel()
                event.stop()
                return
            # Any other key (manual typing, no longer supported) falls
            # through: `_any_panel_active()` below already keeps it from
            # reaching the chat prompt, and letting it bubble un-stopped
            # means "escape" (not a priority binding, unlike enter/arrows)
            # still reaches `action_cancel_stream` to back out of the edit.

        prompt = self.query_one("#prompt", TextArea)
        if event.key == "ctrl+j" and prompt.has_focus:
            prompt.insert("\n")
            event.stop()
            return
        # Bracketed paste never reaches this app (see the `/models` key-field
        # comment above) — Textual has no `Paste` message to fall back on,
        # so `ctrl+v` does nothing unless handled explicitly here. Reads the
        # OS clipboard directly (same ctypes helper the key field uses) and
        # inserts at the cursor; auto-focuses the prompt first if it wasn't
        # already focused, mirroring the printable-character fallback below.
        if event.key == "ctrl+v" and not self._any_panel_active():
            clipboard_text = self._read_clipboard_text()
            if clipboard_text:
                if not prompt.has_focus:
                    prompt.focus(scroll_visible=False)
                prompt.insert(clipboard_text)
            event.stop()
            return
        # `.focus()`/`.insert()` are programmatic API calls that work even
        # while `read_only=True` (Textual only blocks the widget's own
        # keyboard-driven edit actions) — so this auto-focus-and-type
        # fallback must check panel state itself, not just `has_focus`,
        # or a panel being open wouldn't actually keep the prompt inert.
        if prompt.has_focus or not event.is_printable or self._any_panel_active():
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
    ) -> str | None:
        loop = asyncio.get_event_loop()
        self._pending_choice = loop.create_future()
        self.query_one(ChoiceBar).show(question, options, 0)
        return await self._pending_choice

    _TIER_WIZARD_BACK = "__back__"

    async def _run_tier_wizard(self, tier: TierName, conversation: ScrollableContainer) -> None:
        """`/tier <TIER>`'s real implementation: a sequence of `_ask_choice`
        screens (the same mechanism the first-run permissions prompt uses),
        not a persistent grid — model, then effort, then thinking (only when
        `tier` is CORE and the picked model supports it), then a confirm
        screen. Every screen but the first prepends a "‹ back" option so the
        whole sequence is freely re-walkable; only the confirm step actually
        writes anything, and only on "confirm"."""
        catalog = load_model_catalog()
        keyed = [
            entry for entry in catalog.values()
            if credentials.has_api_key(credentials.credential_key(entry.provider, entry.name))
        ]
        if not keyed:
            await conversation.mount(MessageWidget(
                MessageKind.ERROR, "no models have a stored key — run /models first",
            ))
            conversation.scroll_end(animate=False)
            return

        step = "model"
        model: str | None = None
        effort: str | None = None
        thinking = False
        thinking_applicable = False

        while True:
            if step == "model":
                choice = await self._ask_choice(
                    f"{tier.value.upper()}: pick a model",
                    [(e.name, e.name) for e in keyed],
                )
                if choice is None:
                    break
                model, step = choice, "effort"
                continue

            entry = catalog[model]  # type: ignore[index]  # `model` is set on every step past "model"

            if step == "effort":
                choice = await self._ask_choice(
                    f"{tier.value.upper()}: pick an effort for {model}",
                    [(self._TIER_WIZARD_BACK, "‹ back")] + [(e, e) for e in entry.efforts],
                )
                if choice is None:
                    break
                if choice == self._TIER_WIZARD_BACK:
                    step = "model"
                    continue
                effort = choice
                thinking_applicable = tier is TierName.CORE and entry.thinking
                thinking = False
                step = "thinking" if thinking_applicable else "confirm"
                continue

            if step == "thinking":
                choice = await self._ask_choice(
                    f"{tier.value.upper()}: enable thinking for {model}?",
                    [(self._TIER_WIZARD_BACK, "‹ back"), ("y", "Yes"), ("n", "No")],
                )
                if choice is None:
                    break
                if choice == self._TIER_WIZARD_BACK:
                    step = "effort"
                    continue
                thinking = choice == "y"
                step = "confirm"
                continue

            # step == "confirm"
            choice = await self._ask_choice(
                f"{tier.value.upper()}: {model} (effort={effort}, thinking={thinking}) — save?",
                [(self._TIER_WIZARD_BACK, "‹ back"), ("confirm", "Confirm"), ("cancel", "Cancel")],
            )
            if choice is None or choice == "cancel":
                break
            if choice == self._TIER_WIZARD_BACK:
                step = "thinking" if thinking_applicable else "effort"
                continue

            assert model is not None and effort is not None  # guaranteed by the step order above
            save_tier_binding(tier, TierBinding(model=model, default_effort=effort, thinking=thinking))
            self._agent.reconfigure_touchpoints()
            summary = f"{tier.value.upper()}: {model} (effort={effort}, thinking={thinking})"
            if tiers_configured():
                # Completing the last tier is a real new-session start: GEKAI.md
                # was never actually read+audited while tiers were incomplete
                # (the directive audit needs a working touchpoint) — catch it up
                # now the same way `/clear` already does.
                await self._clear_session()
                conversation = self.query_one("#conversation", ScrollableContainer)
                await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, summary))
                conversation.scroll_end(animate=False)
                return
            self._refresh_status_bar()
            await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, summary))
            conversation.scroll_end(animate=False)
            return

        await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, "cancelled"))
        conversation.scroll_end(animate=False)

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
            cmd = palette.selected_name
            palette.hide()
            if cmd:
                if self._params_for(cmd):
                    prompt.clear()
                    prompt.text = f"/{cmd} "
                    self._prompt_move_to_end(prompt)
                    self._focus_prompt()
                    return
                stripped = f"/{cmd}"
        if not stripped:
            self._focus_prompt()
            return
        if self._history is not None:
            self._history.append(stripped)
        history_panel = self.query_one("#history-panel", HistoryPanel)
        if history_panel.display:
            history_panel.hide()
        prompt.clear()
        self._hide_directive_notice()
        conversation = self.query_one("#conversation", ScrollableContainer)
        if not tiers_configured():
            _slash_name = stripped.lstrip("/").split(None, 1)[0] if stripped.startswith("/") else ""
            if _slash_name not in ("models", "tier", "exit"):
                await conversation.mount(MessageWidget(
                    MessageKind.WARNING,
                    "model tiers are not configured — run /models to store API keys, then /tier to assign FAST/SUPP/CORE",
                ))
                conversation.scroll_end(animate=False)
                self._focus_prompt()
                return
        if stripped.startswith("/"):
            _slash_parts = stripped.lstrip("/").split(None, 1)
            _slash_name = _slash_parts[0] if _slash_parts else ""
            if _slash_name in self._invocable_subagents:
                _subagent = self._invocable_subagents[_slash_name]
                _user_prompt = _slash_parts[1].strip() if len(_slash_parts) > 1 else ""
                _params_spec = self._params_for(_slash_name) or ""
                _param_required = bool(_params_spec) and not _params_spec.startswith("<optional")
                if _param_required and not _user_prompt:
                    await conversation.mount(MessageWidget(MessageKind.ERROR, f"/{_slash_name} needs a prompt — e.g. /{_slash_name} <instructions>"))
                    conversation.scroll_end(animate=False)
                    self._focus_prompt()
                    return
                await conversation.mount(MessageWidget(MessageKind.USER, stripped))
                self._worker = self.run_worker(
                    self._stream(_resolve_at_refs(_user_prompt), forced_seed=_subagent.name),
                    exclusive=True,
                )
                return
            if _slash_name == "models":
                await conversation.mount(MessageWidget(MessageKind.USER, stripped))
                conversation.scroll_end(animate=False)
                await self._open_models_panel(conversation)
                self._focus_prompt()
                return
            if _slash_name == "tier":
                # Only a single token after the name is the wizard's
                # trigger — no args (listing) or extra tokens (a usage
                # error) fall through to the plain `TierCommand` dispatch
                # below, unchanged.
                _tier_arg = _slash_parts[1].strip() if len(_slash_parts) > 1 else ""
                if _tier_arg and " " not in _tier_arg:
                    await conversation.mount(MessageWidget(MessageKind.USER, stripped))
                    conversation.scroll_end(animate=False)
                    try:
                        _tier = TierName(_tier_arg.lower())
                    except ValueError:
                        names = "/".join(t.value.upper() for t in TierName)
                        await conversation.mount(MessageWidget(
                            MessageKind.ERROR, f"unknown tier {_tier_arg!r} — expected one of {names}"
                        ))
                        conversation.scroll_end(animate=False)
                        self._focus_prompt()
                        return
                    # Must run as a worker, not be awaited inline: this call
                    # is itself inside the Enter keybinding's own action
                    # handler, which Textual's message pump awaits directly.
                    # `_run_tier_wizard` blocks on `_ask_choice` futures that
                    # only resolve from a LATER keypress — awaited inline,
                    # the pump would never get back around to dispatch that
                    # keypress, deadlocking the whole app (confirmed: it did).
                    # `_init_session`'s identical `_ask_choice` usage avoids
                    # this the same way, via `run_worker` below.
                    self.run_worker(self._run_tier_wizard(_tier, conversation), exclusive=True)
                    self._focus_prompt()
                    return
            if _slash_name == "compact":
                if self._worker is not None and not self._worker.is_finished:
                    await conversation.mount(MessageWidget(MessageKind.ERROR, "a turn is already running — wait for it to finish before compacting"))
                    conversation.scroll_end(animate=False)
                    self._focus_prompt()
                    return
                if self._session is None:
                    await conversation.mount(MessageWidget(MessageKind.ERROR, "no active session to compact"))
                    conversation.scroll_end(animate=False)
                    self._focus_prompt()
                    return
                _user_prompt = _slash_parts[1].strip() if len(_slash_parts) > 1 else ""
                await conversation.mount(MessageWidget(MessageKind.USER, stripped))
                conversation.scroll_end(animate=False)
                await self._run_compact(conversation, _user_prompt)
                return
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
            await conversation.mount(MessageWidget(MessageKind.ERROR if result.error else MessageKind.COMMAND_RESULT, output))
            conversation.scroll_end(animate=False)
            if self._session is not None:
                append_event(self._session, output, source="command")
            if result.reconfigure:
                # `/tier` just rewrote a binding — re-resolve now rather than
                # waiting on process_stream()'s lazy self-heal, which only
                # retries while resolution is still failing (the status bar
                # would otherwise show the pre-command CORE binding until the
                # next restart).
                self._agent.reconfigure_touchpoints()
                self._refresh_status_bar()
            if result.exit_app:
                self._quit(had_prior=had_prior)
                return
            self._focus_prompt()
            return
        await conversation.mount(MessageWidget(MessageKind.USER, stripped))
        self._worker = self.run_worker(self._stream(_resolve_at_refs(stripped)), exclusive=True)

    async def _open_models_panel(self, conversation: ScrollableContainer) -> None:
        """`/models`'s real implementation: no worker, no async "flow" — just
        seeds `self._models_edit` from the catalog and renders the grid.
        Every further edit is driven synchronously by the ordinary key-event
        action cascade (`action_navigate_*`, `action_confirm_or_submit`,
        `action_cancel_stream`), the same way `FilePanel`/`HistoryPanel` are
        already driven in this file."""
        catalog = load_model_catalog()
        if not catalog:
            await conversation.mount(MessageWidget(
                MessageKind.ERROR,
                "the model catalog is empty — nothing to configure",
            ))
            conversation.scroll_end(animate=False)
            return

        self._models_edit = _ModelsEdit(entries=list(catalog.values()), key_input={})
        self._render_models_panel()

    def _render_models_panel(self) -> None:
        """Rebuild one `ModelRowView` per catalog entry from
        `self._models_edit` and re-show the panel — called after every edit
        (key set via clipboard, key cleared)."""
        edit = self._models_edit
        if edit is None:
            return
        panel = self.query_one(ModelsPanel)
        sel_row, sel_column = panel.selected_cell
        rows: list[ModelRowView] = []
        for i, entry in enumerate(edit.entries):
            cred_key = credentials.credential_key(entry.provider, entry.name)
            if edit.key_editing and i == sel_row and sel_column == "key":
                key_display = edit.key_edit_buffer or "press p to paste"
            else:
                key_display = _models_display_key(cred_key, edit.key_input)
            rows.append(ModelRowView(
                provider=entry.provider,
                model=entry.name,
                key=key_display,
                status="✓ keyed" if _models_key_present(cred_key, edit.key_input) else "no key",
            ))
        panel.show(rows)

    async def _handle_models_enter(self, panel: ModelsPanel) -> None:
        edit = self._models_edit
        if edit is None:
            return
        row, column = panel.selected_cell
        conversation = self.query_one("#conversation", ScrollableContainer)

        if row == len(edit.entries):
            if column == "cancel":
                await self._cancel_models_edit(conversation)
            elif column == "ok":
                await self._commit_models_edit(conversation)
            return

        if column == "key":
            entry = edit.entries[row]
            if edit.key_editing:
                # Confirm: an empty buffer is an explicit clear, stored as
                # "" (see `_models_display_key`/`_models_key_present`) rather
                # than leaving `key_input` untouched, which would mean "no
                # change — keep whatever's in the keyring".
                edit.key_input[credentials.credential_key(entry.provider, entry.name)] = edit.key_edit_buffer
                edit.key_editing = False
                edit.key_edit_buffer = ""
            else:
                # Always starts blank — never pre-filled with the existing
                # masked value, per the locked design.
                edit.key_editing = True
                edit.key_edit_buffer = ""
            self._render_models_panel()

    async def _cancel_models_edit(self, conversation: ScrollableContainer) -> None:
        self._models_edit = None
        self.query_one(ModelsPanel).hide()
        await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, "cancelled"))
        conversation.scroll_end(animate=False)

    async def _commit_models_edit(self, conversation: ScrollableContainer) -> None:
        edit = self._models_edit
        if edit is None:
            return

        # A model with no key is not an incomplete form — it's simply a model
        # this user doesn't use, and `/tier` refuses to bind it. So there is
        # nothing to validate here: only "was anything actually staged".
        if not edit.key_input:
            self.query_one(ModelsPanel).hide()
            self._models_edit = None
            await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, "kept actual keys"))
            conversation.scroll_end(animate=False)
            return

        # Batched at commit, not as each key is typed/pasted — the ONLY
        # place `set_api_key`/`delete_api_key` are called, per the locked
        # design. "" means an explicit clear (see `_handle_models_enter`'s
        # confirm step); a non-empty value is a real key to store.
        for cred_key, value in edit.key_input.items():
            if value:
                credentials.set_api_key(cred_key, value)
            else:
                credentials.delete_api_key(cred_key)

        summaries = [
            f"{cred_key}: {'stored' if value else 'cleared'}"
            for cred_key, value in edit.key_input.items()
        ]

        self.query_one(ModelsPanel).hide()
        self._models_edit = None
        # Re-resolve immediately rather than waiting on process_stream()'s
        # lazy self-heal (which only retries while resolution is still failing)
        # — a tier whose model just gained its key becomes resolvable now, and
        # the status bar (self._agent.model/.effort) would otherwise keep
        # showing the pre-commit state until the next restart.
        self._agent.reconfigure_touchpoints()
        self._refresh_status_bar()
        # First line sits right next to the "⎿" MessageWidget already
        # prepends (see agent/tui/widgets.py::MessageWidget._as_markup);
        # continuation lines are indented 2 spaces to land in that same
        # column, mirroring the tool-log "⎿"/"   " alignment convention
        # used elsewhere in this file.
        result_text = summaries[0] + "".join(f"\n  {line}" for line in summaries[1:])
        await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, result_text))
        conversation.scroll_end(animate=False)

    async def _run_step(
        self, raw: str, seed: str | None, *,
        turn_id: str, session_id: str, conversation: ScrollableContainer,
        stage: list[str],
        append_user: bool = True,
    ) -> _StepResult:
        """Render dispatch for `raw`. `seed` (case 3 — an explicit `/agent-x` or a
        single-duty match) names the specialist the sequencer is seeded with;
        it no longer means "run the whole turn as that subagent's identity"
        (plan 27 — the sequencer+interpreter own every spawn, root stays root).
        `stage` is a 1-element mutable holder the caller's except-block reads to attribute
        which sub-stage failed.

        Bookkeeping (token/tool counts, outcome, telemetry) lives in
        `agent.harness.turn.run_step` (plan 28 Phase 0 extraction) — this
        method only renders what that stream reports, via `_on_event`.
        """
        stage[0] = "harness"
        ws_renderer: SubAgentRenderer | None = None
        active_renderer: SubAgentRenderer | None = None
        # Stack of open delegation renderers, innermost last. A subagent P
        # dispatched via a task-graph step (or a seed) can itself delegate to
        # a child C through the `delegate` tool (agent/tools/delegate.py) —
        # both P and C emit DelegationStart/DoneEvent on the same event bus,
        # so a flat start(P) -> start(C) -> done(C) -> done(P) sequence must
        # nest, not overwrite. Correlation here is positional (push on start,
        # pop on done) rather than matched by an id, which is sound today
        # because delegation on this bus is strictly sequential on two
        # independent guarantees: `run_task_graph` (agent/harness/interpreter.py)
        # is a plain `for` loop with `await`, and the delegate tool is
        # registered `is_concurrency_safe=False`, so `ToolRegistry.run`
        # (agent/llm/tools.py) serializes it instead of gathering concurrently.
        # If either guarantee ever changes, `DelegationStarted`/`DelegationCompleted`
        # (agent/llm/events.py) already carry a `run_id` that could be threaded
        # through to `DelegationStartEvent`/`DelegationDoneEvent` for exact
        # correlation instead of positional — not needed now.
        delegation_stack: list[SubAgentRenderer] = []
        # Reset per-turn streaming state (plan 34 Phase 2): `_assistant_widget`
        # doubles as "has a TextChunkEvent already mounted the live answer
        # widget this turn" — stale state from a prior turn would otherwise
        # make `_stream()`'s finalization step below (re)use last turn's
        # widget instead of creating this turn's.
        self._assistant_widget = None
        self._streamed_answer = ""

        # One shared thinking-preview line for the whole turn, mounted here
        # — before `run_step` below can emit a single event — so it always
        # sits at the very top of the turn's output, above every
        # renderer's own header/badge, and never moves.
        thinking_line = _ThinkingLine(conversation)
        await thinking_line.mount()

        async def _on_event(item: AgentEvent | str) -> None:
            nonlocal ws_renderer, active_renderer
            if isinstance(item, str):
                return
            if isinstance(item, TaskGraphHaltedEvent):
                await conversation.mount(MessageWidget(
                    MessageKind.ERROR,
                    f"task graph halted at step {item.step_index + 1} ({item.agent}): {item.reason} "
                    "— prior steps' work is kept, nothing rolled back",
                ))
            elif isinstance(item, SubAgentStartEvent):
                ws_renderer = SubAgentRenderer(conversation)
                active_renderer = ws_renderer
                # `item.name` is "root" for the ordinary no-graph turn, or a
                # real subagent name for an explicit `/alias` seed dispatch
                # (the only other caller bound to this event) — resolve
                # against the roster so a seed dispatch gets its own colored
                # badge, same as a task-graph step's DelegationStartEvent
                # handler just below, instead of falling through to root's
                # turn, which mounts no header of its own (the shared
                # `_ThinkingLine` covers it).
                resolved = next((s for s in SUBAGENTS if s.name == item.name), None)
                if resolved is not None:
                    await ws_renderer.start(
                        item.name, namespace=resolved.namespace, bg_color=NAMESPACE_COLORS[resolved.namespace],
                    )
                else:
                    await ws_renderer.start(item.name)
            elif isinstance(item, DelegationStartEvent):
                resolved = next((s for s in SUBAGENTS if s.name == item.agent_name), None)
                if resolved is not None:
                    depth = len(delegation_stack) + 1  # ws_renderer occupies depth 0
                    child_renderer = SubAgentRenderer(conversation, depth=depth)
                    await child_renderer.start(
                        resolved.name,
                        namespace=resolved.namespace,
                        ui_label=item.mission or _fallback_ui_label(item.task),
                        bg_color=NAMESPACE_COLORS[resolved.namespace],
                    )
                    delegation_stack.append(child_renderer)
                    active_renderer = child_renderer
            elif isinstance(item, DelegationDoneEvent):
                if delegation_stack:
                    closed = delegation_stack.pop()
                    await closed.done()
                else:
                    # Unmatched done — today's positional correlation (see the
                    # comment on `delegation_stack` above) should make this
                    # unreachable; guard rather than crash if it ever isn't.
                    pass
                active_renderer = delegation_stack[-1] if delegation_stack else ws_renderer
            elif isinstance(item, ThinkingTokenEvent):
                # Unconditional — regardless of which renderer (root,
                # subagent, nested delegation) is currently active, every
                # thinking token feeds the one shared line, not a
                # per-renderer preview.
                thinking_line.update_chunk(item.text)
            elif active_renderer:
                if isinstance(item, LogEvent):
                    await active_renderer.log(item.message, tool_name=item.tool_name)
                elif isinstance(item, DiffEvent):
                    await conversation.mount(DiffWidget(item.path, item.diff_lines))
                    if self._session is not None:
                        append_diff(self._session, item.path, item.diff_lines, turn=turn_id)
                    conversation.scroll_end(animate=False)
                elif isinstance(item, InferEndEvent):
                    active_renderer.accumulate_tokens(item)
                    self._other_ops_tokens += (item.prompt_tokens or 0) + (item.completion_tokens or 0)
                elif isinstance(item, TextChunkEvent):
                    self._streamed_answer += item.text
                    if self._assistant_widget is None:
                        self._assistant_widget = MessageWidget(MessageKind.ASSISTANT, self._streamed_answer)
                        await conversation.mount(self._assistant_widget)
                    else:
                        self._assistant_widget.update(self._streamed_answer)
                    conversation.scroll_end(animate=False)
                elif isinstance(item, StatusUpdateEvent):
                    await active_renderer.status_update(item)
                elif isinstance(item, DoneEvent):
                    if ws_renderer is not None:
                        await ws_renderer.done(item.thinking_chars)
                        # Unconditional even when `ws_renderer` itself has a
                        # badge (explicit `/alias` seed dispatch) — the
                        # shared line always ends with the root-style
                        # "Thought for X" summary.
                        thinking_line.finish(f"Thought for {ws_renderer.turn_summary_text()}")

        try:
            turn_result = await harness_turn.run_step(
                self._agent, self._session, raw, seed,
                turn_id=turn_id, session_id=session_id,
                permission_callback=self._permission_callback,
                hidden_grant_callback=self._hidden_grant_callback,
                append_user=append_user,
                on_event=_on_event,
                on_directive_verdict=self._apply_directive_verdict,
            )
        finally:
            # Safety net matching `ws_renderer.stop_spinner()` in `_stream`'s
            # own finally block below — guards against a leaked spinner task
            # if `DoneEvent` never arrives (e.g. an error mid-turn). A no-op
            # if `thinking_line.finish()` already stopped it above.
            thinking_line.stop_spinner()

        return _StepResult(
            outcome=turn_result.outcome,
            answer=turn_result.answer,
            max_iter_hit=turn_result.max_iter_hit,
            budget_exhausted=turn_result.budget_exhausted,
            review_report=turn_result.review_report,
            query_tool_count=turn_result.query_tool_count,
            ws_renderer=ws_renderer,
        )

    async def _stream(self, user_input: str, forced_seed: str | None = None) -> None:
        start = time.monotonic()
        events = self._agent.events
        turn_id = events.new_turn()
        session_id = self.session_id
        events.emit("turn.start", session=session_id, turn=turn_id, input_len=len(user_input))
        outcome = "ok"
        stage = ["harness"]
        verb = _OPERATIVE_VERB
        color = OPERATIVE_COLOR
        conversation = self.query_one("#conversation", ScrollableContainer)
        ws_renderer: SubAgentRenderer | None = None

        await self._maybe_warn_tiers_unconfigured(conversation)

        if self._session is not None and self._compact_failure_count < 3:
            session_tokens = _estimate_session_tokens(self._session)
            if compact.context_state(session_tokens, self._context_limit) == "auto":
                if await self._run_compact(conversation, instructions=""):
                    self._compact_failure_count = 0
                else:
                    self._compact_failure_count += 1

        async def _render_step_result(result: _StepResult, render_turn_id: str) -> None:
            if self._session is not None:
                self._refresh_status_bar()
            if result.max_iter_hit and not result.answer:
                if self._assistant_widget is not None:
                    await self._assistant_widget.remove()
                    self._assistant_widget = None
                await conversation.mount(MessageWidget(MessageKind.ERROR, "agent hit iteration limit without producing a response"))
            else:
                answer = result.answer
                # If TextChunkEvents already mounted the live-streaming widget
                # this turn (plan 34 Phase 2, no-graph direct dispatch), finish
                # it in place with the same persisted answer rather than
                # mounting a second copy — the graph path never streams, so
                # `_assistant_widget` is still None there and this falls back
                # to the original mount-fresh behavior unchanged.
                if self._assistant_widget is not None:
                    self._assistant_widget.update(answer)
                else:
                    self._assistant_widget = MessageWidget(MessageKind.ASSISTANT, answer)
                    await conversation.mount(self._assistant_widget)
                if result.budget_exhausted:
                    await conversation.mount(MessageWidget(MessageKind.WARNING, "response may be incomplete — the turn hit its tool-call budget before finishing"))
                if result.review_report is not None:
                    review_text = f"review findings:\n{result.review_report}"
                    await conversation.mount(MessageWidget(MessageKind.WARNING, review_text))
                    if self._session is not None:
                        append_event(self._session, review_text, source="review")
                elapsed = time.monotonic() - start
                operation_text = f"* {verb[1]} for {_fmt_duration(elapsed)}" + (f" ({result.query_tool_count} {'tool' if result.query_tool_count == 1 else 'tools'})" if result.query_tool_count > 0 else "")
                await conversation.mount(MessageWidget(MessageKind.OPERATION, operation_text, color=color))
                if self._session is not None:
                    append_operation(self._session, operation_text, color, turn=render_turn_id)
            conversation.scroll_end(animate=False)

        try:
            await self._start_status_animation(verb[0], color)

            step_result = await self._run_step(
                user_input, forced_seed,
                turn_id=turn_id, session_id=session_id, conversation=conversation,
                stage=stage,
            )
            ws_renderer = step_result.ws_renderer
            await _render_step_result(step_result, turn_id)

            # Auto-continue exactly once when the turn hit its tool-call
            # budget but still salvaged an answer: fire a single follow-up
            # turn seeded with that answer (the model's own "what remains"
            # summary) instead of leaving the user to retype a "resume your
            # work" prompt that forces a cold-restart rediscovery of files
            # the model already knew about. If the continuation ALSO
            # exhausts its budget, stop here — no third attempt.
            if step_result.budget_exhausted and step_result.answer:
                continuation_prompt = (
                    "the previous attempt hit its tool-call budget before finishing. "
                    "here is what you reported as done and remaining — continue from there:\n\n"
                    + step_result.answer
                )
                continuation_turn_id = events.new_turn()
                step_result = await self._run_step(
                    continuation_prompt, None,
                    turn_id=continuation_turn_id, session_id=session_id, conversation=conversation,
                    stage=stage,
                )
                ws_renderer = step_result.ws_renderer
                await _render_step_result(step_result, continuation_turn_id)

            if step_result.outcome == "max_iterations":
                outcome = "max_iterations"
        except Exception as error:
            error_msg = str(error) or type(error).__name__
            await conversation.mount(MessageWidget(MessageKind.ERROR, error_msg))
            append_event(self._session, error_msg, source="error")
            events.emit("error", level="warning", session=session_id, turn=turn_id, stage=stage[0], error_type=type(error).__name__, message=error_msg)
            outcome = "error"
            conversation.scroll_end(animate=False)
        finally:
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

    async def _run_compact(self, conversation: ScrollableContainer, instructions: str) -> bool:
        assert self._session is not None
        self._show_hint("compacting…")
        try:
            tier = resolve_touchpoint("micro", load_model_catalog(), load_tier_bindings())
            summary = await compact.summarize(self._session.messages, tier, instructions)
            compact.apply_summary(self._session, summary)
            append_compact(self._session, summary)
            await conversation.mount(MessageWidget(MessageKind.COMMAND_RESULT, "conversation compacted"))
            self._refresh_status_bar()
            success = True
        except Exception as error:
            await conversation.mount(MessageWidget(MessageKind.ERROR, f"compact failed: {error}"))
            success = False
        finally:
            self._clear_hint()
        conversation.scroll_end(animate=False)
        self._focus_prompt()
        return success

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

    def _toggle_esc_pending_clear(self) -> None:
        prompt = self.query_one("#prompt", TextArea)
        if self._esc_pending:
            prompt.clear()
            self._clear_hint()
        elif prompt.text:
            self._esc_pending = True
            self._show_hint("ESC again to clear input")

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

    async def action_cancel_stream(self) -> None:
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

        models_panel = self.query_one(ModelsPanel)
        if models_panel.display:
            edit = self._models_edit
            if edit is not None and edit.key_editing:
                # Discard the in-progress buffer only — not the whole
                # `/models` edit — mirroring how navigating away from the key
                # cell also discards it (see `_exit_models_key_edit`).
                edit.key_editing = False
                edit.key_edit_buffer = ""
                self._render_models_panel()
                return
            # Same helper as the `[cancel]` cell (step 7/8 of the design) —
            # Esc and `[cancel]` behave identically, not "hide silently".
            await self._cancel_models_edit(self.query_one("#conversation", ScrollableContainer))
            self._focus_prompt()
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

        self._toggle_esc_pending_clear()

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


        models_panel = self.query_one(ModelsPanel)
        if models_panel.display:
            await self._handle_models_enter(models_panel)
            return

        file_panel = self.query_one("#file-panel", FilePanel)
        if file_panel.display:
            self._consume_file_selection(file_panel.selected_text)
            return

        panel = self.query_one("#history-panel", HistoryPanel)
        if panel.display:
            self._consume_history_selection(panel.selected_text)
            return

        await self._submit_prompt()

    def _consume_file_selection(self, selected_text: str | None) -> None:
        file_panel = self.query_one("#file-panel", FilePanel)
        file_panel.hide()
        if selected_text is not None:
            prompt = self.query_one("#prompt", TextArea)
            at_pos = self._file_at_pos
            if at_pos != -1 and at_pos < len(prompt.text):
                new_value = prompt.text[:at_pos] + f"@{selected_text} "
                prompt.text = new_value
                self._prompt_move_to_end(prompt)
        self._file_at_pos = -1
        self._focus_prompt()

    def _consume_history_selection(self, selected_text: str | None) -> None:
        panel = self.query_one("#history-panel", HistoryPanel)
        if selected_text:
            prompt = self.query_one("#prompt", TextArea)
            prompt.text = selected_text
            self._prompt_move_to_end(prompt)
        panel.hide()
        self._focus_prompt()

    def on_file_panel_row_clicked(self, event: FilePanel.RowClicked) -> None:
        file_panel = self.query_one("#file-panel", FilePanel)
        file_panel.select_index(event.index)
        self._consume_file_selection(file_panel.selected_text)

    def on_history_panel_row_clicked(self, event: HistoryPanel.RowClicked) -> None:
        panel = self.query_one("#history-panel", HistoryPanel)
        panel.select_index(event.index)
        self._consume_history_selection(panel.selected_text)

    @staticmethod
    def _clipboard_sequence() -> int:
        try:
            import ctypes
            return ctypes.windll.user32.GetClipboardSequenceNumber()
        except Exception:
            return 0

    @staticmethod
    def _read_clipboard_text() -> str | None:
        """Best-effort read of the OS clipboard's text content (Windows only,
        same ctypes-only approach as `_clipboard_sequence` — no new
        dependency). Returns None on any failure: non-Windows, empty
        clipboard, or non-text content.

        `GetClipboardData`/`GlobalLock` return pointer-sized handles; ctypes
        defaults an undeclared `restype` to a 32-bit `c_int`, which silently
        truncates those handles on 64-bit Windows whenever the real address
        is above 4GB (routine on a 64-bit process) — the truncated address
        is garbage, `wstring_at` on it raises, and the broad `except`
        below turns that into a false "clipboard is empty". `restype`/
        `argtypes` must be declared explicitly to keep the full pointer.
        """
        try:
            import ctypes

            CF_UNICODETEXT = 13
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            user32.OpenClipboard.argtypes = [ctypes.c_void_p]
            user32.GetClipboardData.argtypes = [ctypes.c_uint]
            user32.GetClipboardData.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

            if not user32.OpenClipboard(None):
                return None
            try:
                handle = user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    return None
                locked = kernel32.GlobalLock(handle)
                if not locked:
                    return None
                try:
                    return ctypes.wstring_at(locked)
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        except Exception:
            return None

    async def _poll_clipboard(self) -> None:
        """Keeps `last` fresh for the whole app lifetime (so opening the
        `/models` key-edit field never fires a stale notice for an earlier,
        unrelated copy), but only surfaces the visible notice while that
        field is the one thing actually listening for a pasted key — the
        sequence number itself is OS-wide (any app's copy bumps it), so
        showing it unconditionally reported on every copy anywhere on the
        system, not just Gekai."""
        last = self._clipboard_sequence()
        while True:
            await asyncio.sleep(0.4)
            current = self._clipboard_sequence()
            if current != last:
                last = current
                edit = self._models_edit
                if edit is not None and edit.key_editing:
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

    def _apply_directive_verdict(self, path: str, verdict: AuditVerdict | None) -> None:
        """`GekaiAgent.start_directive_audit`'s callback (plan 35 v3 concept
        3): one slot, one line, last verdict wins — there is no per-file
        store, so a second audit landing (GEKAI.md, then a foreign-file
        read, or vice versa) simply overwrites whatever is showing now.
        Not a toast: the condition reported can last the whole session, so
        the notice persists until the next verdict replaces it, with no
        timer to auto-hide it.

        GEKAI.md always reports — in flight, YES, and NO all leave a mark,
        because its read is invisible and silence would be indistinguishable
        from a broken audit. A foreign file only warns on YES; a clean
        foreign-file read stays silent because the user asked for that read
        and watched it happen (concept 3)."""
        try:
            notice = self.query_one("#directive-notice", Static)
        except Exception:
            return

        is_gekai_md = path == "GEKAI.md"

        if verdict is None:
            notice.update(f"⋯ checking {path}")
            notice.styles.color = "grey"
            notice.display = True
            return

        if verdict.has_directives:
            notice.update(f"⚠  {path} contains agent directives")
            notice.styles.color = "yellow"
            notice.display = True
            return

        if is_gekai_md:
            notice.update("✓ GEKAI.md loaded")
            notice.styles.color = "green"
            notice.display = True
            return

        notice.update("")
        notice.display = False

    def _hide_directive_notice(self) -> None:
        try:
            notice = self.query_one("#directive-notice", Static)
            notice.update("")
            notice.display = False
        except Exception:
            pass

    def _any_panel_active(self) -> bool:
        return (
            self.query_one("#file-panel", FilePanel).display
            or self.query_one("#history-panel", HistoryPanel).display
            or self.query_one(ChoiceBar).display
            or self.query_one(CommandPalette).display
            or self.query_one(ModelsPanel).display
        )

    def _sync_prompt_lock(self) -> None:
        """The prompt must be visibly inert — no cursor, no blink, no typed
        input landing in it — whenever a panel navigated purely by arrow
        keys/clicks (HistoryPanel/ChoiceBar/ModelsPanel) is showing; typing
        only makes sense once that panel closes and the prompt is the active
        surface again. FilePanel and CommandPalette are excluded: they are
        typeahead filters over the prompt's own text ("@"/"/" + what's typed
        after), so locking the prompt while they're open would swallow the
        very keystrokes they filter on. `read_only` blocks keyboard edits,
        `show_cursor=False` (its Textual-documented pairing) hides the caret
        entirely rather than just freezing it mid-blink."""
        prompt = self.query_one("#prompt", TextArea)
        active = (
            self.query_one("#history-panel", HistoryPanel).display
            or self.query_one(ChoiceBar).display
            or self.query_one(ModelsPanel).display
        )
        if active == prompt.read_only:
            return
        prompt.read_only = active
        prompt.show_cursor = not active
        if active:
            prompt.blur()
        else:
            self._focus_prompt()

    async def _poll_prompt_lock(self) -> None:
        while True:
            await asyncio.sleep(0.15)
            self._sync_prompt_lock()

    def action_scroll_to_top(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_home(animate=False)

    def action_scroll_to_end(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_end(animate=False)

    def action_scroll_page_up(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_page_up(animate=False)

    def action_scroll_page_down(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_page_down(animate=False)

    def _exit_models_key_edit(self) -> None:
        """Navigating away from an in-progress key edit (arrows, or landing
        on a different cell) discards the typed/pasted buffer — per the
        locked design, the only way text reaches `key_input` is confirming
        it in place with Enter (see `_handle_models_enter`)."""
        edit = self._models_edit
        if edit is not None and edit.key_editing:
            edit.key_editing = False
            edit.key_edit_buffer = ""
            self._render_models_panel()

    def action_navigate_up(self) -> None:
        if self.query_one("#file-panel", FilePanel).display:
            self.query_one("#file-panel", FilePanel).move_up()
            return
        if self.query_one(ModelsPanel).display:
            self._exit_models_key_edit()
            self.query_one(ModelsPanel).move_up()
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
        if self.query_one(ModelsPanel).display:
            self._exit_models_key_edit()
            self.query_one(ModelsPanel).move_down()
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

    def action_navigate_left(self) -> None:
        # `left`/`right` have no existing App-level binding (unlike up/down),
        # so this — and `action_navigate_right` — are new. Both `ModelsPanel`
        # and `ChoiceBar` previously relied on `on_key`'s left/right special
        # case (see `on_key` above) to move their cursor; since these are
        # `priority=True` bindings, that `on_key` path can no longer fire for
        # left/right (a priority binding always claims the key before it is
        # forwarded to the focused widget — see `action_navigate_up`'s own
        # comment/precedent for the same problem on up/down), so both panels'
        # left/right handling is reproduced here instead of left to `on_key`.
        if self.query_one(ModelsPanel).display:
            self._exit_models_key_edit()
            self.query_one(ModelsPanel).move_left()
            return
        if self.query_one(ChoiceBar).display:
            self.query_one(ChoiceBar).move_left()
            return
        prompt = self.query_one("#prompt", TextArea)
        if prompt.has_focus:
            prompt.action_cursor_left()

    def action_navigate_right(self) -> None:
        if self.query_one(ModelsPanel).display:
            self._exit_models_key_edit()
            self.query_one(ModelsPanel).move_right()
            return
        if self.query_one(ChoiceBar).display:
            self.query_one(ChoiceBar).move_right()
            return
        prompt = self.query_one("#prompt", TextArea)
        if prompt.has_focus:
            prompt.action_cursor_right()

    async def action_select_command(self, name: str) -> None:
        palette = self.query_one(CommandPalette)
        palette.hide()
        prompt = self.query_one("#prompt", TextArea)
        if self._params_for(name):
            prompt.text = f"/{name} "
            self._prompt_move_to_end(prompt)
            self._focus_prompt()
            return
        prompt.text = f"/{name}"
        await self._submit_prompt()

    @on(events.Click, "#scroll-hint")
    def _scroll_hint_clicked(self, event: events.Click) -> None:
        event.stop()
        self.action_scroll_to_end()

        self._toggle_esc_pending_clear()

        self._clear_status()
        self._focus_prompt()
