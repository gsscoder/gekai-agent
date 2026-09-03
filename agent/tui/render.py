"""Live turn rendering: the two widget groups a streaming turn drives.

`SubAgentRenderer` owns one agent's header, tool log, and Done line;
`_ThinkingLine` owns the single shared thinking preview pinned above them.
Both are driven entirely by `AgentEvent`s, so the app's event handler stays a
dispatch table rather than also owning the drawing.
"""

from __future__ import annotations

import asyncio
import time
import re

from rich.text import Text
from textual.containers import ScrollableContainer
from textual.widgets import ProgressBar, Static

from agent.events import InferEndEvent, StatusUpdateEvent
from agent.tools.catalog import EDIT_TOOLS, FS_TOOLS, READ_TOOLS, SHELL_TOOLS

from .status import _fmt_duration_verbose, _fmt_tokens
from .widgets import MessageKind, MessageWidget


def _subagent_header_markup(name: str, bg_color: str, ui_label: str) -> str:
    markup = f"[black on {bg_color} bold] {name} [/]"
    if ui_label:
        markup += f"[white]\\[{ui_label}][/white]"
    return markup


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
