from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyfiglet
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.message import Message
from textual.widgets import Input, ProgressBar, Static
from textual.worker import Worker

from agent import __version_core__, __version_label__
from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.persistence import now_utc_str
from agent.router import Intent, Session
from agent.settings import load_context_limit, load_ws_scan_staleness_min, resolve_permissions, save_permissions
from agent.ws_explorer.enrichment import _get_git_state
from agent.ui import random_accent_color, random_farewell, random_operative_verb
from agent.subagent import SubAgentEvent, SubAgentStartEvent, LogEvent, InferEndEvent, DoneEvent, StatusUpdateEvent

from .palette import CommandPalette
from .permissions import PermissionScreen
from .widgets import ChoiceBar, MessageKind, MessageWidget


class ConversationContainer(ScrollableContainer):
    class Scrolled(Message):
        def __init__(self, at_end: bool) -> None:
            super().__init__()
            self.at_end = at_end

    def watch_scroll_y(self, scroll_y: float) -> None:
        at_end = scroll_y >= self.max_scroll_y or self.max_scroll_y <= 0
        self.post_message(self.Scrolled(at_end=at_end))


_SPINNER_FRAMES = ["|", "/", "-", "\\"]
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
        self._start_time: float = time.monotonic()
        self._debug = debug
        self._current_tool: str = ""
        self._current_count: int = 0
        self._current_widget: Static | None = None
        self._current_prefix: str = ""
        self._progress_bar: ProgressBar | None = None
        self._header_widget: MessageWidget | None = None
        self._spinner_task: asyncio.Task | None = None

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
            self._current_widget = widget
            self._current_prefix = prefix
        else:
            widget = Static(f"{prefix} {message}")
            await self._conversation.mount(widget)
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

    def accumulate_tokens(self, event: "InferEndEvent") -> None:
        if event.prompt_tokens:
            self._total_tokens += event.prompt_tokens
        if event.completion_tokens:
            self._total_tokens += event.completion_tokens

    async def done(self) -> None:
        if self._spinner_task is not None:
            self._spinner_task.cancel()
            self._spinner_task = None
        if self._header_widget is not None:
            self._header_widget.query_one(".header-dot", Static).update("[#666666]●[/#666666]")
            self._header_widget.query_one(".header-text", Static).update("[#666666]Thought[/#666666]")
            self._header_widget = None
        if self._progress_bar is not None:
            await self._progress_bar.remove()
            self._progress_bar = None
        elapsed = time.monotonic() - self._start_time
        parts = [_fmt_duration_verbose(elapsed)]
        if self._total_tokens > 0:
            parts.insert(0, f"{_fmt_tokens(self._total_tokens)} tokens")
        summary = " · ".join(parts)
        prefix = "  ⎿" if self._first_item else "   "
        self._first_item = False
        await self._conversation.mount(Static(f"{prefix} Done ({summary})"))
        self._conversation.scroll_end(animate=False)


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
    # Includes transcript + persistent system messages ([preference], [artifact], <lang>).
    # Artifacts from prior Query turns are what make this number grow meaningfully.
    return sum(len(str(m.get("content") or "")) for m in session.messages) // 4


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
        padding: 1 0 1 0;
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
        height: 3;
        layers: input marker;
        border-top: solid #3a3a3a;
        border-bottom: solid #3a3a3a;
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
        height: 1;
        padding: 0 0 0 2;
        background: ansi_default;
        background-tint: transparent;
        color: ansi_default;
    }

    #prompt:focus {
        background: ansi_default;
        background-tint: transparent;
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
        Binding("ctrl+down", "scroll_to_end", "Scroll to bottom", priority=True),
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
        self._needs_permissions = needs_permissions
        self._worker: Worker | None = None
        self._workspace: dict | None = None
        self._assistant_widget: MessageWidget | None = None
        self._status_task: asyncio.Task[None] | None = None
        self._status_stop: asyncio.Event | None = None
        self._status_frame: int = 0
        self._status_verb: str = ""
        self._status_start: float = 0.0
        self._current_lang: str = "EN"
        self._esc_pending: bool = False
        self._pending_choice: asyncio.Future[str | None] | None = None
        self._context_limit: int = 128_000
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
            with Container(id="input-area"):
                yield Input(id="prompt", compact=True)
                yield Static("❯", id="prompt-marker")
            yield Static("", id="context-bar")
            yield Static(f"[dim]{__version_core__}[/dim] [bold white]{__version_label__}[/bold white]", id="version-bar")

    async def on_mount(self) -> None:
        if self._needs_permissions:
            self.push_screen(PermissionScreen(), callback=self._on_permission_selected)
            return
        self.run_worker(self._init_session(), exclusive=True)

    def _on_permission_selected(self, choice: str | None) -> None:
        perms = resolve_permissions(choice) if choice else None
        if perms is None:
            self.exit()
            return
        save_permissions(self._working_dir, perms)
        self._agent.permissions = perms
        self.run_worker(self._init_session(), exclusive=True)

    async def _init_session(self) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)

        banner_text = pyfiglet.figlet_format("gek-AI", font="small_slant").rstrip()
        await conversation.mount(MessageWidget(MessageKind.BANNER, banner_text))
        cache_path = self._working_dir / ".gekai" / "workspace.json"
        workspace: dict = {}

        if not cache_path.exists():
            await self._run_ws_explorer(conversation)
            workspace = self._workspace
        else:
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

        override = load_context_limit(self._working_dir)
        self._context_limit = override if override is not None else _context_limit(self._agent.model)
        self.query_one("#context-bar", Static).update(
            _fmt_status_bar(self._agent.model, self._working_dir.name, self._branch, _estimate_session_tokens(self._session), self._context_limit)
        )

        if self._restored_messages:
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

    async def _clear_session(self) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)
        await conversation.remove_children()
        self._session = self._agent.start_session(self._workspace)
        self.query_one("#context-bar", Static).update(
            _fmt_status_bar(self._agent.model, self._working_dir.name, self._branch, _estimate_session_tokens(self._session), self._context_limit)
        )
        self._current_lang = "EN"
        self._assistant_widget = None
        banner_text = pyfiglet.figlet_format("gek-AI", font="small_slant").rstrip()
        await conversation.mount(MessageWidget(MessageKind.BANNER, banner_text))
        self._focus_prompt()

    @property
    def session_id(self) -> str | None:
        return self._session.id if self._session else None

    @property
    def session_has_interactions(self) -> bool:
        if self._session is None:
            return False
        return any(m.get("role") == "user" for m in self._session.messages)

    def _focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus(scroll_visible=False)

    def on_input_changed(self, event: Input.Changed) -> None:
        palette = self.query_one(CommandPalette)
        if event.value.startswith("/"):
            palette.filter(event.value[1:])
        else:
            palette.hide()
        if self._esc_pending:
            self._clear_hint()

    def on_key(self, event: events.Key) -> None:
        choice_bar = self.query_one(ChoiceBar)
        if choice_bar.display and event.key in ("left", "right", "up", "down"):
            if event.key in ("left", "up"):
                choice_bar.move_left()
            else:
                choice_bar.move_right()
            event.stop()
            return

        palette = self.query_one(CommandPalette)
        if palette.display and event.key in ("up", "down"):
            if event.key == "up":
                palette.move_up()
            else:
                palette.move_down()
            event.stop()
            return

        if self._esc_pending and event.key != "escape":
            self._clear_hint()

        prompt = self.query_one("#prompt", Input)
        if prompt.has_focus or not event.is_printable:
            return
        prompt.focus(scroll_visible=False)
        prompt.insert_text_at_cursor(event.character)
        event.stop()

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

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._pending_choice is not None and not self._pending_choice.done():
            event.input.value = ""
            choice_bar = self.query_one(ChoiceBar)
            key = choice_bar.selected_key
            choice_bar.hide()
            future = self._pending_choice
            self._pending_choice = None
            future.set_result(key)
            self._focus_prompt()
            return
        if self._session is None:
            self._focus_prompt()
            return
        if self._worker is not None and not self._worker.is_finished:
            self._focus_prompt()
            return

        palette = self.query_one(CommandPalette)
        stripped = event.value.strip()
        if palette.display:
            cmd = palette.selected_command
            palette.hide()
            if cmd:
                stripped = f"/{cmd}"
        if not stripped:
            self._focus_prompt()
            return
        event.input.value = ""
        conversation = self.query_one("#conversation", ScrollableContainer)
        if stripped.startswith("/"):
            await conversation.mount(MessageWidget(MessageKind.USER, stripped))
            conversation.scroll_end(animate=False)
            cmd_name = stripped.lstrip("/").split()[0]
            if cmd_name == "workspace:rebuild":
                self._worker = self.run_worker(self._rebuild_workspace(), exclusive=True)
                return
            result = await self._command_registry.dispatch(stripped)
            if result.output:
                await conversation.mount(MessageWidget(MessageKind.ASSISTANT, result.output, color="#ffd700"))
            if result.clear_session:
                await self._clear_session()
                return
            if result.exit_app:
                farewell = random_farewell()
                await conversation.mount(MessageWidget(MessageKind.ASSISTANT, farewell))
                conversation.scroll_end(animate=False)
                await asyncio.sleep(0.8 + len(farewell.split(" ")) * 0.20)
                self.exit()
                return
            self._focus_prompt()
            return
        await conversation.mount(MessageWidget(MessageKind.USER, stripped))
        self._worker = self.run_worker(self._stream(stripped), exclusive=True)

    async def _run_ws_explorer(self, conversation: ScrollableContainer) -> None:
        """Run WsExplorer and update session workspace context."""
        explorer = self._agent.create_ws_explorer(self._working_dir)
        renderer: SubAgentRenderer | None = None
        try:
            async for event in explorer.run():
                if isinstance(event, SubAgentStartEvent):
                    await self._start_status_animation("scanning workspace", random_accent_color())
                    renderer = SubAgentRenderer(conversation)
                    await renderer.start(event.name, event.description, event.color)
                elif isinstance(event, StatusUpdateEvent) and renderer:
                    await renderer.status_update(event)
                elif isinstance(event, LogEvent) and renderer:
                    await renderer.log(event.message)
                elif isinstance(event, InferEndEvent) and renderer:
                    renderer.accumulate_tokens(event)
                elif isinstance(event, DoneEvent) and renderer:
                    await renderer.done()
        except Exception as error:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"workspace scan error: {error}"))
            conversation.scroll_end(animate=False)
        finally:
            await self._stop_status_animation()
        cache_path = self._working_dir / ".gekai" / "workspace.json"
        try:
            self._workspace = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._workspace = explorer.workspace or {}
        if self._workspace and self._session is not None:
            self._agent.update_workspace_context(self._session, self._workspace)

    async def _maybe_rescan_workspace(self, conversation: ScrollableContainer) -> None:
        """Run workspace re-scan activation logic. Updates session and self._workspace if scan fires."""
        cache_path = self._working_dir / ".gekai" / "workspace.json"
        if not cache_path.exists():
            # Onboarding fallback — should not normally happen here
            await self._run_ws_explorer(conversation)
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
        if await self._ask_choice("Workspace changed — rescan?", [("y", "Yes"), ("n", "No")]) == "y":
            await self._run_ws_explorer(conversation)

    async def _stream(self, user_input: str) -> None:
        start = time.monotonic()
        verb = random_operative_verb()
        color = random_accent_color()
        conversation = self.query_one("#conversation", ScrollableContainer)
        answer_chunks: list[str] = []
        ws_renderer: SubAgentRenderer | None = None
        query_tool_count: int = 0

        try:
            await self._start_status_animation(verb[0], color)
            normalized, src_lang = await self._agent.normalize(user_input)
            segments = await self._agent.classify(normalized)
            if self._agent.debug:
                labels = []
                for intent, _sub, plan in segments:
                    label = intent.name
                    if plan:
                        label += "+plan"
                    labels.append(label)
                debug_text = f"\\[classifier: {', '.join(labels)}]"
                await conversation.mount(
                    MessageWidget(MessageKind.OPERATION, debug_text, color="#BA55D3")
                )
                if normalized == user_input:
                    norm_debug = "\\[OK]"
                else:
                    snippet = (normalized[:30] + "...") if len(normalized) > 30 else normalized
                    norm_debug = f"\\[{snippet}, {src_lang}]" if src_lang else f"\\[{snippet}]"
                await conversation.mount(
                    MessageWidget(MessageKind.OPERATION, norm_debug, color="#BA55D3")
                )
            new_lang = src_lang or "EN"
            if new_lang != self._current_lang:
                self._current_lang = new_lang
                lang_hint = f"<lang>\nfrom now on answer in: {new_lang}"
                self._session.messages.append({"role": "system", "content": lang_hint})
            # Activation: check workspace staleness before any QUERY+plan
            if any(intent == Intent.QUERY and plan for intent, _, plan in segments):
                await self._stop_status_animation()
                await self._maybe_rescan_workspace(conversation)
                await self._start_status_animation(verb[0], color)
            async for item in self._agent.process_stream(self._session, normalized, segments, original_input=user_input):
                if isinstance(item, str):
                    answer_chunks.append(item)
                elif isinstance(item, SubAgentEvent):
                    if isinstance(item, SubAgentStartEvent):
                        ws_renderer = SubAgentRenderer(conversation, debug=self._agent.debug)
                        await ws_renderer.start(item.name, item.description, item.color)
                    elif ws_renderer:
                        if isinstance(item, LogEvent):
                            await ws_renderer.log(item.message, tool_name=item.tool_name)
                            if ws_renderer.name == "query":
                                query_tool_count += 1
                        elif isinstance(item, InferEndEvent):
                            ws_renderer.accumulate_tokens(item)
                        elif isinstance(item, StatusUpdateEvent):
                            await ws_renderer.status_update(item)
                        elif isinstance(item, DoneEvent):
                            await ws_renderer.done()
            if self._session is not None:
                self.query_one("#context-bar", Static).update(
                    _fmt_context_pct(_estimate_session_tokens(self._session), self._context_limit)
                )
            answer = "".join(answer_chunks).rstrip("\n")
            self._assistant_widget = MessageWidget(MessageKind.ASSISTANT, answer)
            await conversation.mount(self._assistant_widget)
            elapsed = time.monotonic() - start
            await conversation.mount(
                MessageWidget(
                    MessageKind.OPERATION,
                    f"* {verb[1]} for {_fmt_duration(elapsed)}" + (f" ({query_tool_count} tools)" if query_tool_count > 0 else ""),
                    color=color,
                )
            )
            conversation.scroll_end(animate=False)
        except Exception as error:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"error: {error}"))
            conversation.scroll_end(animate=False)
        finally:
            await self._stop_status_animation()
            self._worker = None
            self._focus_prompt()

    async def _rebuild_workspace(self) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)
        try:
            await self._run_ws_explorer(conversation)
            self._session = self._agent.start_session(self._workspace or {})
            conversation.scroll_end(animate=False)
        except Exception as error:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"error: {error}"))
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
        if self._pending_choice is not None and not self._pending_choice.done():
            self.query_one(ChoiceBar).hide()
            future = self._pending_choice
            self._pending_choice = None
            future.set_result(None)
            return

        palette = self.query_one(CommandPalette)
        if palette.display:
            self.query_one("#prompt", Input).value = ""
            palette.hide()
            self._clear_hint()
            return
        if self._worker is not None and not self._worker.is_finished:
            self._worker.cancel()
            self._clear_hint()
            self._clear_status()
            self._focus_prompt()
            return

    def action_scroll_to_end(self) -> None:
        self.query_one("#conversation", ConversationContainer).scroll_end(animate=False)

    def action_select_command(self, name: str) -> None:
        palette = self.query_one(CommandPalette)
        palette.hide()
        prompt = self.query_one("#prompt", Input)
        prompt.value = f"/{name}"
        prompt.action_submit()

    @on(events.Click, "#scroll-hint")
    def _scroll_hint_clicked(self, event: events.Click) -> None:
        event.stop()
        self.action_scroll_to_end()

        prompt = self.query_one("#prompt", Input)
        if self._esc_pending:
            prompt.value = ""
            self._clear_hint()
        elif prompt.value:
            self._esc_pending = True
            self._show_hint("ESC again to clear input")

        self._clear_status()
        self._focus_prompt()
