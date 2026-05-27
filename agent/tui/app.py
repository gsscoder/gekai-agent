from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyfiglet
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Input, ProgressBar, Static
from textual.worker import Worker

from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.router import Intent, Session
from agent.settings import load_ws_scan_staleness_min, resolve_permissions, save_permissions
from agent.ws_explorer.enrichment import _get_git_state
from agent.ui import random_accent_color, random_farewell, random_operative_verb
from agent.subagent import SubAgentEvent, SubAgentStartEvent, LogEvent, InferStartEvent, InferDeltaEvent, InferEndEvent, DoneEvent, StatusUpdateEvent

from .palette import CommandPalette
from .permissions import PermissionScreen
from .widgets import MessageKind, MessageWidget


_SPINNER_FRAMES = ["|", "/", "-", "\\"]


def _write_asked_timestamp(cache_path: Path) -> None:
    try:
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    scan_state = existing.get("scan_state", {})
    scan_state["asked_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    existing["scan_state"] = scan_state
    try:
        cache_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except OSError:
        pass


def _refresh_scan_timestamp(cache_path: Path) -> None:
    try:
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    scan_state = existing.get("scan_state", {})
    scan_state["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
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

    async def start(self, name: str, description: str, color: str) -> None:
        self.name = name
        await self._conversation.mount(Static("", classes="assistant-spacer"))
        bg = color or "grey50"
        header_markup = f"[bold black on {bg}]{name}[/bold black on {bg}][white]({description})[/white]"
        await self._conversation.mount(MessageWidget(MessageKind.HEADER, header_markup))

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
        await self._conversation.mount(Static("", classes="assistant-spacer"))
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
    return f"{verb.capitalize()}... [white]({_fmt_elapsed(elapsed)} · thinking)[/white]"


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
        padding-bottom: 2;
        background: ansi_default;
    }

    #question-bar {
        height: 1;
        background: ansi_default;
        color: orange;
        padding: 0 1 0 0;
        display: none;
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
    """

    BINDINGS = [
        ("escape", "cancel_stream", "Cancel"),
        ("ctrl+c", "quit", "Quit"),
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
        self._pending_question: asyncio.Future[str] | None = None
        super().__init__(**kwargs)
        self.ansi_color = True

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="conversation")
        with Container(id="footer"):
            yield Static("", id="question-bar")
            yield Static("", id="status-line")
            yield Static("", id="status-spacer")
            yield CommandPalette(self._command_registry, id="command-palette")
            yield Static("", id="hint-area")
            with Container(id="input-area"):
                yield Input(id="prompt", compact=True)
                yield Static("❯", id="prompt-marker")

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
        await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"gekai v{self._version}"))
        if self._branch:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"{self._working_dir.name} | {self._branch}"))
        else:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, self._working_dir.name))

        cache_path = self._working_dir / ".gekai" / "workspace.json"
        workspace: dict = {}

        if not cache_path.exists():
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
            workspace = explorer.workspace or {}
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

        if self._restored_messages:
            for msg in self._restored_messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    await conversation.mount(MessageWidget(MessageKind.USER, content))
                elif role == "assistant":
                    await conversation.mount(MessageWidget(MessageKind.ASSISTANT, content))

        self._focus_prompt()
        self.call_after_refresh(self._focus_prompt)

    async def _clear_session(self) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)
        await conversation.remove_children()
        self._session = self._agent.start_session(self._workspace)
        self._current_lang = "EN"
        self._assistant_widget = None
        banner_text = pyfiglet.figlet_format("gek-AI", font="small_slant").rstrip()
        await conversation.mount(MessageWidget(MessageKind.BANNER, banner_text))
        await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"gekai v{self._version}"))
        if self._branch:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"{self._working_dir.name} | {self._branch}"))
        else:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, self._working_dir.name))
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

    async def _ask_inline(self, question: str) -> str:
        loop = asyncio.get_event_loop()
        self._pending_question = loop.create_future()
        bar = self.query_one("#question-bar", Static)
        bar.update(question)
        bar.display = True
        return await self._pending_question

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._pending_question is not None and not self._pending_question.done():
            answer = event.value.strip()
            event.input.value = ""
            bar = self.query_one("#question-bar", Static)
            bar.display = False
            bar.update("")
            future = self._pending_question
            self._pending_question = None
            conversation = self.query_one("#conversation", ScrollableContainer)
            await conversation.mount(MessageWidget(MessageKind.USER, answer or "n"))
            conversation.scroll_end(animate=False)
            future.set_result(answer or "n")
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
                await conversation.mount(MessageWidget(MessageKind.SYSTEM, result.output))
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
        if explorer.workspace:
            self._workspace = explorer.workspace
            if self._session is not None:
                self._agent.update_workspace_context(self._session, explorer.workspace)

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
            await asyncio.to_thread(_refresh_scan_timestamp, cache_path)
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
        await asyncio.to_thread(_write_asked_timestamp, cache_path)
        answer = await self._ask_inline("Workspace may have changed. Scan again? [y/N]")
        if answer.lower() in ("y", "yes"):
            await self._run_ws_explorer(conversation)

    async def _stream(self, user_input: str) -> None:
        start = time.monotonic()
        verb = random_operative_verb()
        color = random_accent_color()
        conversation = self.query_one("#conversation", ScrollableContainer)
        answer_chunks: list[str] = []
        completion_tokens: int = 0
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

    def action_cancel_stream(self) -> None:
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

        prompt = self.query_one("#prompt", Input)
        if self._esc_pending:
            prompt.value = ""
            self._clear_hint()
        elif prompt.value:
            self._esc_pending = True
            self._show_hint("ESC again to clear input")

        self._clear_status()
        self._focus_prompt()


class _PlaceholderApp(App[None]):
    CSS = GekaiApp.CSS

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="conversation")
        yield Input(placeholder="Message...", id="prompt")

    async def on_mount(self) -> None:
        conversation = self.query_one("#conversation", ScrollableContainer)
        await conversation.mount(Static("gekai TUI — placeholder"))
        await conversation.mount(Static("Type a message below and press Enter."))


def main() -> None:
    _PlaceholderApp().run()


if __name__ == "__main__":
    main()
