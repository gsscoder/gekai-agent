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
from textual.widgets import Input, Static
from textual.worker import Worker

from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.handlers.chat import UsageInfo
from agent.router import Session
from agent.settings import resolve_permissions, save_permissions
from agent.ui import random_accent_color, random_farewell, random_operative_verb
from agent.subagent import SubAgentEvent, SubAgentStartEvent, LogEvent, InferStartEvent, InferEndEvent, DoneEvent
from agent.ws_explorer import Mode

from .palette import CommandPalette
from .permissions import PermissionScreen
from .widgets import MessageKind, MessageWidget


_SPINNER_FRAMES = ["|", "/", "-", "\\"]


class SubAgentRenderer:
    """Manages header + L-connector state for rendering subagent events into the conversation."""

    def __init__(self, conversation: ScrollableContainer) -> None:
        self._conversation = conversation
        self._first_item = True

    async def start(self, name: str) -> None:
        await self._conversation.mount(Static("", classes="assistant-spacer"))
        await self._conversation.mount(MessageWidget(MessageKind.HEADER, name))

    async def log(self, message: str) -> None:
        prefix = "⎿" if self._first_item else " "
        self._first_item = False
        await self._conversation.mount(
            MessageWidget(MessageKind.SYSTEM, f"{prefix} {message}")
        )
        self._conversation.scroll_end(animate=False)


def _fmt_duration(elapsed: float) -> str:
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    return f"{elapsed / 60:.1f}m"


def _estimate_tokens(text: str) -> int:
    return max(0, round(len(text) / 4))



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
        self._status_text: str = ""
        self._current_lang: str = "EN"
        super().__init__(**kwargs)
        self.ansi_color = True

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="conversation")
        with Container(id="footer"):
            yield Static("", id="status-line")
            yield Static("", id="status-spacer")
            yield CommandPalette(self._command_registry, id="command-palette")
            with Container(id="input-area"):
                yield Input(id="prompt", compact=True)
                yield Static("❯", id="prompt-marker")

    async def on_mount(self) -> None:
        if self._needs_permissions:
            self.push_screen(PermissionScreen(), callback=self._on_permission_selected)
            return
        await self._init_session()

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

        # Banner
        banner_text = pyfiglet.figlet_format("gek-AI", font="small_slant").rstrip()
        await conversation.mount(MessageWidget(MessageKind.BANNER, banner_text))
        await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"gekai v{self._version}"))
        if self._branch:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"{self._working_dir.name} | {self._branch}"))
        else:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, self._working_dir.name))


        # Workspace cache
        cache_path = self._working_dir / ".gekai" / "workspace.json"
        max_age = 30 * 60 if self._restored_id else 15 * 60
        cache_age = (
            (datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime)
            if cache_path.exists()
            else float("inf")
        )
        if cache_age < max_age:
            workspace = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            color = random_accent_color()
            await self._start_status_animation("scanning workspace...", color)
            explorer = self._agent.create_ws_explorer(self._working_dir, Mode.SCAN)
            renderer: SubAgentRenderer | None = None
            async for event in explorer.run():
                if isinstance(event, SubAgentStartEvent):
                    renderer = SubAgentRenderer(conversation)
                    await renderer.start(event.name)
                elif isinstance(event, LogEvent) and renderer:
                    await renderer.log(event.message)
            await self._stop_status_animation()
            workspace = explorer.workspace

        self._workspace = workspace

        # Start session
        self._session = self._agent.start_session(
            workspace,
            restored_messages=self._restored_messages,
            session_id=self._restored_id,
        )

        # Replay history
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

    def on_key(self, event: events.Key) -> None:
        palette = self.query_one(CommandPalette)
        if palette.display and event.key in ("up", "down"):
            if event.key == "up":
                palette.move_up()
            else:
                palette.move_down()
            event.stop()
            return

        prompt = self.query_one("#prompt", Input)
        if prompt.has_focus or not event.is_printable:
            return
        prompt.focus(scroll_visible=False)
        prompt.insert_text_at_cursor(event.character)
        event.stop()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
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

    async def _stream(self, user_input: str) -> None:
        start = time.monotonic()
        verb = random_operative_verb()
        color = random_accent_color()
        conversation = self.query_one("#conversation", ScrollableContainer)
        answer_chunks: list[str] = []
        completion_tokens: int = 0
        ws_renderer: SubAgentRenderer | None = None

        def _verb_status() -> str:
            tokens_part = f" (↓ {completion_tokens})" if completion_tokens > 0 else ""
            return f"{verb[0]}...{tokens_part}"

        try:
            await self._start_status_animation(_verb_status(), color)
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
            async for item in self._agent.process_stream(self._session, normalized, segments, original_input=user_input):
                if isinstance(item, str):
                    answer_chunks.append(item)
                    completion_tokens = _estimate_tokens("".join(answer_chunks))
                    self._status_text = _verb_status()
                elif isinstance(item, SubAgentEvent):
                    if isinstance(item, SubAgentStartEvent):
                        ws_renderer = SubAgentRenderer(conversation)
                        await ws_renderer.start(item.name)
                    elif ws_renderer:
                        if isinstance(item, LogEvent):
                            await ws_renderer.log(item.message)
                        elif isinstance(item, InferStartEvent):
                            self._status_text = f"{verb[0]}..."
                        elif isinstance(item, InferEndEvent):
                            if item.prompt_tokens is not None and item.completion_tokens is not None:
                                self._status_text = f"{verb[0]}... (↑ {item.prompt_tokens}  ↓ {item.completion_tokens})"
                        elif isinstance(item, DoneEvent):
                            self._status_text = _verb_status()
                elif isinstance(item, UsageInfo):
                    completion_tokens = item.completion_tokens
                    self._status_text = _verb_status()
            answer = "".join(answer_chunks).rstrip("\n")
            self._assistant_widget = MessageWidget(MessageKind.ASSISTANT, answer)
            await conversation.mount(self._assistant_widget)
            elapsed = time.monotonic() - start
            await conversation.mount(
                MessageWidget(
                    MessageKind.OPERATION,
                    f"* {verb[1]} for {_fmt_duration(elapsed)}",
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
        color = random_accent_color()
        verb = random_operative_verb()
        conversation = self.query_one("#conversation", ScrollableContainer)
        try:
            explorer = self._agent.create_ws_explorer(self._working_dir, Mode.FULL)
            renderer: SubAgentRenderer | None = None
            async for event in explorer.run():
                if isinstance(event, SubAgentStartEvent):
                    await self._start_status_animation("scanning workspace...", color)
                    renderer = SubAgentRenderer(conversation)
                    await renderer.start(event.name)
                elif isinstance(event, LogEvent) and renderer:
                    await renderer.log(event.message)
                elif isinstance(event, InferStartEvent):
                    await self._start_status_animation(f"{verb[0]}...", color)
                elif isinstance(event, InferEndEvent):
                    if event.prompt_tokens is not None and event.completion_tokens is not None:
                        self._status_text = f"{verb[0]}... (↑ {event.prompt_tokens}  ↓ {event.completion_tokens})"
                elif isinstance(event, DoneEvent):
                    pass
            if explorer.workspace:
                self._workspace = explorer.workspace
            self._session = self._agent.start_session(self._workspace)
            conversation.scroll_end(animate=False)
        except Exception as error:
            await conversation.mount(MessageWidget(MessageKind.SYSTEM, f"error: {error}"))
            conversation.scroll_end(animate=False)
        finally:
            await self._stop_status_animation()
            self._worker = None
            self._focus_prompt()

    async def _animate_status(self, color: str, stop: asyncio.Event) -> None:
        try:
            while not stop.is_set():
                self._tick_status(self._status_text, color)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.15)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._clear_status()

    async def _start_status_animation(self, text: str, color: str) -> None:
        self._status_text = text
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

    def _tick_status(self, text: str, color: str) -> None:
        frame = _SPINNER_FRAMES[self._status_frame % len(_SPINNER_FRAMES)]
        self._status_frame += 1
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

    def action_cancel_stream(self) -> None:
        palette = self.query_one(CommandPalette)
        if palette.display:
            self.query_one("#prompt", Input).value = ""
            palette.hide()
            return
        if self._worker is not None and not self._worker.is_finished:
            self._worker.cancel()
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
