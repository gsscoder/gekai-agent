from __future__ import annotations

from textual import events
from textual.widgets import Static

from agent.commands.registry import CommandRegistry


class CommandPalette(Static):
    DEFAULT_CSS = """
    CommandPalette {
        display: none;
        height: auto;
        background: ansi_default;
        padding: 0 2;
    }
    """

    def __init__(
        self,
        registry: CommandRegistry,
        subagents: list[tuple[str, str]] | None = None,
        **kwargs: object,
    ) -> None:
        self._registry = registry
        self._subagents: list[tuple[str, str]] = subagents if subagents is not None else []
        self._items: list[tuple[str, str, bool]] = []
        self._selected: int = 0
        cmd_max = max((len(cmd.name) for cmd in registry.commands()), default=0)
        sub_max = max((len(name) for name, _ in self._subagents), default=0)
        self._max_name_len = max(cmd_max, sub_max)
        super().__init__(**kwargs)  # type: ignore[arg-type]

    @property
    def selected_name(self) -> str | None:
        if not self._items:
            return None
        return self._items[self._selected][0]

    @property
    def selected_is_subagent(self) -> bool:
        if not self._items:
            return False
        return self._items[self._selected][2]

    def filter(self, typed: str) -> None:
        typed_lower = typed.lower()
        commands: list[tuple[str, str, bool]] = [
            (cmd.name, cmd.description, False)
            for cmd in self._registry.commands()
            if cmd.name.lower().startswith(typed_lower)
        ]
        subagents: list[tuple[str, str, bool]] = [
            (name, desc, True)
            for name, desc in self._subagents
            if name.lower().startswith(typed_lower)
        ]
        self._items = commands + subagents
        self._selected = 0
        self._refresh_display()

    def move_up(self) -> None:
        if self._items:
            self._selected = (self._selected - 1) % len(self._items)
            self._refresh_display()

    def move_down(self) -> None:
        if self._items:
            self._selected = (self._selected + 1) % len(self._items)
            self._refresh_display()

    def hide(self) -> None:
        self.display = False

    def on_click(self, event: events.Click) -> None:
        if not self._items:
            return
        idx = event.y
        if 0 <= idx < len(self._items):
            self.app.action_select_command(self._items[idx][0])  # type: ignore[attr-defined]
        event.stop()

    def _refresh_display(self) -> None:
        if not self._items:
            self.display = False
            return
        lines: list[str] = []
        for i, (name, desc, is_subagent) in enumerate(self._items):
            padded = f"/{name}".ljust(self._max_name_len + 3)
            if is_subagent:
                if i == self._selected:
                    lines.append(f"[bold #ffa500]❯ {padded}{desc}[/bold #ffa500]")
                else:
                    lines.append(f"[#ffa500]  {padded}{desc}[/#ffa500]")
            else:
                if i == self._selected:
                    lines.append(f"[bold white]❯ {padded}{desc}[/bold white]")
                else:
                    lines.append(f"[dim]  {padded}{desc}[/dim]")
        self.update("\n".join(lines))
        self.display = True
