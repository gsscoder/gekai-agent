from __future__ import annotations

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

    def __init__(self, registry: CommandRegistry, **kwargs: object) -> None:
        self._registry = registry
        self._items: list[tuple[str, str]] = []
        self._selected: int = 0
        self._max_name_len = max(
            (len(cmd.name) for cmd in registry.commands()), default=0
        )
        super().__init__(**kwargs)

    @property
    def selected_command(self) -> str | None:
        if not self._items:
            return None
        return self._items[self._selected][0]

    def filter(self, typed: str) -> None:
        typed_lower = typed.lower()
        self._items = [
            (cmd.name, cmd.description)
            for cmd in self._registry.commands()
            if cmd.name.lower().startswith(typed_lower)
        ]
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

    def _refresh_display(self) -> None:
        if not self._items:
            self.display = False
            return
        lines = []
        for i, (name, desc) in enumerate(self._items):
            padded = f"/{name}".ljust(self._max_name_len + 3)
            if i == self._selected:
                lines.append(f"[bold cyan]❯ {padded}{desc}[/bold cyan]")
            else:
                lines.append(f"[dim]  {padded}{desc}[/dim]")
        self.update("\n".join(lines))
        self.display = True
