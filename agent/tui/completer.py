from __future__ import annotations

from textual.suggester import Suggester

from agent.commands.registry import CommandRegistry


class SlashCommandSuggester(Suggester):
    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry
        super().__init__(case_sensitive=False)

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        typed = value[1:].lower()
        for cmd in self._registry.commands():
            if cmd.name.lower().startswith(typed):
                return "/" + cmd.name
        return None
