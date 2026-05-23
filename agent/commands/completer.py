from __future__ import annotations

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from .registry import CommandRegistry


class SlashCommandCompleter(Completer):
    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        typed = text[1:]
        for cmd in self._registry.commands():
            if cmd.name.startswith(typed):
                yield Completion(
                    text=cmd.name,
                    start_position=-len(typed),
                    display=f"/{cmd.name}",
                    display_meta=cmd.description,
                )
