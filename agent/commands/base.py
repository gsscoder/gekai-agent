from dataclasses import dataclass
from typing import Protocol


@dataclass
class CommandResult:
    output: str | None = None
    exit_app: bool = False
    clear_session: bool = False
    error: bool = False  # render `output` as an error rather than a plain command result
    reconfigure: bool = False  # tier bindings changed — re-resolve touchpoints and refresh the status bar


class Command(Protocol):
    name: str
    description: str
    params: str = ""

    async def execute(self, args: list[str]) -> CommandResult: ...
