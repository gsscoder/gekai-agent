from dataclasses import dataclass
from typing import Protocol


@dataclass
class CommandResult:
    output: str | None = None
    exit_app: bool = False


class Command(Protocol):
    name: str
    description: str

    async def execute(self, args: list[str]) -> CommandResult: ...
