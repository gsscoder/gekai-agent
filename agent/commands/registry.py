from .base import Command, CommandResult


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        self._commands[command.name] = command

    async def dispatch(self, raw: str) -> CommandResult:
        parts = raw.lstrip("/").split()
        name = parts[0] if parts else ""
        args = parts[1:]

        if name not in self._commands:
            return CommandResult(output=f"unknown command: {name}")

        return await self._commands[name].execute(args)
