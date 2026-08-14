from .base import CommandResult


class ClearCommand:
    name = "clear"
    description = "Clear chat and start a new session"
    params = ""

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult(clear_session=True)
