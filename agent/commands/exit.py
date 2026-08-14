from .base import CommandResult


class ExitCommand:
    name = "exit"
    description = "Exit to terminal"
    params = ""

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult(exit_app=True)
