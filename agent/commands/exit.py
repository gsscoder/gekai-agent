from .base import CommandResult


class ExitCommand:
    name = "exit"
    description = "Exit to terminal"
    params = ""
    works_unconfigured = True  # the user must always be able to quit

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult(exit_app=True)
