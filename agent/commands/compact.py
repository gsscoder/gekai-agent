from .base import CommandResult


class CompactCommand:
    name = "compact"
    description = "Summarize the conversation to free up context"
    params = "<optional focus instructions>"

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult()
