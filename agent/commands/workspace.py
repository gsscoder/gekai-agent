from .base import CommandResult


class WorkspaceRebuildCommand:
    name = "workspace:rebuild"
    description = "Rebuild workspace index and enrichment"

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult()
