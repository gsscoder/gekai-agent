from .base import CommandResult


# [dead code] command removed from registry — file kept as reference
class WorkspaceRebuildCommand:
    name = "workspace:rebuild"
    description = "Rebuild workspace index and enrichment"

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult()
