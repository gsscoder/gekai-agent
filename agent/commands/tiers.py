from .base import CommandResult


class TiersCommand:
    name = "tiers"
    description = "View/edit model tier bindings (FAST/SUPP/CORE)"

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult()
