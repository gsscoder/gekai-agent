from .base import CommandResult


class CompactCommand:
    """Summarizing needs the live session and mounts its own progress into the
    conversation, so this returns the `compact` UI action carrying the
    optional focus instructions."""

    name = "compact"
    description = "Summarize the conversation to free up context"
    params = "<optional focus instructions>"

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult(ui_action="compact", ui_arg=" ".join(args).strip())
