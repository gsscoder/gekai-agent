from .base import CommandResult


class ModelsCommand:
    """Registered only so the command shows up in the palette — the TUI
    intercepts `/models` before dispatch and opens the grid panel itself
    (see `agent/tui/app.py::_open_models_panel`)."""

    name = "models"
    description = "Store the API key for each supported model"
    params = ""

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult()
