from .base import CommandResult


class ModelsCommand:
    """Opens the model grid, where a key is stored per model. The grid itself
    is live UI, so this returns the `models` UI action and the TUI draws it
    (`agent/tui/app.py::_open_models_panel`)."""

    name = "models"
    description = "Store the API key for each supported model"
    params = ""
    works_unconfigured = True  # must stay reachable before tiers are configured

    async def execute(self, args: list[str]) -> CommandResult:
        return CommandResult(ui_action="models")
