from __future__ import annotations

from pathlib import Path

from agent.settings import save_scope_gate

from .base import CommandResult


class ConfigGateCommand:
    name = "config:gate"
    description = "Enable or disable the scope gate  (on|off)"

    def __init__(self, working_dir: Path) -> None:
        self._working_dir = working_dir

    async def execute(self, args: list[str]) -> CommandResult:
        arg = args[0].lower() if args else ""
        if arg not in ("on", "off"):
            return CommandResult(output="usage: /config:gate on|off")
        enabled = arg == "on"
        save_scope_gate(self._working_dir, enabled)
        state = "on" if enabled else "off"
        return CommandResult(
            output=f"scope gate {state} — saved to project settings",
            scope_gate=enabled,
        )
