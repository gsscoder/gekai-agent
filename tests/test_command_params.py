"""Coverage for the "param hint" feature's data model: `Command.params` and
`Subagent.params` declarations, and `CommandRegistry.get()`.

Source: agent/commands/base.py (`Command.params: str = ""`), agent/commands/
clear.py, agent/commands/compact.py, agent/commands/exit.py, agent/commands/
tiers.py (concrete `params` values), agent/subagents/__init__.py
(`Subagent.params: str = "<subagent prompt>"` default), agent/commands/
registry.py (`CommandRegistry.get`).

No gaps/assumptions — these are plain attribute/behavior checks against
existing declared values.
"""

from __future__ import annotations

from agent.commands.clear import ClearCommand
from agent.commands.compact import CompactCommand
from agent.commands.exit import ExitCommand
from agent.commands.registry import CommandRegistry
from agent.commands.tiers import TiersCommand
from agent.subagents import Subagent


# REQ-001: ClearCommand declares no params (per agent/commands/clear.py)
def test_clear_command_has_no_params() -> None:
    assert ClearCommand.params == ""


# REQ-002: ExitCommand declares no params (per agent/commands/exit.py)
def test_exit_command_has_no_params() -> None:
    assert ExitCommand.params == ""


# REQ-003: TiersCommand declares no params (per agent/commands/tiers.py)
def test_tiers_command_has_no_params() -> None:
    assert TiersCommand.params == ""


# REQ-004: CompactCommand declares an optional param spec (per
# agent/commands/compact.py)
def test_compact_command_declares_optional_focus_instructions_param() -> None:
    assert CompactCommand.params == "<optional focus instructions>"


# REQ-005: Subagent.params defaults to "<subagent prompt>" when not
# overridden (per agent/subagents/__init__.py's dataclass field default)
def test_subagent_default_params_is_subagent_prompt() -> None:
    default_subagent = Subagent(name="probe", namespace="generic", description="d")
    assert default_subagent.params == "<subagent prompt>"


# REQ-006: CommandRegistry.get returns the registered command instance for a
# known name (per agent/commands/registry.py's `get`)
def test_registry_get_returns_registered_command() -> None:
    registry = CommandRegistry()
    clear_command = ClearCommand()
    registry.register(clear_command)

    assert registry.get("clear") is clear_command


# REQ-007: CommandRegistry.get returns None for an unregistered name (per
# agent/commands/registry.py's `dict.get` fallback)
def test_registry_get_returns_none_for_unknown_command() -> None:
    registry = CommandRegistry()

    assert registry.get("nonexistent") is None
