from dataclasses import dataclass
from typing import Protocol


@dataclass
class CommandResult:
    output: str | None = None
    exit_app: bool = False
    clear_session: bool = False
    error: bool = False  # render `output` as an error rather than a plain command result
    reconfigure: bool = False  # tier bindings changed — re-resolve touchpoints and refresh the status bar
    # A command whose real work needs live UI (a grid panel, a sequence of
    # choice screens) names that UI action here and the TUI runs it. The
    # command still owns its own name, argument parsing, and usage errors;
    # only the drawing belongs to the view. Empty = an ordinary command whose
    # `output` is the whole result.
    ui_action: str = ""
    ui_arg: str = ""


class Command(Protocol):
    name: str
    description: str
    params: str = ""
    # True for the commands that must stay reachable before model tiers are
    # configured — the ones that let the user configure them, or quit.
    works_unconfigured: bool = False

    async def execute(self, args: list[str]) -> CommandResult: ...
