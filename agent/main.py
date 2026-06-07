from __future__ import annotations

import argparse
from pathlib import Path

import sys

from . import __version__
from .agent import GekaiAgent
from .commands.clear import ClearCommand
from .commands.config import ConfigGateCommand
from .commands.exit import ExitCommand
from .commands.registry import CommandRegistry
from .persistence import load_session, load_timeline
from .settings import Permissions, bootstrap_global_settings, load_global_settings, load_permissions, load_scope_gate, validate_gate_config
from .tui.app import GekaiApp
from .shell import resolve_shell
from .workspace import get_git_branch


def main() -> None:
    bootstrap_global_settings()
    load_global_settings()
    resolve_shell()

    parser = argparse.ArgumentParser(prog="gekai")
    parser.add_argument(
        "-d", "--working-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="working directory (default: current directory)",
    )
    parser.add_argument("--debug", action="store_true", help="show intent classification")
    parser.add_argument(
        "-r", "--resume",
        metavar="SESSION_ID",
        help="resume a previous session by ID",
    )
    args = parser.parse_args()

    restored_id: str | None = None
    restored_messages: list[dict] | None = None
    restored_timeline: list[dict] | None = None

    if args.resume and args.working_dir is not None:
        print("error: --resume and --working-dir cannot be used together")
        raise SystemExit(1)

    if args.resume:
        result = load_session(args.resume)
        if result is None:
            print(f"session {args.resume} not found")
            raise SystemExit(1)
        restored_id, working_dir, restored_messages = result
        timeline_result = load_timeline(args.resume)
        if timeline_result is not None:
            _, restored_timeline = timeline_result
    else:
        working_dir = (args.working_dir or Path.cwd()).resolve()

    branch: str | None = get_git_branch(working_dir)

    permissions = load_permissions(working_dir)
    needs_permissions = permissions is None
    if needs_permissions:
        permissions = Permissions(read=False, write=False, exec=False)

    errors = validate_gate_config(working_dir)
    if errors:
        for e in errors:
            print(f"  config error: {e}")
        sys.exit(1)

    agent = GekaiAgent(working_dir=working_dir, permissions=permissions, debug=args.debug)

    registry = CommandRegistry()
    registry.register(ClearCommand())
    registry.register(ConfigGateCommand(working_dir))
    registry.register(ExitCommand())

    app = GekaiApp(
        agent=agent,
        registry=registry,
        working_dir=working_dir,
        version=__version__,
        branch=branch,
        restored_id=restored_id,
        restored_messages=restored_messages,
        restored_timeline=restored_timeline,
        needs_permissions=needs_permissions,
    )
    app.run()

    session_id = app.session_id
    if session_id and app.session_has_interactions:
        from rich.console import Console
        console = Console()
        console.print()
        console.print(f"[grey50]resume session:\ngekai --resume {session_id}[/grey50]")


if __name__ == "__main__":
    main()
