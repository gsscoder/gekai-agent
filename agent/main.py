from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .agent import GekaiAgent
from .commands.clear import ClearCommand
from .commands.exit import ExitCommand
from .commands.registry import CommandRegistry
from .commands.workspace import WorkspaceRebuildCommand
from .persistence import load_session
from .settings import Permissions, bootstrap_global_settings, load_global_settings, load_permissions
from .tui.app import GekaiApp
from .workspace import get_git_branch


def main() -> None:
    bootstrap_global_settings()
    load_global_settings()

    parser = argparse.ArgumentParser(prog="gekai")
    parser.add_argument(
        "-d", "--working-dir",
        type=Path,
        default=Path.cwd(),
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

    working_dir: Path = args.working_dir.resolve()
    branch: str | None = get_git_branch(working_dir)

    permissions = load_permissions(working_dir)
    needs_permissions = permissions is None
    if needs_permissions:
        permissions = Permissions(read=False, write=False)

    agent = GekaiAgent(working_dir=working_dir, permissions=permissions, debug=args.debug)

    restored_id: str | None = None
    restored_messages: list[dict] | None = None
    if args.resume:
        result = load_session(args.resume, working_dir)
        if result is None:
            print(f"session {args.resume} not found, starting fresh")
        else:
            restored_id, restored_messages = result

    registry = CommandRegistry()
    registry.register(ClearCommand())
    registry.register(ExitCommand())
    registry.register(WorkspaceRebuildCommand())

    app = GekaiApp(
        agent=agent,
        registry=registry,
        working_dir=working_dir,
        version=__version__,
        branch=branch,
        restored_id=restored_id,
        restored_messages=restored_messages,
        needs_permissions=needs_permissions,
    )
    app.run()

    session_id = app.session_id
    if session_id and app.session_has_interactions:
        from .ui import console
        console.print()
        console.print(f"[dim]Resume session:[/dim]\n[grey50]gekai --resume {session_id}[/grey50]")


if __name__ == "__main__":
    main()
