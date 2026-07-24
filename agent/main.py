from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .agent import GekaiAgent
from .commands.clear import ClearCommand
from .commands.exit import ExitCommand
from .commands.tiers import TiersCommand
from .commands.registry import CommandRegistry
from .persistence import load_session, load_timeline
from .settings import Permissions, bootstrap_global_settings, load_global_settings, load_permissions
from .tui.app import GekaiApp
from .shell import resolve_shell
from .workspace import get_git_branch


def _silence_proactor_pipe_del() -> None:
    """ponytail: known CPython/Windows bug (gh-83413) — ProactorEventLoop's
    subprocess pipe transports close via a deferred loop callback; if the
    app's own loop shuts down first (e.g. on /exit right after run_command),
    GC finalizes the transport later and its __del__ crashes formatting its
    own __repr__() on the already-OS-closed pipe, printed as a harmless but
    noisy "Exception ignored in __del__". Not reliably avoidable by winning
    the timing race per-callsite (see agent/tools/shell.py's own attempt) —
    this is the standard fix (used by httpx, aiohttp, etc.): swallow the
    crash in the finalizer itself so nothing prints."""
    if sys.platform != "win32":
        return
    from asyncio.proactor_events import _ProactorBasePipeTransport

    original_del = _ProactorBasePipeTransport.__del__

    def _safe_del(self: _ProactorBasePipeTransport) -> None:
        try:
            original_del(self)
        except Exception:
            pass

    _ProactorBasePipeTransport.__del__ = _safe_del


def main() -> None:
    _silence_proactor_pipe_del()
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

    subparsers = parser.add_subparsers(dest="command")
    dump_parser = subparsers.add_parser("dump", help="print session data and exit")
    dump_subparsers = dump_parser.add_subparsers(dest="scope")
    prompts_parser = dump_subparsers.add_parser("prompts", help="dump a session's user prompts")
    prompts_parser.add_argument("session_id", metavar="SESSION_ID")

    args = parser.parse_args()

    if args.command == "dump":
        if args.resume or args.working_dir is not None or args.debug:
            print("error: dump cannot be combined with other flags")
            raise SystemExit(1)
        if args.scope is None:
            parser.error("dump requires a scope (prompts)")
        from .dump import dump
        raise SystemExit(dump(args.scope, args.session_id))

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

    agent = GekaiAgent(working_dir=working_dir, permissions=permissions, debug=args.debug)

    registry = CommandRegistry()
    registry.register(ClearCommand())
    registry.register(ExitCommand())
    registry.register(TiersCommand())

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

    agent.events.emit(
        "run.exit",
        duration_s=agent.events.runtime_s(),
        turn_count=agent.events.turn_count,
        reason=app.exit_reason,
    )
    agent.events.close()

    session_id = app.session_id
    if session_id and app.session_has_interactions:
        from rich.console import Console

        console = Console()
        console.print()
        console.print(f"[grey50]resume session:\ngekai --resume {session_id}[/grey50]")


if __name__ == "__main__":
    main()
