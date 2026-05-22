import argparse
import asyncio
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .agent import GekaiAgent
from .commands.exit import ExitCommand
from .commands.registry import CommandRegistry
from .persistence import load_session, save_session
from .settings import load_permissions, prompt_permissions
from .handlers.chat import UsageInfo
from .ui import console, make_spinner_display, print_banner, random_operative_verb, render_operation_summary, render_response, restyle_user_input
from .workspace import get_git_branch, scan_workspace


def _block_until_esc(stop: threading.Event) -> None:
    if sys.platform == "win32":
        import msvcrt
        while not stop.is_set():
            if msvcrt.kbhit() and msvcrt.getch() == b"\x1b":
                return
            time.sleep(0.05)
    else:
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not stop.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0] and sys.stdin.read(1) == "\x1b":
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def _run(working_dir: Path, debug: bool = False, resume_id: str | None = None) -> None:
    branch = get_git_branch(working_dir)
    print_banner(__version__, working_dir, branch)

    permissions = load_permissions(working_dir)
    if permissions is None:
        permissions = await prompt_permissions(working_dir)
        if permissions is None:
            console.print("\n[dim]access denied, exiting[/dim]")
            return

    agent = GekaiAgent(working_dir=working_dir, permissions=permissions, debug=debug)

    restored_id = None
    restored_messages = None
    if resume_id:
        result = load_session(resume_id, working_dir)
        if result is None:
            console.print(f"[yellow]session {resume_id} not found, starting fresh[/yellow]")
        else:
            restored_id, restored_messages = result

    cache_path = working_dir / ".gekai" / "workspace.json"
    max_age = 30 * 60 if restored_id else 15 * 60
    cache_age = (
        (datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime)
        if cache_path.exists() else float("inf")
    )
    if cache_age < max_age:
        workspace = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("initializing workspace", total=None)
            workspace = scan_workspace(working_dir, on_step=lambda msg: progress.update(task, description=msg))

    session = agent.start_session(workspace, restored_messages=restored_messages, session_id=restored_id)

    if restored_messages:
        console.print("[dim]— resuming session —[/dim]\n")
        for msg in restored_messages:
            if msg["role"] == "user":
                console.print(f"[bold cyan]❯[/bold cyan] {msg['content']}")
            else:
                render_response(msg["content"])
                console.print()

    registry = CommandRegistry()
    registry.register(ExitCommand())

    pt_session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    prompt_message = FormattedText([("ansicyan bold", "❯ ")])

    while True:
        try:
            user_input = await pt_session.prompt_async(prompt_message)
        except (EOFError, KeyboardInterrupt):
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        restyle_user_input(stripped)

        if stripped.startswith("/"):
            result = await registry.dispatch(stripped)
            if result.output:
                console.print(result.output)
            if result.exit_app:
                break
            continue

        try:
            start = time.monotonic()
            stop_esc = threading.Event()
            loop = asyncio.get_running_loop()
            verb = random_operative_verb()
            state: dict[str, object] = {"frame": 0, "tokens": 0, "usage": None}
            chunks: list[str] = []

            async def _interact(live: Live) -> None:
                segments = await agent.classify(stripped)
                if debug:
                    for intent, sub in segments:
                        console.print(f"[grey50]debug: {intent.value}: {sub}[/grey50]")
                async for item in agent.process_stream(session, stripped, segments):
                    if isinstance(item, UsageInfo):
                        state["usage"] = item
                        continue
                    chunks.append(item)
                    state["tokens"] += 1
                    live.update(make_spinner_display(state["frame"], state["tokens"], verb))

            with Live(console=console, refresh_per_second=12, transient=True) as live:
                live.update(make_spinner_display(state["frame"], state["tokens"], verb))

                async def _animate() -> None:
                    while True:
                        await asyncio.sleep(0.1)
                        state["frame"] += 1
                        live.update(make_spinner_display(state["frame"], state["tokens"], verb))

                spinner_task = asyncio.create_task(_animate())
                interact_task = asyncio.create_task(_interact(live))
                esc_future = loop.run_in_executor(None, _block_until_esc, stop_esc)

                done, _ = await asyncio.wait(
                    {interact_task, esc_future},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                stop_esc.set()
                spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass

            if interact_task in done:
                interact_task.result()
                usage = state["usage"]
                render_operation_summary(
                    time.monotonic() - start,
                    verb,
                )
                console.print()
                render_response("".join(chunks))
                save_session(session)
                console.print()
            else:
                interact_task.cancel()
                try:
                    await interact_task
                except asyncio.CancelledError:
                    pass
                console.print("\n[dim]Cancelled — What should Gekai do instead?[/dim]")

        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")

    console.print()
    render_response("Goodbye.")
    console.print()
    console.print(f"[grey50]Resume this session with:\n  gekai --resume {session.id}[/grey50]")


def main() -> None:
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
    asyncio.run(_run(working_dir=args.working_dir.resolve(), debug=args.debug, resume_id=args.resume))


if __name__ == "__main__":
    main()
